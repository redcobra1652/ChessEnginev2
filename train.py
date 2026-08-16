"""
train.py -- Training loop for NNUE.

Key changes vs original:
  - Reservoir is saved to disk after filling and reloaded on subsequent runs,
    avoiding re-streaming all PGNs (can take 10-30 min for 400k games).
  - Reservoir is re-filled from PGNs every --refill-every epochs so later
    epochs don't overfit the same fixed sample.
  - --reservoir-cache controls the cache path (default: reservoir.pkl.gz
    next to --out).
  - --no-cache disables both saving and loading (useful for debugging).
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn

import data
import model as M
import convert as C

WDL_SCALE = 410.0   # centipawns → WDL sigmoid scale (matches Stockfish training)


# ── Loss ─────────────────────────────────────────────────────────────────────

def cp_to_wdl(cp: torch.Tensor) -> torch.Tensor:
    """Convert centipawn score to WDL probability via sigmoid."""
    return torch.sigmoid(cp / WDL_SCALE)


def nnue_loss(output: torch.Tensor,
              score_cp: torch.Tensor,
              wdl: torch.Tensor,
              lam: float = 0.7) -> torch.Tensor:
    """
    Blended MSE loss between predicted WDL and a mixture of:
      - engine score converted to WDL  (weight lam)
      - actual game result             (weight 1 - lam)

    output   : raw network logits in centipawn-scale (before sigmoid)
    score_cp : engine eval in centipawns, side-to-move relative
    wdl      : game result in [0, 1], side-to-move relative
    """
    pred   = torch.sigmoid(output / WDL_SCALE)
    target = lam * cp_to_wdl(score_cp) + (1.0 - lam) * wdl
    return nn.functional.mse_loss(pred, target)


# ── Reservoir helpers ─────────────────────────────────────────────────────────

def _default_cache_path(out_path: str) -> str:
    """Place the cache file next to the checkpoint with a .pkl.gz extension."""
    base = os.path.splitext(out_path)[0]
    return base + "_reservoir.pkl.gz"


def fill_reservoir(reservoir: data.Reservoir,
                   pgn_paths: list[str],
                   pawn_units: bool = True,
                   num_buckets: int = 8) -> None:
    """Stream all PGN positions into the reservoir, printing progress."""
    t0 = time.time()
    for i, item in enumerate(
        data.iter_positions(pgn_paths, pawn_units=pawn_units, num_buckets=num_buckets)
    ):
        reservoir.add(item)
        if (i + 1) % 100_000 == 0:
            elapsed = time.time() - t0
            print(f"  streamed {i+1:,} positions  |  reservoir={len(reservoir):,}"
                  f"  |  {elapsed:.0f}s", flush=True)
    print(f"  done — {reservoir.total_seen:,} seen, "
          f"{len(reservoir):,} in reservoir  ({time.time()-t0:.1f}s)", flush=True)


def get_reservoir(args, pgn_paths: list[str]) -> data.Reservoir:
    """
    Return a filled Reservoir, loading from cache if available.

    Logic:
      1. If --no-cache: always stream fresh.
      2. If cache file exists and capacity matches: load it.
      3. Otherwise: stream PGNs, then save to cache.
    """
    cache_path = args.reservoir_cache or _default_cache_path(args.out)

    if not args.no_cache and os.path.exists(cache_path):
        try:
            reservoir = data.Reservoir.load(cache_path)
            if reservoir.capacity == args.reservoir:
                return reservoir
            print(f"  Cache capacity mismatch "
                  f"({reservoir.capacity:,} ≠ {args.reservoir:,}) — re-streaming.", flush=True)
        except (ValueError, Exception) as e:
            print(f"  Could not load cache ({e}) — re-streaming.", flush=True)

    print(f"\nFilling reservoir (capacity={args.reservoir:,}) …", flush=True)
    reservoir = data.Reservoir(args.reservoir)
    fill_reservoir(reservoir, pgn_paths, pawn_units=args.pawn_units, num_buckets=args.buckets)

    if not args.no_cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
        reservoir.save(cache_path)

    return reservoir


# ── Binary streaming training loop ───────────────────────────────────────────

def train_one_epoch_bin(net: nn.Module,
                        optimizer: torch.optim.Optimizer,
                        bin_path: str,
                        batch_size: int,
                        steps_per_epoch: int,
                        device: torch.device,
                        lam: float = 0.7) -> float:
    """
    Train one epoch by streaming directly from a .bin file.
    Bypasses the reservoir entirely — no RAM overhead beyond one batch at a time.
    ~400-600k pos/s vs ~22k/s from PGN.

    Expects iter_positions_bin to yield 6-tuples:
        (white_indices, black_indices, score_cp, wdl, bucket, stm_white)
    """
    import numpy as np

    net.train()
    total_loss = 0.0
    steps_done = 0

    wi_buf, bi_buf, cp_buf, wdl_buf, b_buf, stm_buf = [], [], [], [], [], []

    for wi, bi, cp, wdl, bucket, stm in C.iter_positions_bin(bin_path, shuffle=True):
        wi_buf.append(wi)
        bi_buf.append(bi)
        cp_buf.append(cp)
        wdl_buf.append(wdl)
        b_buf.append(bucket)
        stm_buf.append(stm)

        if len(wi_buf) < batch_size:
            continue

        wi_t   = torch.from_numpy(np.stack(wi_buf)).long().to(device)
        bi_t   = torch.from_numpy(np.stack(bi_buf)).long().to(device)
        cp_t   = torch.tensor(cp_buf,  dtype=torch.float32).to(device)
        wdl_t  = torch.tensor(wdl_buf, dtype=torch.float32).to(device)
        b_t    = torch.tensor(b_buf,   dtype=torch.int64).to(device)
        stm_t  = torch.tensor(stm_buf, dtype=torch.bool).to(device)

        optimizer.zero_grad(set_to_none=True)
        out  = net(wi_t, bi_t, bucket=b_t, stm_white=stm_t)
        loss = nnue_loss(out, cp_t, wdl_t, lam=lam)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        steps_done += 1
        wi_buf, bi_buf, cp_buf, wdl_buf, b_buf, stm_buf = [], [], [], [], [], []

        if steps_done >= steps_per_epoch:
            break

    return total_loss / max(steps_done, 1)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(net: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    reservoir: data.Reservoir,
                    batch_size: int,
                    steps_per_epoch: int,
                    device: torch.device,
                    lam: float = 0.7) -> float:
    net.train()
    total_loss = 0.0

    for _ in range(steps_per_epoch):
        wi, bi, cp, wdl, buckets, stm = reservoir.sample_batch(batch_size)
        wi      = wi.to(device)
        bi      = bi.to(device)
        cp      = cp.to(device)
        wdl     = wdl.to(device)
        buckets = buckets.to(device)
        stm     = stm.to(device)

        optimizer.zero_grad(set_to_none=True)
        out  = net(wi, bi, bucket=buckets, stm_white=stm)
        loss = nnue_loss(out, cp, wdl, lam=lam)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / steps_per_epoch


# ── LR schedule ───────────────────────────────────────────────────────────────

def make_scheduler(optimizer, total_epochs: int, start_epoch: int = 0, warmup: int = 2):
    """
    Linear warmup for `warmup` epochs then cosine decay to 5% of peak LR.

    total_epochs : full training horizon (not just this run's --epochs)
    start_epoch  : epoch we're resuming from; used to position correctly in the
                   cosine curve so a resumed run continues smoothly instead of
                   restarting warmup.

    Common mistake: passing args.epochs (this run) instead of the full horizon.
    If you train for 20 epochs, stop, and resume for 20 more, total_epochs
    should be 40 both times so the LR curve is consistent.
    """
    floor = 0.05  # 5% of peak — keeps updates meaningful in late training
    def lr_lambda(epoch):
        e = epoch + start_epoch  # absolute position in the full schedule
        if e < warmup:
            return (e + 1) / max(warmup, 1)
        progress = (e - warmup) / max(1, total_epochs - warmup)
        progress  = min(progress, 1.0)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(net, optimizer, epoch, path):
    torch.save({
        "epoch":      epoch,
        "state_dict": net.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }, path)


def load_checkpoint(path, net, optimizer=None) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["state_dict"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0)


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train simple NNUE from PGN files")

    # Data
    p.add_argument("--pgn",         nargs="+", default=None,
                   help="PGN file(s) or directory/glob")
    p.add_argument("--bin",         default=None,
                   help="Pre-converted .bin file (from convert.py). "
                        "Skips reservoir entirely; ~20x faster data loading. "
                        "Cannot be combined with --pgn.")
    p.add_argument("--pawn-units",  action="store_true", default=True,
                   help="PGN %%eval is in pawn units (default=True; most engines)")
    p.add_argument("--no-pawn-units", dest="pawn_units", action="store_false",
                   help="PGN %%eval is in raw centipawns")

    # Reservoir
    p.add_argument("--reservoir",       type=int, default=2_000_000,
                   help="Reservoir capacity (positions)")
    p.add_argument("--reservoir-cache", default=None,
                   help="Path for reservoir cache (.pkl.gz). "
                        "Default: <out>_reservoir.pkl.gz")
    p.add_argument("--no-cache",        action="store_true",
                   help="Disable reservoir disk cache (always re-stream)")
    p.add_argument("--refill-every",    type=int, default=0,
                   help="Re-stream PGNs and rebuild reservoir every N epochs. "
                        "0 = never refill (use same reservoir throughout). "
                        "Recommended: 5 for large datasets.")

    # Training
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--total-epochs", type=int,   default=None,
                   help="Full training horizon for the LR schedule. Defaults to "
                        "--epochs. Set this when resuming so the cosine curve "
                        "doesn't restart: e.g. first run --epochs 20 --total-epochs 40, "
                        "resume with --epochs 20 --total-epochs 40 --resume ckpt.pt")
    p.add_argument("--batch-size",  type=int,   default=8192)
    p.add_argument("--steps",       type=int,   default=None,
                   help="Steps per epoch. Default: reservoir_size // batch_size")
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--lambda",      type=float, default=0.7, dest="lam",
                   help="WDL blend: 1.0 = pure engine score, 0.0 = pure game result")
    p.add_argument("--wd",          type=float, default=1e-5,
                   help="AdamW weight decay")

    # Model
    p.add_argument("--hidden",      type=int,   default=256)
    p.add_argument("--buckets",     type=int,   default=8)

    # I/O
    p.add_argument("--out",         required=True,
                   help="Output checkpoint path (.pt)")
    p.add_argument("--resume",      default=None,
                   help="Resume from this checkpoint")
    p.add_argument("--save-every",  type=int,   default=1,
                   help="Save checkpoint every N epochs")
    p.add_argument("--device",      default="auto")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    # ── Validate data source ──────────────────────────────────────────────────
    if args.bin and args.pgn:
        raise SystemExit("Error: --bin and --pgn are mutually exclusive.")
    if not args.bin and not args.pgn:
        raise SystemExit("Error: one of --bin or --pgn is required.")

    use_bin = bool(args.bin)

    # ── PGN paths (only when not using .bin) ──────────────────────────────────
    if not use_bin:
        pgn_paths = data.resolve_pgn_paths(args.pgn)
        print(f"PGN files ({len(pgn_paths)}): {pgn_paths[:3]}"
              + (" …" if len(pgn_paths) > 3 else ""))
    else:
        pgn_paths = []
        n_pos = C._read_header(args.bin)
        print(f"Binary file: {args.bin}  ({n_pos:,} positions)")

    # ── Model ─────────────────────────────────────────────────────────────────
    net = M.NNUE(hidden_size=args.hidden, num_buckets=args.buckets).to(device)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Model: {total_params:,} parameters  "
          f"(hidden={args.hidden}, buckets={args.buckets})")

    # Create output directory early so reservoir cache save doesn't fail
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(args.resume, net, optimizer)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    total_epochs = args.total_epochs or (start_epoch + args.epochs)
    # Scheduler is constructed with the absolute position already baked in via
    # start_epoch, so no manual fast-forward is needed — the lambda offsets by
    # start_epoch internally and we always call scheduler.step() once per epoch.
    scheduler = make_scheduler(optimizer, total_epochs, start_epoch=start_epoch)

    # ── Reservoir (PGN mode) or bin header (binary mode) ─────────────────────
    if use_bin:
        reservoir = None
        steps = args.steps or max(1, n_pos // args.batch_size)
    else:
        reservoir = get_reservoir(args, pgn_paths)
        steps = args.steps or max(1, len(reservoir) // args.batch_size)

    print(f"\nTraining  epochs={args.epochs}  steps/epoch={steps}  "
          f"batch={args.batch_size}  lam={args.lam}\n", flush=True)

    # ── Epoch loop ────────────────────────────────────────────────────────────
    best_loss = float("inf")
    for epoch in range(start_epoch, start_epoch + args.epochs):

        # Re-fill reservoir periodically (PGN mode only).
        if (not use_bin
                and args.refill_every > 0
                and epoch > start_epoch
                and (epoch - start_epoch) % args.refill_every == 0):
            print(f"\n[epoch {epoch+1}] Re-filling reservoir …", flush=True)
            reservoir = data.Reservoir(args.reservoir)
            fill_reservoir(reservoir, pgn_paths,
                           pawn_units=args.pawn_units, num_buckets=args.buckets)
            if not args.no_cache:
                cache_path = args.reservoir_cache or _default_cache_path(args.out)
                reservoir.save(cache_path)
            steps = args.steps or max(1, len(reservoir) // args.batch_size)

        t0 = time.time()
        if use_bin:
            loss = train_one_epoch_bin(net, optimizer, args.bin, args.batch_size,
                                       steps, device, lam=args.lam)
        else:
            loss = train_one_epoch(net, optimizer, reservoir, args.batch_size,
                                   steps, device, lam=args.lam)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f"epoch {epoch+1:4d}  loss={loss:.6f}  "
              f"lr={lr_now:.2e}  time={elapsed:.1f}s", flush=True)

        is_last  = (epoch == start_epoch + args.epochs - 1)
        save_now = ((epoch + 1) % args.save_every == 0) or is_last
        if save_now:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
            save_checkpoint(net, optimizer, epoch + 1, args.out)

            if loss < best_loss:
                best_loss = loss
                best_path = args.out.replace(".pt", "_best.pt")
                save_checkpoint(net, optimizer, epoch + 1, best_path)
                print(f"  ✓ new best  loss={best_loss:.6f}  → {best_path}", flush=True)


if __name__ == "__main__":
    main()
