#!/usr/bin/env python3
"""
net_eval_diagnostic.py -- Held-out generalization check for checkpoints/model.nnue.

Compares the NNUE's *static* eval (no search, via the engine's `eval` UCI
debug command) against a blended ground-truth target on positions sampled
from games/*.pgn -- games the engine actually played after training, not
positions from positions.bin. These are independent of the Lichess-cloud-eval
training set, which is the point: it's a real (if imperfect -- see caveats
printed at the end) signal on whether the net generalizes, which the 0.0073
training-set MSE in train.py cannot answer alone.

Ground truth per position is target = lam * cp_to_wdl(sf_cp) + (1-lam) * wdl_result,
i.e. Stockfish's search score blended with the actual game's result the way
train.py's nnue_loss() does (lam defaults to train.py's --lambda default,
0.7). This is NOT the same as "compare directly to 0.0073" -- the position
DISTRIBUTION here (self-play endgames/tactics from games this engine played)
is not the training distribution (Lichess cloud-eval positions, mostly
opening/middlegame). Both the blend match and the distribution mismatch are
handled explicitly below rather than glossed over.

Also reports:
  - a per-material-bucket breakdown (same bucket formula as data.py / the
    engine), to separate "net is weak everywhere" from "net is weak only in
    undertrained buckets" -- these have very different implications.
  - a linear regression of net_cp on sf_cp, to catch a systematic centipawn
    *scale* mismatch (e.g. if Stockfish's internal cp normalization doesn't
    match whatever scale the Lichess cloud-eval labels used) before treating
    MAE/correlation as meaningful.

This does NOT produce an Elo number on its own -- there is no trustworthy
eval-RMSE-to-Elo calibration curve for this architecture on record, and
fabricating one would be exactly the kind of unfounded claim this script
exists to avoid. Read the numbers as "does the net generalize, and where is
it weak," and combine with actual measured game results (tournament.py) for
strength.

Usage:
    python3 net_eval_diagnostic.py [--samples 800] [--sf-movetime 300] [--lam 0.7]
"""
import argparse
import glob
import math
import os
import random
import subprocess
import time

import chess
import chess.pgn

BASE       = os.path.expanduser("~/Documents/ChessModelv2/nnue")
ENGINE     = os.path.join(BASE, "nnue_engine")
NNUE_FILE  = os.path.join(BASE, "checkpoints/model.nnue")
SF         = os.path.join(BASE, "stockfish/stockfish-macos-m1-apple-silicon")
PGN_GLOB   = os.path.join(BASE, "games/*.pgn")

WDL_SCALE   = 410.0  # matches train.py's WDL_SCALE
CP_CLAMP    = 1500   # clamp before the sigmoid so mate-adjacent scores don't saturate
NUM_BUCKETS = 8       # matches data.py / nnue_engine.cpp -- must stay in lockstep


def cp_to_wdl(cp: float) -> float:
    cp = max(-CP_CLAMP, min(CP_CLAMP, cp))
    return 1.0 / (1.0 + math.exp(-cp / WDL_SCALE))


def bucket_of(board: chess.Board) -> int:
    """Same formula as data.py:62 and nnue_engine.cpp:1910-1913:
    non-king piece count * num_buckets // 32, clamped."""
    n = sum(1 for pt in chess.PIECE_TYPES if pt != chess.KING
            for _ in board.pieces(pt, chess.WHITE)) + \
        sum(1 for pt in chess.PIECE_TYPES if pt != chess.KING
            for _ in board.pieces(pt, chess.BLACK))
    return min(NUM_BUCKETS - 1, max(0, n * NUM_BUCKETS // 32))


class UciEngine:
    def __init__(self, path, opts=None):
        self.p = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        for name, value in (opts or {}).items():
            self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def _wait_for(self, token, timeout=10):
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = self.p.stdout.readline()
            if token in line:
                return line
        raise TimeoutError(f"never saw {token!r}")

    def static_eval_cp(self, fen, timeout=10):
        """nnue_engine only: raw static eval via the `eval` debug command,
        side-to-move relative, no search."""
        self._send(f"position fen {fen}")
        self._send("eval")
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = self.p.stdout.readline()
            if line.startswith("eval "):
                return int(line.split()[1])
            if line.startswith("info string no net loaded"):
                raise RuntimeError("net not loaded")
        raise TimeoutError("no eval response")

    def search_score_cp(self, fen, movetime_ms, timeout=30):
        """Stockfish only: search score via go movetime N, side-to-move
        relative. Mate scores are mapped to +/-CP_CLAMP."""
        self._send(f"position fen {fen}")
        self._send(f"go movetime {movetime_ms}")
        last_score = None
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = self.p.stdout.readline()
            if not line:
                break
            if line.startswith("info") and " score " in line:
                parts = line.split()
                for i, tok in enumerate(parts):
                    if tok == "score":
                        kind = parts[i + 1]
                        val = int(parts[i + 2])
                        if kind == "cp":
                            last_score = val
                        elif kind == "mate":
                            last_score = CP_CLAMP if val > 0 else -CP_CLAMP
            if line.startswith("bestmove"):
                break
        return last_score

    def quit(self):
        try:
            self._send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def sample_positions(n_target, seed=0):
    """Sample (fen, wdl_result_stm, bucket) from games/*.pgn -- real games
    played after training, independent of positions.bin. wdl_result_stm is
    the actual game outcome (1.0 win / 0.5 draw / 0.0 loss) from the
    perspective of the side to move at that sampled position -- this is the
    "wdl" term train.py's loss blends in. Skips in-check positions (engine
    skips the NNUE forward pass in check) and the first few opening plies."""
    rng = random.Random(seed)
    pgn_files = sorted(glob.glob(PGN_GLOB))
    if not pgn_files:
        raise SystemExit(f"No PGN files found at {PGN_GLOB}")

    candidates = []
    for path in pgn_files:
        with open(path) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                result = game.headers.get("Result", "*")
                if result == "1-0":
                    wdl_white = 1.0
                elif result == "0-1":
                    wdl_white = 0.0
                elif result == "1/2-1/2":
                    wdl_white = 0.5
                else:
                    continue  # unfinished / unknown result -- skip, no ground truth

                board = game.board()
                ply = 0
                for move in game.mainline_moves():
                    board.push(move)
                    ply += 1
                    if ply < 10:
                        continue
                    if board.is_check():
                        continue
                    wdl_stm = wdl_white if board.turn == chess.WHITE else (1.0 - wdl_white)
                    candidates.append((board.fen(), wdl_stm, bucket_of(board)))

    rng.shuffle(candidates)
    return candidates[:n_target]


def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")


def linreg(xs, ys):
    """Least-squares slope/intercept of ys ~ xs. slope far from 1.0 (with
    intercept far from 0) flags a systematic cp-scale mismatch between the
    two engines' centipawn units."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den if den > 0 else float("nan")
    intercept = my - slope * mx
    return slope, intercept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=800)
    ap.add_argument("--sf-movetime", type=int, default=300,
                     help="ms per position for Stockfish's ground-truth search score")
    ap.add_argument("--lam", type=float, default=0.7,
                     help="WDL blend weight on the search-eval term, matching train.py's --lambda default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(BASE, "net_eval_diagnostic_results.csv"))
    args = ap.parse_args()

    print(f"Sampling up to {args.samples} positions from {PGN_GLOB} ...")
    samples = sample_positions(args.samples, seed=args.seed)
    print(f"  got {len(samples)} candidate positions")

    nnue = UciEngine(ENGINE, {"NNFile": NNUE_FILE, "Hash": "64", "Threads": "1"})
    sf   = UciEngine(SF,     {"Hash": "64", "Threads": "1"})

    rows = []  # (fen, net_cp, sf_cp, wdl_result_stm, bucket)
    t0 = time.time()
    for i, (fen, wdl_stm, bucket) in enumerate(samples):
        try:
            net_cp = nnue.static_eval_cp(fen)
            sf_cp  = sf.search_score_cp(fen, args.sf_movetime)
        except Exception as e:
            print(f"  [skip] {fen}: {e}")
            continue
        if sf_cp is None:
            continue
        rows.append((fen, net_cp, sf_cp, wdl_stm, bucket))
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(samples)}  ({elapsed:.0f}s, {elapsed/(i+1):.2f}s/pos)", flush=True)

    nnue.quit()
    sf.quit()

    with open(args.out, "w") as f:
        f.write("fen,net_cp,sf_cp,wdl_result_stm,bucket\n")
        for fen, net_cp, sf_cp, wdl_stm, bucket in rows:
            f.write(f'"{fen}",{net_cp},{sf_cp},{wdl_stm},{bucket}\n')

    n = len(rows)
    net_cps  = [r[1] for r in rows]
    sf_cps   = [r[2] for r in rows]
    net_wdl  = [cp_to_wdl(c) for c in net_cps]
    target   = [args.lam * cp_to_wdl(r[2]) + (1 - args.lam) * r[3] for r in rows]

    mse_blend = sum((a - b) ** 2 for a, b in zip(net_wdl, target)) / n
    mae_cp    = sum(abs(a - b) for a, b in zip(net_cps, sf_cps)) / n
    r_cp      = pearson(net_cps, sf_cps)
    slope, intercept = linreg(sf_cps, net_cps)

    print(f"\n{'='*60}")
    print(f"  N positions:                    {n}")
    print(f"  MAE net_cp vs sf_cp:             {mae_cp:.1f} cp")
    print(f"  Pearson r (cp space):            {r_cp:.4f}")
    print(f"  net_cp ~ sf_cp regression:       slope={slope:.3f}  intercept={intercept:+.1f}")
    print(f"    (slope far from 1.0 => systematic cp-scale mismatch between engines --")
    print(f"     rescale before trusting MAE/r above)")
    print(f"  MSE vs lam={args.lam} blended target:  {mse_blend:.6f}")
    print(f"    (train.py's reported 0.0073 is a TRAINING-set loss under the same blend;")
    print(f"     this is a same-formula but different-distribution number, not a strict")
    print(f"     apples-to-apples comparison -- see script docstring)")
    print(f"  RMSE (win-prob points):          {math.sqrt(mse_blend)*100:.2f} pp")

    print(f"\n  Per-bucket breakdown (bucket = non-king piece count, 0=endgame..7=full board):")
    print(f"  {'bucket':>6}  {'n':>5}  {'MAE cp':>8}  {'MSE(blend)':>11}")
    for b in range(NUM_BUCKETS):
        brows = [r for r in rows if r[4] == b]
        if not brows:
            print(f"  {b:>6}  {0:>5}      --           --")
            continue
        bn = len(brows)
        b_mae = sum(abs(r[1] - r[2]) for r in brows) / bn
        b_target = [args.lam * cp_to_wdl(r[2]) + (1 - args.lam) * r[3] for r in brows]
        b_net_wdl = [cp_to_wdl(r[1]) for r in brows]
        b_mse = sum((a - b_) ** 2 for a, b_ in zip(b_net_wdl, b_target)) / bn
        print(f"  {b:>6}  {bn:>5}  {b_mae:>8.1f}  {b_mse:>11.6f}")

    print(f"\n  Results CSV: {args.out}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
