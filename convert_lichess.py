"""
convert_lichess.py -- Convert Lichess eval JSON stream to v4 .bin format.

Reads the Lichess evaluation database (jsonl, one position per line) and
writes the same 125-byte v4 binary format used by convert.py / train.py.

Streaming: never loads more than one line into memory. Pipe directly from
zstd decompression:

  zstd -d --stdout lichess_db_eval.jsonl.zst | python3 convert_lichess.py --out positions.bin
  zstd -d --stdout lichess_db_eval.jsonl.zst | python3 convert_lichess.py --out positions.bin --limit 256000000

Chess960 filtering:
  Positions with Chess960 castling rights (A-H / a-h) are skipped.
  Standard castling rights use only K/Q/k/q; anything else is Chess960.

Eval selection:
  Each position has multiple evals at different depths/knodes. We pick
  the highest-depth eval that has exactly one PV (single best move eval,
  most reliable). If no single-PV eval exists, we take the first PV of
  the highest-depth multi-PV eval.

  Mate scores are skipped (no reliable centipawn value).
  Positions with |cp| > max_eval_cp (default 5000) are skipped.

WDL:
  The Lichess eval database does not include game results, so we derive
  WDL from the centipawn score using the same sigmoid as training:
    wdl = sigmoid(cp / WDL_SCALE)  where WDL_SCALE = 410 (matches train.py)
  This means lam=1.0 in train.py is equivalent to using engine score only,
  which is correct for this dataset (no game outcome available).

Output: same v4 .bin format as convert.py, readable by train.py --bin.
"""

import argparse
import json
import math
import os
import struct
import sys
import time

import chess
import numpy as np

import data    # board_to_halfkp, get_bucket, MAX_ACTIVE, HALFKP_SIZE
import convert # MAGIC, VERSION, HEADER_SIZE, POSITION_BYTES, pack_position,
               # _write_header

# ── Constants ─────────────────────────────────────────────────────────────────

WDL_SCALE   = 410.0   # must match train.py
MATE_CP     = 30000   # sentinel — positions with |cp| >= this are skipped
MAX_EVAL_CP = 5000    # filter extreme evals (same as data.iter_positions)

# Standard castling rights characters — anything else = Chess960
_STANDARD_CASTLE = set("KQkq-")


# ── Chess960 detection ────────────────────────────────────────────────────────

def is_chess960(fen: str) -> bool:
    """
    Return True if the FEN contains Chess960 castling rights.

    Standard FEN castling field uses only K, Q, k, q, or -.
    Chess960 FENs use file letters A-H / a-h to specify rook files.
    """
    parts = fen.split()
    if len(parts) < 3:
        return False
    castling = parts[2]
    return not all(c in _STANDARD_CASTLE for c in castling)


# ── WDL from centipawns ───────────────────────────────────────────────────────

def cp_to_wdl(cp: float) -> float:
    """Convert centipawn score to WDL via sigmoid (matches train.py cp_to_wdl)."""
    return 1.0 / (1.0 + math.exp(-cp / WDL_SCALE))


# ── Eval extraction ───────────────────────────────────────────────────────────

def best_cp(evals: list) -> float | None:
    """
    Extract the best centipawn score from a position's eval list.

    Strategy:
      1. Among all evals, prefer highest depth.
      2. Among equal-depth evals, prefer single-PV (most reliable).
      3. Take cp from the first PV of the chosen eval.
      4. Return None if the best eval is a mate score or out of range.
    """
    if not evals:
        return None

    # Sort by depth descending, then by number of PVs ascending (prefer 1-PV)
    candidates = sorted(evals,
                        key=lambda e: (-e.get("depth", 0), len(e.get("pvs", []))))

    for ev in candidates:
        pvs = ev.get("pvs", [])
        if not pvs:
            continue
        pv = pvs[0]
        if "mate" in pv:
            return None   # skip mate-in-N positions
        cp = pv.get("cp")
        if cp is None:
            continue
        cp = float(cp)
        if abs(cp) > MAX_EVAL_CP:
            return None
        return cp

    return None


# ── Header helpers (mirrors convert.py) ──────────────────────────────────────

def _write_header(f, n_positions: int):
    f.seek(0)
    f.write(struct.pack("<IIII",
                        convert.MAGIC,
                        convert.VERSION,
                        n_positions,
                        data.MAX_ACTIVE))


# ── Main converter ────────────────────────────────────────────────────────────

def convert_stream(in_stream,
                   out_path: str,
                   limit: int = 0,
                   num_buckets: int = 8,
                   min_depth: int = 20,
                   report_every: int = 500_000) -> int:
    """
    Read JSON lines from in_stream, write v4 .bin to out_path.

    limit=0 means no limit (write all qualifying positions).
    min_depth skips evals shallower than this (default 20 — filters low-quality evals).
    Returns total positions written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    t0 = time.time()
    n_written  = 0
    n_skipped  = 0
    n_read     = 0

    with open(out_path, "wb") as f:
        # Write placeholder header — patched with final count at end
        _write_header(f, 0)

        for raw_line in in_stream:
            n_read += 1

            # ── Parse JSON ────────────────────────────────────────────────────
            try:
                pos = json.loads(raw_line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue

            fen   = pos.get("fen", "")
            evals = pos.get("evals", [])

            # ── Chess960 filter ───────────────────────────────────────────────
            if is_chess960(fen):
                n_skipped += 1
                continue

            # ── Depth filter ──────────────────────────────────────────────────
            # Only keep positions where at least one eval reached min_depth
            max_depth = max((e.get("depth", 0) for e in evals), default=0)
            if max_depth < min_depth:
                n_skipped += 1
                continue

            # ── Eval extraction ───────────────────────────────────────────────
            cp = best_cp(evals)
            if cp is None:
                n_skipped += 1
                continue

            # ── Board setup ───────────────────────────────────────────────────
            try:
                # Lichess FENs omit move clocks; chess.Board handles this fine
                board = chess.Board(fen)
            except Exception:
                n_skipped += 1
                continue

            if not board.is_valid():
                n_skipped += 1
                continue

            # ── STM conversion ────────────────────────────────────────────────
            # Lichess cp is from White's perspective; negate for Black STM
            stm_white    = (board.turn == chess.WHITE)
            score_cp_stm = cp if stm_white else -cp

            # ── WDL from sigmoid ──────────────────────────────────────────────
            wdl_stm = cp_to_wdl(score_cp_stm)

            # ── HalfKP encoding ───────────────────────────────────────────────
            white_idx, black_idx = data.board_to_halfkp(board)
            bucket = data.get_bucket(board, num_buckets=num_buckets)

            # ── Pack and write ────────────────────────────────────────────────
            f.write(convert.pack_position(
                white_idx, black_idx,
                score_cp_stm, wdl_stm,
                bucket, stm_white,
            ))
            n_written += 1

            # ── Progress ──────────────────────────────────────────────────────
            if n_written % report_every == 0:
                elapsed = time.time() - t0
                rate    = n_written / max(elapsed, 1)
                size_gb = (convert.HEADER_SIZE + n_written * convert.POSITION_BYTES) / 1e9
                print(
                    f"\r  {n_written:,} written  |  {n_read:,} read  |  "
                    f"{rate:,.0f} pos/s  |  {size_gb:.2f} GB  |  {elapsed:.0f}s",
                    end="", flush=True,
                )

            if limit and n_written >= limit:
                break

        # ── Patch header ──────────────────────────────────────────────────────
        _write_header(f, n_written)

    elapsed  = time.time() - t0
    size_gb  = os.path.getsize(out_path) / 1e9
    skip_pct = 100 * n_skipped / max(n_read, 1)
    print(f"\n\nDone.")
    print(f"  Read:     {n_read:,} lines")
    print(f"  Written:  {n_written:,} positions  ({size_gb:.2f} GB)")
    print(f"  Skipped:  {n_skipped:,} ({skip_pct:.1f}%)  "
          f"[chess960 / mate / depth<{min_depth} / invalid]")
    print(f"  Time:     {elapsed:.0f}s  ({n_written/max(elapsed,1):,.0f} pos/s)")
    return n_written


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert Lichess eval JSONL stream to v4 .bin for NNUE training"
    )
    p.add_argument("--out",         required=True,
                   help="Output .bin file path")
    p.add_argument("--limit",       type=int, default=0,
                   help="Stop after this many positions (0 = all). "
                        "Use 256000000 for 256M positions (~32 GB).")
    p.add_argument("--min-depth",   type=int, default=20,
                   help="Skip positions where max eval depth < this (default: 20)")
    p.add_argument("--buckets",     type=int, default=8,
                   help="Number of material buckets (must match training, default: 8)")
    p.add_argument("--input",       default=None,
                   help="Input .jsonl file (default: read from stdin). "
                        "Supports plain .jsonl or piped zstd stream.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.input:
        print(f"Reading from: {args.input}", flush=True)
        in_stream = open(args.input, "r", encoding="utf-8", errors="replace")
    else:
        print("Reading from stdin (pipe zstd -d --stdout file.zst | python3 convert_lichess.py ...)",
              flush=True)
        in_stream = sys.stdin

    limit_str = f"{args.limit:,}" if args.limit else "unlimited"
    print(f"Output:       {args.out}", flush=True)
    print(f"Limit:        {limit_str} positions", flush=True)
    print(f"Min depth:    {args.min_depth}", flush=True)
    print(f"Buckets:      {args.buckets}", flush=True)
    print(flush=True)

    try:
        convert_stream(
            in_stream,
            args.out,
            limit=args.limit,
            num_buckets=args.buckets,
            min_depth=args.min_depth,
        )
    finally:
        if args.input:
            in_stream.close()


if __name__ == "__main__":
    main()
