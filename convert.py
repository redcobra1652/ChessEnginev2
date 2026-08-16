"""
convert.py -- Convert PGN files to a fast-loading binary format (.bin).

Binary format per position (fixed 125 bytes, little-endian):
  uint16[30] white_indices   ( 60 bytes, 65535 = padding sentinel)
  uint16[30] black_indices   ( 60 bytes, 65535 = padding sentinel)
  int16      score_cp        (  2 bytes, centipawns STM-relative, clamped ±32767)
  uint8      wdl             (  1 byte,  wdl * 255 rounded, decode: / 255.0)
  uint8      bucket_idx      (  1 byte)
  uint8      stm_white       (  1 byte,  1=White to move, 0=Black)
  ─────────────────────────────────────
  Total: 30*2 + 30*2 + 2 + 1 + 1 + 1 = 125 bytes per position

Padding sentinel: uint16 value 65535 (0xFFFF) marks unused index slots.
HalfKP indices are 0..40959, so 65535 is safely out of range.

File layout:
  [header]
    uint32  magic      = 0x4E4E5545  ("NNUE")
    uint32  version    = 0x00000004  (v4: uint16 indices, compact eval fields)
    uint32  n_pos      = number of positions
    uint32  max_active = MAX_ACTIVE (sanity check on load)
  [positions]
    position[0]
    position[1]
    ...

Note: v3 files (253 bytes/pos, int32 indices) are not compatible.
Delete old .bin files and reconvert with this version.

Usage:
  python3 convert.py --pgn warm_train/ --out positions.bin
  python3 convert.py --pgn new_games/ --out positions.bin --append
  python3 convert.py --pgn warm_train/ --out positions.bin --workers 4
  python3 train.py --bin positions.bin --out checkpoints/nnue.pt ...
"""

import argparse
import multiprocessing as mp
import os
import struct
import sys
import time

import data  # for iter_positions, resolve_pgn_paths, MAX_ACTIVE

# ── Binary format constants ───────────────────────────────────────────────────

MAGIC        = 0x4E4E5545   # "NNUE" in ASCII
VERSION      = 0x00000004   # v4: uint16 indices, compact eval fields
HEADER_SIZE  = 16           # 4 × uint32
PADDING_IDX  = 0xFFFF       # sentinel for unused uint16 index slots (> max HalfKP index 40959)
POSITION_BYTES = (
    data.MAX_ACTIVE * 2     # white_indices  uint16[30]
  + data.MAX_ACTIVE * 2     # black_indices  uint16[30]
  + 2                       # score_cp       int16  (centipawns, clamped ±32767)
  + 1                       # wdl            uint8  (wdl * 255)
  + 1                       # bucket_idx     uint8
  + 1                       # stm_white      uint8
)   # = 125 bytes


# ── Per-position pack/unpack ──────────────────────────────────────────────────

# Struct format: 30H 30H h B B B  (all little-endian)
# H = uint16, h = int16, B = uint8
_PACK_FMT = f"<{data.MAX_ACTIVE}H{data.MAX_ACTIVE}HhBBB"
_STRUCT   = struct.Struct(_PACK_FMT)

assert _STRUCT.size == POSITION_BYTES, \
    f"struct size mismatch: {_STRUCT.size} != {POSITION_BYTES}"

_CP_MAX = 32767   # int16 ceiling


def pack_position(white_idx, black_idx, score_cp, wdl, bucket, stm_white):
    """Pack one position tuple into 125 bytes.

    Indices: -1 padding slots are mapped to PADDING_IDX (65535).
    score_cp: clamped to ±32767 and stored as int16.
    wdl: multiplied by 255 and rounded to uint8.
    """
    # Map -1 padding → 65535; valid indices 0..40959 are unchanged
    wi = white_idx.tolist()
    bi = black_idx.tolist()
    wi = [PADDING_IDX if x < 0 else x for x in wi]
    bi = [PADDING_IDX if x < 0 else x for x in bi]

    cp_i16  = int(max(-_CP_MAX, min(_CP_MAX, int(score_cp))))
    wdl_u8  = int(round(float(wdl) * 255.0))
    wdl_u8  = max(0, min(255, wdl_u8))

    return _STRUCT.pack(
        *wi,
        *bi,
        cp_i16,
        wdl_u8,
        int(bucket),
        int(bool(stm_white)),
    )


def unpack_position(buf: bytes, offset: int):
    """Unpack one position from buf at offset.

    Returns (wi, bi, score_cp, wdl, bucket, stm_white) —
    same 6-tuple as data.iter_positions, compatible with train.py.

    Indices: 65535 sentinel slots are mapped back to -1 (padding).
    score_cp: returned as float32.
    wdl: decoded from uint8 as float32 (/ 255.0).
    """
    import numpy as np
    vals = _STRUCT.unpack_from(buf, offset)
    M = data.MAX_ACTIVE

    wi_raw = np.array(vals[:M],    dtype=np.int32)
    bi_raw = np.array(vals[M:2*M], dtype=np.int32)
    # Map sentinel 65535 back to -1 so model._accumulate() padding mask works
    wi_raw[wi_raw == PADDING_IDX] = -1
    bi_raw[bi_raw == PADDING_IDX] = -1

    cp       = np.float32(vals[2*M])          # int16 → float32
    wdl      = np.float32(vals[2*M+1]) / 255.0
    bucket   = np.int64(vals[2*M+2])
    stm_white = bool(vals[2*M+3])
    return wi_raw, bi_raw, cp, wdl, bucket, stm_white


# ── Worker process ────────────────────────────────────────────────────────────

def _worker(args_tuple):
    """
    Convert a single PGN file to a bytes blob.
    Runs in a worker process — no shared state.
    Returns (pgn_path, n_positions, bytes_blob).
    """
    pgn_path, pawn_units, num_buckets = args_tuple
    chunks = []
    n = 0
    try:
        for wi, bi, cp, wdl, bucket, stm_white in data.iter_positions(
            [pgn_path],
            pawn_units=pawn_units,
            num_buckets=num_buckets,
        ):
            chunks.append(pack_position(wi, bi, cp, wdl, bucket, stm_white))
            n += 1
    except Exception as e:
        # Don't crash the whole run on one bad file
        print(f"\n  [warn] {pgn_path}: {e}", file=sys.stderr, flush=True)
    return pgn_path, n, b"".join(chunks)


# ── Header helpers ────────────────────────────────────────────────────────────

def _write_header(f, n_positions: int):
    f.seek(0)
    f.write(struct.pack("<IIII", MAGIC, VERSION, n_positions, data.MAX_ACTIVE))


def _read_header(path: str):
    with open(path, "rb") as f:
        raw = f.read(HEADER_SIZE)
    magic, version, n_pos, max_active = struct.unpack("<IIII", raw)
    if magic != MAGIC:
        raise ValueError(f"{path}: not a valid .bin file (bad magic 0x{magic:08X})")
    if version != VERSION:
        raise ValueError(
            f"{path}: version mismatch (file=0x{version:08X}, code=0x{VERSION:08X}). "
            f"v3 files (253 bytes/pos) are not compatible — delete and reconvert."
        )
    if max_active != data.MAX_ACTIVE:
        raise ValueError(
            f"{path}: MAX_ACTIVE mismatch (file={max_active}, code={data.MAX_ACTIVE})"
        )
    return n_pos


# ── Converter ─────────────────────────────────────────────────────────────────

def convert(pgn_paths: list[str],
            out_path: str,
            append: bool = False,
            workers: int = 0,
            pawn_units: bool = True,
            num_buckets: int = 8) -> int:
    """
    Convert pgn_paths → out_path binary file.

    append=True  adds positions to an existing file.
    workers=0    uses all available CPU cores.
    Returns total positions written.
    """
    n_workers = workers or mp.cpu_count()
    print(f"Converting {len(pgn_paths)} PGN file(s) "
          f"using {n_workers} worker(s) → {out_path}", flush=True)

    # ── Open output file ──────────────────────────────────────────────────────
    existing_n = 0
    if append and os.path.exists(out_path):
        existing_n = _read_header(out_path)
        print(f"  Appending to existing file ({existing_n:,} positions)", flush=True)
        mode = "r+b"
    else:
        mode = "wb"

    total_written = existing_n
    t0 = time.time()
    done = 0

    with open(out_path, mode) as f:
        if not append or not os.path.exists(out_path):
            # Write placeholder header; will be patched at the end
            _write_header(f, 0)
        else:
            # Seek to end to append
            f.seek(0, 2)

        # ── Process files in parallel ─────────────────────────────────────────
        worker_args = [(p, pawn_units, num_buckets) for p in pgn_paths]

        with mp.Pool(processes=n_workers) as pool:
            for pgn_path, n_pos, blob in pool.imap_unordered(_worker, worker_args):
                f.write(blob)
                total_written += n_pos
                done += 1
                elapsed = time.time() - t0
                rate = (total_written - existing_n) / max(elapsed, 1)
                print(
                    f"\r  [{done}/{len(pgn_paths)}]  "
                    f"{total_written:,} positions  |  {rate:,.0f} pos/s  |  "
                    f"{elapsed:.0f}s",
                    end="", flush=True,
                )

        print(flush=True)

        # ── Patch header with final count ─────────────────────────────────────
        _write_header(f, total_written)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    elapsed = time.time() - t0
    new_positions = total_written - existing_n
    print(
        f"Done — {new_positions:,} new positions written  "
        f"({total_written:,} total)  |  {size_mb:.0f} MB  |  {elapsed:.0f}s",
        flush=True,
    )
    return total_written


# ── Fast binary streaming (used by data.py / train.py) ───────────────────────

def iter_positions_bin(bin_path: str, shuffle: bool = True, block: int = 4096):
    """
    Stream positions from a v4 .bin file (125 bytes/position).

    Uses block reads (default 4096 positions at a time) to avoid the overhead
    of seek-per-position on large files, then shuffles within each block.
    Gives good randomness with ~100x fewer syscalls than per-position seeking.

    Yields the same 6-tuple as data.iter_positions:
        (white_indices[int32], black_indices[int32], score_cp[float32],
         wdl[float32], bucket_idx[int64], stm_white[bool])

    Indices are returned as int32 with -1 for padding slots, matching the
    format expected by model._accumulate() and Reservoir.sample_batch().

    Requires version 0x00000004 files. Delete v3 files and reconvert.
    """
    import numpy as np
    import random as _random

    n_pos = _read_header(bin_path)
    block_indices = list(range(0, n_pos, block))
    if shuffle:
        _random.shuffle(block_indices)

    with open(bin_path, "rb") as f:
        for block_start in block_indices:
            count = min(block, n_pos - block_start)
            f.seek(HEADER_SIZE + block_start * POSITION_BYTES)
            raw = f.read(count * POSITION_BYTES)
            offsets = list(range(count))
            if shuffle:
                _random.shuffle(offsets)
            for i in offsets:
                yield unpack_position(raw, i * POSITION_BYTES)


# ── Info / verify ─────────────────────────────────────────────────────────────

def info(bin_path: str):
    """Print summary stats about a .bin file."""
    import numpy as np

    n_pos = _read_header(bin_path)
    size_mb = os.path.getsize(bin_path) / 1024 / 1024
    print(f"File:       {bin_path}")
    print(f"Positions:  {n_pos:,}")
    print(f"Size:       {size_mb:.1f} MB  ({size_mb/max(n_pos,1)*1e6:.0f} bytes/pos)")

    # Sample 10k positions to show cp/wdl distribution
    print("Sampling 10,000 positions for stats …", flush=True)
    import random as _random
    sample_size = min(10_000, n_pos)
    offsets = sorted(_random.sample(range(n_pos), sample_size))
    cps, wdls, buckets = [], [], []
    with open(bin_path, "rb") as f:
        for idx in offsets:
            f.seek(HEADER_SIZE + idx * POSITION_BYTES)
            buf = f.read(POSITION_BYTES)
            _, _, cp, wdl, bucket, _ = unpack_position(buf, 0)
            cps.append(float(cp))
            wdls.append(float(wdl))
            buckets.append(int(bucket))
    cps     = np.array(cps)
    wdls    = np.array(wdls)
    buckets = np.array(buckets)
    print(f"score_cp:   mean={cps.mean():.1f}  std={cps.std():.1f}  "
          f"min={cps.min():.0f}  max={cps.max():.0f}")
    print(f"wdl:        mean={wdls.mean():.3f}  std={wdls.std():.3f}")
    print(f"buckets:    ", end="")
    for b in range(8):
        pct = (buckets == b).mean() * 100
        print(f"{b}:{pct:.1f}%%  ", end="")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert PGN files to fast-loading binary format for NNUE training"
    )
    sub = p.add_subparsers(dest="cmd")

    # ── convert ──────────────────────────────────────────────────────────────
    c = sub.add_parser("convert", help="Convert PGNs to .bin (default command)")
    c.add_argument("--pgn",       nargs="+", required=True,
                   help="PGN file(s), directory, or glob")
    c.add_argument("--out",       required=True,
                   help="Output .bin file")
    c.add_argument("--append",    action="store_true",
                   help="Append to existing .bin file instead of overwriting")
    c.add_argument("--workers",   type=int, default=0,
                   help="Worker processes (default: all CPU cores)")
    c.add_argument("--no-pawn-units", dest="pawn_units", action="store_false",
                   default=True,
                   help="%%eval is in raw centipawns (default: pawn units)")
    c.add_argument("--buckets",   type=int, default=8)

    # ── info ─────────────────────────────────────────────────────────────────
    i = sub.add_parser("info", help="Print stats about a .bin file")
    i.add_argument("bin", help=".bin file to inspect")

    return p, p.parse_args()


def main():
    p, args = parse_args()

    # Default to 'convert' if no subcommand given but --pgn is present
    # (allows: python3 convert.py --pgn ... --out ...)
    if args.cmd is None:
        # Re-parse without subcommand for convenience
        cp = argparse.ArgumentParser()
        cp.add_argument("--pgn",    nargs="+", required=True)
        cp.add_argument("--out",    required=True)
        cp.add_argument("--append", action="store_true")
        cp.add_argument("--workers", type=int, default=0)
        cp.add_argument("--no-pawn-units", dest="pawn_units",
                        action="store_false", default=True)
        cp.add_argument("--buckets", type=int, default=8)
        args = cp.parse_args()

    if hasattr(args, "bin"):
        info(args.bin)
        return

    pgn_paths = data.resolve_pgn_paths(args.pgn)
    print(f"Found {len(pgn_paths)} PGN file(s)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    convert(
        pgn_paths,
        args.out,
        append=args.append,
        workers=args.workers,
        pawn_units=args.pawn_units,
        num_buckets=args.buckets,
    )


if __name__ == "__main__":
    main()
