#!/usr/bin/env python3
"""
At-scale, depth-by-depth node count comparison: nnue_engine vs Stockfish, all
pruning/search machinery combined (not one mechanism in isolation) -- answers
"how many nodes, at what depth, and how does that compare to Stockfish."

For each sampled real-game FEN, runs `go depth D` on both engines and records
every intermediate `info depth N ... nodes X` line (the node count of the
completed iterative-deepening iteration at each depth, not a delta). Then, per
depth d, aggregates across positions: sum of nodes (both engines), and the
ratio of those sums -- this is exactly the "where do our nodes live, and how
much more than Stockfish, at each depth" table.

Usage:
  python3 depth_nodecount_compare.py --samples 100 --maxdepth 14
"""
import argparse
import os
import statistics
import subprocess
import sys

sys.path.insert(0, ".")
from nodecount_sweep import sample_fens

STOCKFISH = "./stockfish/stockfish-macos-m1-apple-silicon"
OURS = "./nnue_engine"
NNFILE = os.path.abspath("checkpoints/model.nnue")


def run_depth_series(cmd, fen, maxdepth, is_ours):
    p = subprocess.Popen([cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def send(l):
        p.stdin.write(l + "\n")
        p.stdin.flush()

    def wait_for(tok):
        while True:
            line = p.stdout.readline()
            if not line:
                raise RuntimeError("engine exited")
            if tok in line:
                return line

    send("uci")
    wait_for("uciok")
    if is_ours:
        send(f"setoption name NNFile value {NNFILE}")
    send("setoption name Hash value 64")
    send("setoption name Threads value 1")
    send("isready")
    wait_for("readyok")
    send(f"position fen {fen}")
    send(f"go depth {maxdepth}")
    by_depth = {}
    while True:
        line = p.stdout.readline()
        if not line:
            break
        line = line.strip()
        if line.startswith("info") and " depth " in line and " nodes " in line:
            toks = line.split()
            try:
                d = int(toks[toks.index("depth") + 1])
                nd = int(toks[toks.index("nodes") + 1])
                by_depth[d] = nd
            except (ValueError, IndexError):
                pass
        if line.startswith("bestmove"):
            break
    send("quit")
    try:
        p.wait(timeout=5)
    except Exception:
        p.kill()
    return by_depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", default="games/vs_sf2750.pgn")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--maxdepth", type=int, default=14)
    ap.add_argument("--ply-lo", type=int, default=10)
    ap.add_argument("--ply-hi", type=int, default=70)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    fens = sample_fens(args.pgn, args.samples, args.ply_lo, args.ply_hi, args.seed)
    print(f"sampled {len(fens)} positions", file=sys.stderr)

    # Cumulative nodes-to-reach-depth-d (what iterative deepening reports),
    # and the incremental cost of iteration d alone (nodes[d] - nodes[d-1]) --
    # the latter is the real "how much work lives at this depth" metric,
    # comparable in shape between engines regardless of overall tree size.
    ours_cum = {d: 0 for d in range(1, args.maxdepth + 1)}
    sf_cum = {d: 0 for d in range(1, args.maxdepth + 1)}
    ours_inc = {d: 0 for d in range(1, args.maxdepth + 1)}
    sf_inc = {d: 0 for d in range(1, args.maxdepth + 1)}
    per_depth_cum_ratios = {d: [] for d in range(1, args.maxdepth + 1)}
    per_depth_inc_ratios = {d: [] for d in range(1, args.maxdepth + 1)}

    for i, fen in enumerate(fens):
        ours = run_depth_series(OURS, fen, args.maxdepth, True)
        sf = run_depth_series(STOCKFISH, fen, args.maxdepth, False)
        for d in range(1, args.maxdepth + 1):
            if d in ours:
                ours_cum[d] += ours[d]
                oinc = ours[d] - ours.get(d - 1, 0)
                ours_inc[d] += oinc
            if d in sf:
                sf_cum[d] += sf[d]
                sinc = sf[d] - sf.get(d - 1, 0)
                sf_inc[d] += sinc
            if d in ours and d in sf and sf[d] > 0:
                per_depth_cum_ratios[d].append(ours[d] / sf[d])
            if d in ours and d in sf and d - 1 in ours and d - 1 in sf:
                sinc = sf[d] - sf[d - 1]
                oinc = ours[d] - ours[d - 1]
                if sinc > 0:
                    per_depth_inc_ratios[d].append(oinc / sinc)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(fens)} done", file=sys.stderr)

    print()
    print("=== Cumulative nodes to reach depth d (iterative-deepening total) ===")
    print(f"{'depth':>5} {'ours_sum':>14} {'sf_sum':>14} {'sum_ratio':>10} "
          f"{'median_ratio':>13} {'n_pos':>6}")
    for d in range(1, args.maxdepth + 1):
        if sf_cum[d] == 0:
            continue
        sum_ratio = ours_cum[d] / sf_cum[d]
        med_ratio = statistics.median(per_depth_cum_ratios[d]) if per_depth_cum_ratios[d] else float("nan")
        print(f"{d:>5} {ours_cum[d]:>14} {sf_cum[d]:>14} {sum_ratio:>10.3f} "
              f"{med_ratio:>13.3f} {len(per_depth_cum_ratios[d]):>6}")

    print()
    print("=== Incremental nodes spent AT iteration d only (nodes[d]-nodes[d-1]) ===")
    print("This is where the search work actually lives, at each nominal depth.")
    print(f"{'depth':>5} {'ours_inc':>14} {'sf_inc':>14} {'inc_ratio':>10} "
          f"{'median_ratio':>13} {'n_pos':>6}")
    grand_total_inc = sum(ours_inc.values())
    running = 0
    for d in range(1, args.maxdepth + 1):
        if sf_inc[d] <= 0:
            continue
        inc_ratio = ours_inc[d] / sf_inc[d] if sf_inc[d] else float("nan")
        med_ratio = statistics.median(per_depth_inc_ratios[d]) if per_depth_inc_ratios[d] else float("nan")
        running += ours_inc[d]
        print(f"{d:>5} {ours_inc[d]:>14} {sf_inc[d]:>14} {inc_ratio:>10.3f} "
              f"{med_ratio:>13.3f} {len(per_depth_inc_ratios[d]):>6}"
              f"   (ours: {100*ours_inc[d]/grand_total_inc:4.1f}% of our total work, "
              f"cum {100*running/grand_total_inc:4.1f}%)")


if __name__ == "__main__":
    main()
