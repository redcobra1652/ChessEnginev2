#!/usr/bin/env python3
"""
Nodes-to-fixed-depth comparison at scale, replacing the earlier 1-2-position
spot checks (documented in CLAUDE.md as chaotically noisy near a reasonable
parameter value: the non-monotonic startpos LMR row, the seelmr sign flip).

Samples deduped FENs from a PGN (ply 10-70, not in check, default
games/vs_sf2750.pgn -- same source/window as the prunestats node-mass
measurement), then for each position runs `go depth N` under two engine
binaries via raw UCI stdin/stdout and records the `nodes` field of the last
`info depth N ...` line. Reports median/IQR of the nodes ratio
(engine_a / engine_b), overall and this can be extended to bucket by
whatever dimension matters.

Usage:
  python3 nodecount_sweep.py --engine-a ./nnue_engine_candidate \
      --engine-b ./nnue_engine_baseline --depth 12 --samples 300 \
      --nnfile checkpoints/model.nnue
"""
import argparse
import random
import statistics
import subprocess
import sys

import chess
import chess.pgn


def sample_fens(pgn_path, n_samples, ply_lo, ply_hi, seed):
    positions = []
    seen = set()
    with open(pgn_path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            ply = 0
            for move in game.mainline_moves():
                board.push(move)
                ply += 1
                if ply_lo <= ply <= ply_hi and not board.is_check():
                    fen = board.fen()
                    if fen not in seen:
                        seen.add(fen)
                        positions.append(fen)
    rng = random.Random(seed)
    rng.shuffle(positions)
    return positions[:n_samples]


class Engine:
    def __init__(self, cmd, nnfile, hash_mb, threads):
        self.p = subprocess.Popen(
            [cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send(f"setoption name NNFile value {nnfile}")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send(f"setoption name Threads value {threads}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def _wait_for(self, token):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine exited unexpectedly")
            if token in line:
                return line

    def nodes_at_depth(self, fen, depth):
        self._send("ucinewgame")
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        last_nodes = None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine exited unexpectedly")
            line = line.strip()
            if line.startswith("info") and " nodes " in line:
                toks = line.split()
                try:
                    idx = toks.index("nodes")
                    last_nodes = int(toks[idx + 1])
                except (ValueError, IndexError):
                    pass
            if line.startswith("bestmove"):
                break
        return last_nodes

    def close(self):
        try:
            self._send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", default="games/vs_sf2750.pgn")
    ap.add_argument("--engine-a", required=True)
    ap.add_argument("--engine-b", required=True)
    ap.add_argument("--nnfile", default="checkpoints/model.nnue")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--ply-lo", type=int, default=10)
    ap.add_argument("--ply-hi", type=int, default=70)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None, help="optional CSV of per-position results")
    args = ap.parse_args()

    print(f"Sampling FENs from {args.pgn} (ply {args.ply_lo}-{args.ply_hi})...",
          file=sys.stderr)
    fens = sample_fens(args.pgn, args.samples, args.ply_lo, args.ply_hi, args.seed)
    print(f"Sampled {len(fens)} deduped positions.", file=sys.stderr)

    ea = Engine(args.engine_a, args.nnfile, args.hash, args.threads)
    eb = Engine(args.engine_b, args.nnfile, args.hash, args.threads)

    ratios = []
    rows = []
    for i, fen in enumerate(fens):
        na = ea.nodes_at_depth(fen, args.depth)
        nb = eb.nodes_at_depth(fen, args.depth)
        if na is None or nb is None or nb == 0:
            continue
        r = na / nb
        ratios.append(r)
        rows.append((fen, na, nb, r))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(fens)} done, running median ratio="
                  f"{statistics.median(ratios):.4f}", file=sys.stderr)

    ea.close()
    eb.close()

    if args.out:
        with open(args.out, "w") as f:
            f.write("fen,nodes_a,nodes_b,ratio\n")
            for fen, na, nb, r in rows:
                f.write(f'"{fen}",{na},{nb},{r:.6f}\n')

    ratios.sort()
    n = len(ratios)
    med = statistics.median(ratios)
    q1 = ratios[n // 4]
    q3 = ratios[(3 * n) // 4]
    mean = statistics.mean(ratios)
    wins = sum(1 for r in ratios if r < 0.999)
    losses = sum(1 for r in ratios if r > 1.001)
    ties = n - wins - losses

    print()
    print(f"depth={args.depth}  n={n}")
    print(f"engine-a: {args.engine_a}")
    print(f"engine-b: {args.engine_b}")
    print(f"nodes ratio (a/b): median={med:.4f}  mean={mean:.4f}  "
          f"IQR=[{q1:.4f}, {q3:.4f}]")
    print(f"a used fewer nodes: {wins}/{n} ({100*wins/n:.1f}%)   "
          f"a used more nodes: {losses}/{n} ({100*losses/n:.1f}%)   "
          f"tied: {ties}/{n}")


if __name__ == "__main__":
    main()
