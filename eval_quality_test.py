#!/usr/bin/env python3
"""
Measure NNUE eval QUALITY, independent of search and independent of
centipawn-scale conventions.

Core idea: an evaluation function's job is to predict who wins.  Sample
positions from real played games, ask each engine for its *static* eval
(no search), and score those evals against the actual game result.

Headline metric is AUC (rank-based): the probability that a randomly
chosen white-win position is scored higher than a randomly chosen
white-loss position.  AUC is invariant to any monotone rescaling, so it
sidesteps the cp-convention gap between this net's Lichess-cloud-eval
training labels and modern Stockfish's normalization -- the confound that
made every previous eval-scale comparison in this project unreliable.
"""
import argparse, random, re, subprocess, sys
import chess, chess.pgn

OURS_RE = re.compile(r"^eval\s+(-?\d+)\s+bucket\s+(\d+)")
SF_FINAL_RE = re.compile(r"^Final evaluation\s+([+-][\d.]+)")
SF_NNUE_RE  = re.compile(r"^NNUE evaluation\s+([+-][\d.]+)")


class OurEngine:
    def __init__(self, path, net):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._send("uci")
        self._wait("uciok")
        self._send(f"setoption name NNFile value {net}")
        self._send("setoption name Threads value 1")
        self._send("setoption name OwnBook value false")
        self._send("isready")
        self._wait("readyok")

    def _send(self, s):
        self.p.stdin.write(s + "\n"); self.p.stdin.flush()

    def _wait(self, tok):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            if line.startswith(tok):
                return line

    def eval_cp(self, fen):
        """Returns white-relative centipawns."""
        self._send(f"position fen {fen}")
        self._send("eval")
        line = self._wait("eval ")
        m = OURS_RE.match(line)
        if not m:
            return None
        cp = int(m.group(1))                       # side-to-move relative
        stm_white = fen.split()[1] == "w"
        return cp if stm_white else -cp            # -> white-relative


class StockfishEngine:
    def __init__(self, path):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._send("uci"); self._wait("uciok")
        self._send("setoption name Threads value 1")
        self._send("isready"); self._wait("readyok")

    def _send(self, s):
        self.p.stdin.write(s + "\n"); self.p.stdin.flush()

    def _wait(self, tok):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("stockfish died")
            if line.startswith(tok):
                return line

    def eval_cp(self, fen):
        """Returns (final_cp, nnue_cp), both white-relative."""
        self._send(f"position fen {fen}")
        self._send("eval")
        final = nnue = None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("stockfish died")
            m = SF_NNUE_RE.match(line)
            if m:
                nnue = int(round(float(m.group(1)) * 100))
            m = SF_FINAL_RE.match(line)
            if m:
                final = int(round(float(m.group(1)) * 100))
                return final, nnue


def sample_positions(pgn_paths, per_game, min_ply, max_ply, seed):
    rng = random.Random(seed)
    out = []
    for path in pgn_paths:
        with open(path) as fh:
            while True:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                res = game.headers.get("Result", "*")
                if res == "1-0":   white_score = 1.0
                elif res == "0-1": white_score = 0.0
                elif res == "1/2-1/2": white_score = 0.5
                else: continue
                board = game.board()
                cands = []
                for ply, mv in enumerate(game.mainline_moves()):
                    board.push(mv)
                    if min_ply <= ply <= max_ply and not board.is_check():
                        cands.append(board.fen())
                if not cands:
                    continue
                for fen in rng.sample(cands, min(per_game, len(cands))):
                    out.append((fen, white_score))
    return out


def auc(scores, labels):
    """Rank-based AUC. labels in {0,1}. Handles ties correctly."""
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos = sum(1 for _, l in pairs if l == 1)
    neg = n - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", nargs="+", default=["games/vs_sf2750.pgn", "elo_ladder_2900.pgn"])
    ap.add_argument("--engine", default="./nnue_engine")
    ap.add_argument("--net", default="checkpoints/model.nnue")
    ap.add_argument("--sf", default="./stockfish/stockfish-macos-m1-apple-silicon")
    ap.add_argument("--per-game", type=int, default=10)
    ap.add_argument("--min-ply", type=int, default=10)
    ap.add_argument("--max-ply", type=int, default=90)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import os
    net_abs = os.path.abspath(args.net)

    pos = sample_positions(args.pgn, args.per_game, args.min_ply, args.max_ply, args.seed)
    if args.limit:
        pos = pos[:args.limit]
    print(f"sampled {len(pos)} positions from {len(args.pgn)} pgn file(s)", file=sys.stderr)

    ours = OurEngine(args.engine, net_abs)
    sf = StockfishEngine(args.sf)

    rows = []
    for i, (fen, ws) in enumerate(pos):
        o = ours.eval_cp(fen)
        f, n = sf.eval_cp(fen)
        if o is None or f is None:
            continue
        rows.append((fen, ws, o, f, n))
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(pos)}", file=sys.stderr)

    print(f"evaluated {len(rows)} positions\n", file=sys.stderr)

    with open("eval_quality_raw.csv", "w") as fh:
        fh.write("fen,white_score,ours_cp,sf_final_cp,sf_nnue_cp\n")
        for fen, ws, o, f, n in rows:
            fh.write(f'"{fen}",{ws},{o},{f},{n}\n')

    dec = [(ws, o, f, n) for _, ws, o, f, n in rows if ws != 0.5]
    labels = [1 if ws == 1.0 else 0 for ws, _, _, _ in dec]
    print("=" * 66)
    print(f"AUC  (predicting eventual game result; decisive games only, n={len(dec)})")
    print("=" * 66)
    print(f"  ours (model.nnue) : {auc([o for _, o, _, _ in dec], labels):.4f}")
    print(f"  SF final eval     : {auc([f for _, _, f, _ in dec], labels):.4f}")
    print(f"  SF raw NNUE eval  : {auc([n for _, _, _, n in dec], labels):.4f}")

    allws = [ws for _, ws, _, _, _ in rows]
    print(f"\nSpearman rho vs game result (all {len(rows)} positions, draws=0.5)")
    print(f"  ours              : {spearman([r[2] for r in rows], allws):.4f}")
    print(f"  SF final eval     : {spearman([r[3] for r in rows], allws):.4f}")
    print(f"  SF raw NNUE eval  : {spearman([r[4] for r in rows], allws):.4f}")

    print(f"\nSpearman rho, ours vs SF final (agreement between the two evals)")
    print(f"  = {spearman([r[2] for r in rows], [r[3] for r in rows]):.4f}")

    import statistics
    print(f"\nMagnitude (median |cp|)  -- scale differs by convention, not a quality metric")
    print(f"  ours          : {statistics.median(abs(r[2]) for r in rows):.0f}")
    print(f"  SF final      : {statistics.median(abs(r[3]) for r in rows):.0f}")
    print("\nwrote eval_quality_raw.csv")


if __name__ == "__main__":
    main()
