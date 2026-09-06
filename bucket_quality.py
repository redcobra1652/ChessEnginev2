#!/usr/bin/env python3
"""
Per-bucket NNUE eval quality.

get_bucket() splits positions by non-king piece count into 8 output
buckets.  This measures each bucket's eval quality separately, against a
deep Stockfish verdict (objective target, no game-outcome path
dependence), with a naive material count as the floor.

A bucket whose Spearman is at or below the material floor is a bucket the
net has learned nothing useful in.
"""
import argparse, random, re, subprocess, sys, collections
sys.path.insert(0, '.')
import chess, chess.pgn
import data
from eval_quality_test import spearman, OurEngine

VAL={chess.PAWN:100,chess.KNIGHT:320,chess.BISHOP:330,chess.ROOK:500,chess.QUEEN:900,chess.KING:0}
def material(b):
    return sum((VAL[p.piece_type] if p.color==chess.WHITE else -VAL[p.piece_type])
               for p in b.piece_map().values())

SCORE=re.compile(r"score (cp|mate) (-?\d+)")

class SF:
    def __init__(self, path, hash_mb=256):
        self.p=subprocess.Popen([path],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL,text=True,bufsize=1)
        self._s("uci"); self._w("uciok")
        self._s("setoption name Threads value 1"); self._s(f"setoption name Hash value {hash_mb}")
        self._s("isready"); self._w("readyok")
    def _s(self,x): self.p.stdin.write(x+"\n"); self.p.stdin.flush()
    def _w(self,t):
        while True:
            l=self.p.stdout.readline()
            if not l: raise RuntimeError("sf died")
            if l.startswith(t): return l
    def deep(self,fen,ms):
        self._s("position fen "+fen); self._s(f"go movetime {ms}")
        last=None
        while True:
            l=self.p.stdout.readline()
            if not l: raise RuntimeError("sf died")
            m=SCORE.search(l)
            if m: last=m
            if l.startswith("bestmove"): break
        if last is None: return None
        if last.group(1)=="mate":
            v=int(last.group(2)); return 20000 if v>0 else -20000
        return int(last.group(2))

def collect(pgns, per_bucket, min_ply, seed):
    """Stratified sample: aim for `per_bucket` positions in each of the 8 buckets."""
    rng=random.Random(seed)
    pools=collections.defaultdict(list); seen=set()
    for path in pgns:
        try: fh=open(path)
        except OSError: continue
        with fh:
            while True:
                g=chess.pgn.read_game(fh)
                if g is None: break
                b=g.board()
                for ply,mv in enumerate(g.mainline_moves()):
                    b.push(mv)
                    if ply<min_ply or b.is_check() or b.is_game_over(): continue
                    f=b.fen()
                    if f in seen: continue
                    seen.add(f)
                    pools[data.get_bucket(b,num_buckets=8)].append(f)
    out=[]
    for bkt in range(8):
        p=pools.get(bkt,[])
        rng.shuffle(p)
        out += [(f,bkt) for f in p[:per_bucket]]
    return out, {k:len(v) for k,v in sorted(pools.items())}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pgn",nargs="+",default=["games/vs_sf2750.pgn","elo_ladder_2900.pgn",
        "sprt_rfp100.pgn","clean_depth_value.pgn","sprt_mt4_vs_1t.pgn","sanity_timefix2.pgn",
        "sanity_book_uci.pgn","sprt_rfp165.pgn","sanity_lmr1675.pgn"])
    ap.add_argument("--engine",default="./nnue_engine")
    ap.add_argument("--net",default="checkpoints/model.nnue")
    ap.add_argument("--sf",default="./stockfish/stockfish-macos-m1-apple-silicon")
    ap.add_argument("--per-bucket",type=int,default=200)
    ap.add_argument("--min-ply",type=int,default=8)
    ap.add_argument("--movetime",type=int,default=150)
    ap.add_argument("--seed",type=int,default=7)
    a=ap.parse_args()

    import os
    pos,avail=collect(a.pgn,a.per_bucket,a.min_ply,a.seed)
    print("available positions per bucket:",avail,file=sys.stderr)
    print(f"sampled {len(pos)}",file=sys.stderr)

    ours=OurEngine(a.engine,os.path.abspath(a.net))
    sf=SF(a.sf)

    rec=collections.defaultdict(list)
    for i,(fen,bkt) in enumerate(pos):
        b=chess.Board(fen)
        o=ours.eval_cp(fen)
        d=sf.deep(fen,a.movetime)
        if o is None or d is None: continue
        if b.turn==chess.BLACK: d=-d
        rec[bkt].append((o,d,material(b)))
        if (i+1)%300==0: print(f"  {i+1}/{len(pos)}",file=sys.stderr)

    print(f"\n{'bucket':<8}{'pieces':<10}{'n':>6}{'ours rho':>11}{'material rho':>14}{'ours sign-err':>15}{'mat sign-err':>14}")
    print("-"*80)
    tot=[]
    for bkt in range(8):
        r=rec.get(bkt,[])
        if len(r)<40: 
            print(f"{bkt:<8}{'':<10}{len(r):>6}   (too few positions to measure)")
            continue
        o=[x[0] for x in r]; d=[x[1] for x in r]; m=[x[2] for x in r]
        dec=[k for k in range(len(r)) if abs(d[k])>100]
        se_o=sum(1 for k in dec if (o[k]>0)!=(d[k]>0))/len(dec)*100 if dec else float('nan')
        se_m=sum(1 for k in dec if (m[k]>0)!=(d[k]>0))/len(dec)*100 if dec else float('nan')
        lo=bkt*32//8; hi=((bkt+1)*32//8)-1
        print(f"{bkt:<8}{f'{lo}-{hi}':<10}{len(r):>6}{spearman(o,d):>11.4f}{spearman(m,d):>14.4f}"
              f"{se_o:>14.1f}%{se_m:>13.1f}%")
        tot.append((bkt,spearman(o,d),spearman(m,d)))
    print("\n(pieces = non-king piece count range that maps to this bucket)")
    beat=[b for b,so,sm in tot if so>sm]
    lose=[b for b,so,sm in tot if so<=sm]
    print(f"\nbuckets where the net beats naive material : {beat}")
    print(f"buckets where it does NOT                  : {lose}")

if __name__=="__main__":
    main()
