"""Control test: rank each static evaluator against a DEEP Stockfish verdict
instead of the game result.  This removes the position-selection bias --
'how good is this position, objectively' has no path dependence, unlike
'did the side to move eventually win this particular game'."""
import csv, re, subprocess, sys, statistics
sys.path.insert(0,'.')
from eval_quality_test import spearman
import chess

VAL={chess.PAWN:100,chess.KNIGHT:320,chess.BISHOP:330,chess.ROOK:500,chess.QUEEN:900,chess.KING:0}
def material(b):
    return sum((VAL[p.piece_type] if p.color==chess.WHITE else -VAL[p.piece_type])
               for p in b.piece_map().values())

SF='./stockfish/stockfish-macos-m1-apple-silicon'
p=subprocess.Popen([SF],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                   stderr=subprocess.DEVNULL,text=True,bufsize=1)
def send(s): p.stdin.write(s+"\n"); p.stdin.flush()
send("uci")
while not p.stdout.readline().startswith("uciok"): pass
send("setoption name Threads value 1"); send("setoption name Hash value 256")
send("isready")
while not p.stdout.readline().startswith("readyok"): pass

SCORE=re.compile(r"score (cp|mate) (-?\d+)")
def deep(fen, ms):
    send("position fen "+fen); send(f"go movetime {ms}")
    last=None
    while True:
        line=p.stdout.readline()
        if not line: raise RuntimeError("sf died")
        m=SCORE.search(line)
        if m: last=m
        if line.startswith("bestmove"): break
    if last is None: return None
    if last.group(1)=="mate":
        v=int(last.group(2)); return 20000 if v>0 else -20000
    return int(last.group(2))

rows=[]
with open('./eval_quality_raw.csv') as fh:
    rows=list(csv.DictReader(fh))
rows=rows[:900]
print(f"deep-searching {len(rows)} positions @200ms...",flush=True)

ours=[];sfstat=[];mat=[];tgt=[]
for i,r in enumerate(rows):
    fen=r['fen']; b=chess.Board(fen)
    d=deep(fen,200)
    if d is None: continue
    # deep score is side-to-move relative -> white relative
    if b.turn==chess.BLACK: d=-d
    tgt.append(d)
    ours.append(int(r['ours_cp']))
    sfstat.append(int(r['sf_final_cp']))
    mat.append(material(b))
    if (i+1)%200==0: print(f"  {i+1}/{len(rows)}",flush=True)

print(f"\nSpearman rho vs Stockfish deep search (200ms, n={len(tgt)})")
print(f"  {'material only':<26}{spearman(mat,tgt):.4f}")
print(f"  {'ours (model.nnue)':<26}{spearman(ours,tgt):.4f}")
print(f"  {'Stockfish static eval':<26}{spearman(sfstat,tgt):.4f}")

# also: mean abs rank-percentile error, and how often each mis-signs vs deep
def missign(v):
    n=sum(1 for a,t in zip(v,tgt) if abs(t)>100 and (a>0)!=(t>0))
    d=sum(1 for t in tgt if abs(t)>100)
    return n/d*100
print(f"\nWrong SIGN vs deep search (only where |deep|>100cp, n={sum(1 for t in tgt if abs(t)>100)})")
print(f"  {'material only':<26}{missign(mat):.1f}%")
print(f"  {'ours (model.nnue)':<26}{missign(ours):.1f}%")
print(f"  {'Stockfish static eval':<26}{missign(sfstat):.1f}%")
