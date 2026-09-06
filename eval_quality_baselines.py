import csv, sys
sys.path.insert(0,'.')
from eval_quality_test import auc, spearman
import chess

VAL={chess.PAWN:100,chess.KNIGHT:320,chess.BISHOP:330,chess.ROOK:500,chess.QUEEN:900,chess.KING:0}
# crude PSQT: centralisation bonus for knights/bishops, pawn advancement
def material(b):
    s=0
    for sq,pc in b.piece_map().items():
        v=VAL[pc.piece_type]
        s+= v if pc.color==chess.WHITE else -v
    return s
def mat_psqt(b):
    s=0
    for sq,pc in b.piece_map().items():
        v=VAL[pc.piece_type]
        f,r=chess.square_file(sq),chess.square_rank(sq)
        cent=(3.5-abs(f-3.5))+(3.5-abs(r-3.5))
        if pc.piece_type in (chess.KNIGHT,chess.BISHOP): v+=int(cent*4)
        if pc.piece_type==chess.PAWN:
            adv=r if pc.color==chess.WHITE else 7-r
            v+=adv*adv*2
        s+= v if pc.color==chess.WHITE else -v
    return s

rows=[]
with open('eval_quality_raw.csv') as fh:
    for r in csv.DictReader(fh):
        b=chess.Board(r['fen'])
        rows.append((float(r['white_score']),int(r['ours_cp']),int(r['sf_final_cp']),
                     material(b),mat_psqt(b), b))

dec=[r for r in rows if r[0]!=0.5]
lab=[1 if r[0]==1.0 else 0 for r in dec]
print(f"{'evaluator':<26}{'AUC':>8}{'Spearman':>11}")
print("-"*45)
for name,idx in [("material only",3),("material + crude PSQT",4),
                 ("ours (model.nnue)",1),("Stockfish 18 final",2)]:
    a=auc([r[idx] for r in dec],lab)
    s=spearman([r[idx] for r in rows],[r[0] for r in rows])
    print(f"{name:<26}{a:>8.4f}{s:>11.4f}")

# breakdown by piece count (proxy for game phase / output bucket)
print("\nAUC by piece count (game phase):")
print(f"{'pieces':<12}{'n':>6}{'ours':>9}{'SF':>9}{'material':>10}")
for lo,hi,label in [(3,10,'endgame'),(11,20,'middlegame'),(21,32,'opening')]:
    sub=[r for r in dec if lo<=len(r[5].piece_map())<=hi]
    if len(sub)<50: continue
    l=[1 if r[0]==1.0 else 0 for r in sub]
    print(f"{label:<12}{len(sub):>6}{auc([r[1] for r in sub],l):>9.4f}"
          f"{auc([r[2] for r in sub],l):>9.4f}{auc([r[3] for r in sub],l):>10.4f}")
