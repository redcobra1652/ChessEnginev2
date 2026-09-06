import csv, sys
sys.path.insert(0,'.')
import chess, torch
import data
from model import NNUE
from eval_quality_test import auc, spearman

net = NNUE()
ck = torch.load('./checkpoints/nnue_best.pt',
                map_location='cpu', weights_only=False)
net.load_state_dict(ck['state_dict']); net.eval()

rows=list(csv.DictReader(open('./eval_quality_raw.csv')))
print(f"scoring {len(rows)} positions with the float model...", flush=True)
fl=[]
with torch.no_grad():
    for i,r in enumerate(rows):
        b=chess.Board(r['fen'])
        wi,bi = data.board_to_halfkp(b)
        w_t=torch.from_numpy(wi).long().unsqueeze(0)
        b_t=torch.from_numpy(bi).long().unsqueeze(0)
        out=net(w_t,b_t,bucket=data.get_bucket(b,num_buckets=8),
                stm_white=(b.turn==chess.WHITE)).item()
        fl.append(out if b.turn==chess.WHITE else -out)
        if (i+1)%400==0: print(f"  {i+1}",flush=True)

ws=[float(r['white_score']) for r in rows]
ours=[int(r['ours_cp']) for r in rows]
dec=[i for i,w in enumerate(ws) if w!=0.5]
lab=[1 if ws[i]==1.0 else 0 for i in dec]
print(f"\nAUC vs game result (n={len(dec)} decisive)")
print(f"  quantized .nnue (engine)  : {auc([ours[i] for i in dec],lab):.4f}")
print(f"  float .pt checkpoint      : {auc([fl[i]   for i in dec],lab):.4f}")
print(f"\nSpearman(quantized, float) = {spearman(ours,fl):.4f}   (1.0 = quantization lossless in rank terms)")
d=sorted(abs(a-b) for a,b in zip(ours,fl))
print(f"|quant - float| cp:  median={d[len(d)//2]:.1f}  p90={d[int(len(d)*.9)]:.1f}  max={d[-1]:.1f}")
