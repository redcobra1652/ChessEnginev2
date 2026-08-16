"""
verify_quantization.py -- Verify the .nnue quantized eval matches the trained
float model within quantization noise.

Standalone diagnostic script (does not touch train.py or serialize.py logic,
only reads their output), per this project's precedent with
net_eval_diagnostic.py. Run this after any retrain or any change to
serialize.py's quantization scheme, BEFORE trusting the resulting
checkpoints/model.nnue in the engine.

What it checks:
  1. Byte-exact: does the .nnue payload match what serialize.py's
     quantise_* functions produce from the source .pt checkpoint? (Catches
     "wrong checkpoint was serialized" and serialize.py bugs where the
     written bytes don't match what was computed.)
  2. Weight-clipping rate per layer: what fraction of each layer's weights
     get clamped to the int8 boundary at its chosen scale constant. A high
     clipping rate (the output layer measured 96.875% at one point) means
     the scale constant is miscalibrated for the actual trained weight
     distribution and is destroying learned information.
  3. Manually replicates the exact integer forward pass nnue_engine.cpp
     runs (reading the .nnue file's raw bytes directly, independent of the
     C++ binary) and compares its cp output against the float model's
     output for a handful of diverse test positions. A match within a few
     cp (quantization noise) is healthy; tens-to-hundreds of cp divergence,
     or a sign flip, means a real quantization bug -- see CLAUDE.md's
     "NNUE output-layer quantization bug" section for the diagnosis that
     found exactly this via this method.

Usage:
    python3 verify_quantization.py [--pt checkpoints/nnue_best.pt] [--nnue checkpoints/model.nnue]
"""
import argparse
import struct
import sys

import chess
import numpy as np
import torch

import data
import serialize as S
from model import NNUE

FT_SCALE, L1_SCALE, L2_SCALE, OUT_SCALE = S.FT_SCALE, S.L1_SCALE, S.L2_SCALE, S.OUT_SCALE
H = 256
HALFKP_SIZE = data.HALFKP_SIZE
NUM_BUCKETS = 8

PIECE_TYPE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]

TEST_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqk2r/pp2bppp/2n2n2/2pp4/8/1P1P1NP1/PBPNPPBP/R2Q1RK1 w kq - 0 9",
    "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "2kr3r/ppp2ppp/2n1b3/2b1P3/3n4/2N2N2/PPP1BPPP/R1B1K2R w KQ - 0 12",
    "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
]


def check_byte_exact(pt_path, nnue_path):
    sd = torch.load(pt_path, map_location="cpu")["state_dict"]
    net = NNUE(hidden_size=256, num_buckets=NUM_BUCKETS)
    net.load_state_dict(sd)
    net.eval()
    sd2 = {k: v.detach().cpu().float() for k, v in net.state_dict().items()}

    expected = b"".join([
        *S.quantise_ft(sd2["ft.weight"], sd2["ft.bias"]),
        *S.quantise_l1(sd2["l1.weight"], sd2["l1.bias"]),
        *S.quantise_l2(sd2["l2.weight"], sd2["l2.bias"]),
        *S.quantise_out(sd2["output.weight"], sd2["output.bias"]),
    ])

    with open(nnue_path, "rb") as f:
        buf = f.read()
    _version, _hash, desc_len = struct.unpack_from("<III", buf, 0)
    actual = buf[12 + desc_len:]

    ok = expected == actual
    print(f"[1] byte-exact match (.nnue payload == quantise_*(checkpoint)): {'PASS' if ok else 'FAIL'}")
    if not ok:
        n = min(len(expected), len(actual))
        for i in range(n):
            if expected[i] != actual[i]:
                print(f"    first diff at byte {i}/{n}")
                break
    return net, sd2


def check_clipping(sd2):
    print("\n[2] weight-clipping rate per layer (fraction hitting the int8 boundary):")
    layers = [
        ("l1.weight", sd2["l1.weight"], L1_SCALE, -128, 127),
        ("l2.weight", sd2["l2.weight"], L2_SCALE, -128, 127),
        ("output.weight", sd2["output.weight"], OUT_SCALE, -128, 127),
    ]
    for name, w, scale, lo, hi in layers:
        scaled = w * scale
        frac = ((scaled < lo) | (scaled > hi)).float().mean().item()
        flag = "  <-- HIGH, scale likely miscalibrated" if frac > 0.05 else ""
        print(f"    {name:16s} clipped={frac*100:6.3f}%  |w| median={w.abs().median():.3f} "
              f"max={w.abs().max():.3f}{flag}")


def halfkp_indices(board):
    white_king = board.king(chess.WHITE)
    black_king_mirror = chess.square_mirror(board.king(chess.BLACK))
    w_idx, b_idx = [], []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None or p.piece_type == chess.KING:
            continue
        type_idx = PIECE_TYPE_ORDER.index(p.piece_type)
        color_idx = 0 if p.color == chess.WHITE else 1
        pidx = type_idx * 2 + color_idx
        w_idx.append(white_king * 640 + sq * 10 + pidx)
        m_sq = chess.square_mirror(sq)
        m_pidx = pidx ^ 1
        b_idx.append(black_king_mirror * 640 + m_sq * 10 + m_pidx)
    return w_idx, b_idx


def load_quantized(nnue_path):
    with open(nnue_path, "rb") as f:
        buf = f.read()
    off = 0
    _version, _hash, desc_len = struct.unpack_from("<III", buf, off); off += 12
    off += desc_len
    ft_w = np.frombuffer(buf, dtype="<i2", count=HALFKP_SIZE * H, offset=off).reshape(HALFKP_SIZE, H).astype(np.int32); off += HALFKP_SIZE * H * 2
    ft_b = np.frombuffer(buf, dtype="<i2", count=H, offset=off).astype(np.int32); off += H * 2
    l1_w = np.frombuffer(buf, dtype="<i1", count=32 * H * 2, offset=off).reshape(32, H * 2).astype(np.int32); off += 32 * H * 2
    l1_b = np.frombuffer(buf, dtype="<i4", count=32, offset=off).astype(np.int64); off += 32 * 4
    l2_w = np.frombuffer(buf, dtype="<i1", count=32 * 32, offset=off).reshape(32, 32).astype(np.int32); off += 32 * 32
    l2_b = np.frombuffer(buf, dtype="<i4", count=32, offset=off).astype(np.int64); off += 32 * 4
    out_w = np.frombuffer(buf, dtype="<i1", count=NUM_BUCKETS * 32, offset=off).reshape(NUM_BUCKETS, 32).astype(np.int32); off += NUM_BUCKETS * 32
    out_b = np.frombuffer(buf, dtype="<i4", count=NUM_BUCKETS, offset=off).astype(np.int64); off += NUM_BUCKETS * 4
    return ft_w, ft_b, l1_w, l1_b, l2_w, l2_b, out_w, out_b


def quantized_eval(fen, weights):
    ft_w, ft_b, l1_w, l1_b, l2_w, l2_b, out_w, out_b = weights
    board = chess.Board(fen)
    w_idx, b_idx = halfkp_indices(board)
    acc_w = ft_b.copy() + ft_w[w_idx].sum(axis=0)
    acc_b = ft_b.copy() + ft_w[b_idx].sum(axis=0)

    stm_white = board.turn == chess.WHITE
    first, second = (acc_w, acc_b) if stm_white else (acc_b, acc_w)
    x = np.concatenate([np.clip(first, 0, FT_SCALE), np.clip(second, 0, FT_SCALE)]).astype(np.int64)

    l1_acc = l1_b + (l1_w.astype(np.int64) * x[None, :]).sum(axis=1)
    l1_val = np.clip(l1_acc >> 7, 0, L1_SCALE).astype(np.int64)

    l2_acc = l2_b + (l2_w.astype(np.int64) * l1_val[None, :]).sum(axis=1)
    l2_val = np.clip(l2_acc // L1_SCALE, 0, L2_SCALE).astype(np.int64)

    num_pieces = len(board.piece_map()) - 2
    bucket = min(NUM_BUCKETS - 1, max(0, num_pieces * NUM_BUCKETS // 32))

    score = out_b[bucket] + int((out_w[bucket].astype(np.int64) * l2_val).sum())
    return score / (L2_SCALE * OUT_SCALE), bucket


def check_eval_agreement(net):
    print("\n[3] quantized vs. float eval on diverse test positions:")
    weights = load_quantized(args.nnue)
    max_diff = 0.0
    for fen in TEST_FENS:
        board = chess.Board(fen)
        w_idx, b_idx = data.board_to_halfkp(board)
        w_t = torch.from_numpy(w_idx).long().unsqueeze(0)
        b_t = torch.from_numpy(b_idx).long().unsqueeze(0)
        bucket = data.get_bucket(board, num_buckets=NUM_BUCKETS)
        stm_white = board.turn == chess.WHITE
        with torch.no_grad():
            float_cp = net(w_t, b_t, bucket=bucket, stm_white=stm_white).item()
        quant_cp, q_bucket = quantized_eval(fen, weights)
        diff = abs(float_cp - quant_cp)
        max_diff = max(max_diff, diff)
        flag = "  <-- suspiciously large" if diff > 15 else ""
        print(f"    {fen[:40]:40s} float={float_cp:8.2f}  quant={quant_cp:8.2f}  diff={diff:6.2f}{flag}")
    verdict = "PASS (quantization-noise level)" if max_diff <= 15 else "FAIL (real divergence, not noise)"
    print(f"\n    max |float - quant| = {max_diff:.2f} cp  ->  {verdict}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pt", default="checkpoints/nnue_best.pt")
    p.add_argument("--nnue", default="checkpoints/model.nnue")
    args = p.parse_args()

    net, sd2 = check_byte_exact(args.pt, args.nnue)
    check_clipping(sd2)
    check_eval_agreement(net)
