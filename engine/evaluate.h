/*
 * evaluate.h — NNUE inference for the HalfKP engine.
 *
 * Responsibilities:
 *   • NNUEWeights  — quantised weight storage loaded from a .nnue file.
 *   • Accumulator  — incremental HalfKP accumulator with push/pop stack.
 *   • nnue_eval()  — integer forward pass → centipawns.
 *   • load_nnue()  — binary .nnue loader (format produced by serialize.py).
 *   • get_bucket() — material-count → output bucket index.
 *
 * Quantisation (must match serialize.py exactly):
 *   FT  : int16 weights/biases;  activations clamped [0, FT_SCALE=127]
 *   L1  : int8  weights, int32 biases; divide accumulation by FT_SCALE
 *   L2  : same pattern, divide by L1_SCALE
 *   Out : int8  weights, int32 biases; divide by OUT_SCALE → centipawns
 *
 * .nnue binary layout (little-endian):
 *   uint32  version   = 0x00000001
 *   uint32  hash      (ignored on load)
 *   uint32  desc_len
 *   char[]  description (desc_len bytes, skipped)
 *   int16[HALFKP_SIZE * H]   ft_weights  (row-major)
 *   int16[H]                 ft_biases
 *   int8[L1_SIZE * H*2]      l1_weights  (row-major [32, H*2])
 *   int32[L1_SIZE]           l1_biases
 *   int8[L2_SIZE * L1_SIZE]  l2_weights
 *   int32[L2_SIZE]           l2_biases
 *   int8[B * L2_SIZE]        out_weights (row-major [B, 32])
 *   int32[B]                 out_biases
 *
 * Dependencies: types.h, position.h
 */

#pragma once
#include "types.h"
#include "position.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ─────────────────────────── weight storage ────────────────────────────────

struct NNUEWeights {
    int16_t ft_w[HALFKP_SIZE][MAX_HIDDEN];   // [feature_idx][hidden]
    int16_t ft_b[MAX_HIDDEN];                 // accumulator bias

    int8_t  l1_w[L1_SIZE][MAX_HIDDEN * 2];   // [out][in]
    int32_t l1_b[L1_SIZE];

    int8_t  l2_w[L2_SIZE][L1_SIZE];
    int32_t l2_b[L2_SIZE];

    int8_t  out_w[MAX_BUCKETS][L2_SIZE];
    int32_t out_b[MAX_BUCKETS];

    int  hidden_size;   // runtime H (≤ MAX_HIDDEN)
    int  num_buckets;   // runtime B (≤ MAX_BUCKETS)
    bool loaded;

    NNUEWeights() : hidden_size(256), num_buckets(8), loaded(false) {
        memset(ft_w,  0, sizeof ft_w);
        memset(ft_b,  0, sizeof ft_b);
        memset(l1_w,  0, sizeof l1_w);
        memset(l1_b,  0, sizeof l1_b);
        memset(l2_w,  0, sizeof l2_w);
        memset(l2_b,  0, sizeof l2_b);
        memset(out_w, 0, sizeof out_w);
        memset(out_b, 0, sizeof out_b);
    }
};

// Single global weight set shared by all search threads.
inline NNUEWeights g_weights;

// ─────────────────────────── .nnue loader ──────────────────────────────────

inline bool load_nnue(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "info string Cannot open " << path << "\n";
        return false;
    }
    f.seekg(0, std::ios::end);
    std::size_t sz = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(sz));
    if (!f) {
        std::cerr << "info string Read error on " << path << "\n";
        return false;
    }

    std::size_t off = 0;
    auto read32 = [&]() -> uint32_t {
        uint32_t v;
        memcpy(&v, buf.data() + off, 4);
        off += 4;
        return v;
    };

    uint32_t version  = read32();
    /* hash */          read32();   // not validated here
    uint32_t desc_len = read32();
    if (version != 0x00000001u) {
        std::cerr << "info string Unknown .nnue version " << version << "\n";
        return false;
    }
    off += desc_len;   // skip description string

    const int H = g_weights.hidden_size;
    const int B = g_weights.num_buckets;

    // ft.weight [HALFKP_SIZE, H]  int16
    for (int i = 0; i < HALFKP_SIZE; ++i)
        for (int j = 0; j < H; ++j) {
            int16_t v;
            memcpy(&v, buf.data() + off, 2);
            off += 2;
            g_weights.ft_w[i][j] = v;
        }
    // ft.bias [H]  int16
    for (int i = 0; i < H; ++i) {
        int16_t v;
        memcpy(&v, buf.data() + off, 2);
        off += 2;
        g_weights.ft_b[i] = v;
    }
    // l1.weight [L1_SIZE, H*2]  int8
    for (int o = 0; o < L1_SIZE; ++o)
        for (int i = 0; i < H * 2; ++i)
            g_weights.l1_w[o][i] = static_cast<int8_t>(buf[off++]);
    // l1.bias [L1_SIZE]  int32
    for (int i = 0; i < L1_SIZE; ++i) {
        int32_t v;
        memcpy(&v, buf.data() + off, 4);
        off += 4;
        g_weights.l1_b[i] = v;
    }
    // l2.weight [L2_SIZE, L1_SIZE]  int8
    for (int o = 0; o < L2_SIZE; ++o)
        for (int i = 0; i < L1_SIZE; ++i)
            g_weights.l2_w[o][i] = static_cast<int8_t>(buf[off++]);
    // l2.bias [L2_SIZE]  int32
    for (int i = 0; i < L2_SIZE; ++i) {
        int32_t v;
        memcpy(&v, buf.data() + off, 4);
        off += 4;
        g_weights.l2_b[i] = v;
    }
    // out.weight [B, L2_SIZE]  int8
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < L2_SIZE; ++i)
            g_weights.out_w[b][i] = static_cast<int8_t>(buf[off++]);
    // out.bias [B]  int32
    for (int b = 0; b < B; ++b) {
        int32_t v;
        memcpy(&v, buf.data() + off, 4);
        off += 4;
        g_weights.out_b[b] = v;
    }

    g_weights.loaded = true;
    std::cerr << "info string Loaded " << path
              << "  H=" << H << "  buckets=" << B << "\n";
    return true;
}

// ─────────────────────────── accumulator entry ─────────────────────────────

// One frame on the accumulator stack: two half-vectors, one per king perspective.
struct AccEntry {
    int16_t w[MAX_HIDDEN];   // white-king perspective
    int16_t b[MAX_HIDDEN];   // black-king perspective
};

// ─────────────────────────── Accumulator ───────────────────────────────────
//
// Mirrors the Python NNUE forward pass:
//   acc = sum(ft_weights[feature]) + ft_bias   (per king perspective)
//
// Incremental update: on each do_move we only add/subtract the features
// that changed (moved piece + captured piece).  King moves and castling
// require a full rebuild because the king square itself is part of every
// HalfKP feature index on that side.
//
// Usage (mirrors old nnue_engine.cpp Accumulator):
//   acc.reset(pos);                 — full rebuild from scratch
//   acc.push_incremental(pos, m);   — before do_move() for non-king moves
//   acc.push_full(pos_after);       — after do_move() for king / castle moves
//   acc.null_push();                — for null-move pruning
//   acc.pop();                      — after undo_move()

class Accumulator {
public:
    static constexpr int MAX_STACK = MAX_PLY + 4;

    AccEntry stack[MAX_STACK];
    int      top = 0;

    // ── Full rebuild from pos (call at search root or after king move) ────────
    void reset(const Position &pos) {
        top = 0;
        _build(pos, stack[0]);
    }

    // ── Incremental push for a quiet / non-king move ──────────────────────────
    // Call BEFORE do_move() so we can read the pre-move board.
    void push_incremental(const Position &pos, Move m) {
        assert(top + 1 < MAX_STACK);
        AccEntry &prev = stack[top];
        AccEntry &next = stack[top + 1];
        const NNUEWeights &W = g_weights;
        const int H = W.hidden_size;

        // Start from previous accumulator
        memcpy(next.w, prev.w, H * sizeof(int16_t));
        memcpy(next.b, prev.b, H * sizeof(int16_t));

        Square    from     = move_from(m);
        Square    to       = move_to(m);
        PieceType pt       = pos.piece_on[from];
        Color     us       = pos.stm;
        Color     them     = static_cast<Color>(!us);
        int       pidx     = PIECE_IDX[pt][us];
        int       wk       = pos.king_sq[WHITE];
        int       bk_mir   = sq_mirror(pos.king_sq[BLACK]);
        MoveFlag  flags    = move_flags(m);

        // Remove the moving piece from its source square
        _sub(next, halfkp_w(wk,     from, pidx), halfkp_b(bk_mir, from, pidx), H, W);
        // Add it to the destination square
        _add(next, halfkp_w(wk,     to,   pidx), halfkp_b(bk_mir, to,   pidx), H, W);

        // Remove captured piece (if any)
        if (flags == MF_EP) {
            Square cap_sq = (us == WHITE) ? to - 8 : to + 8;
            int    cp_idx = PIECE_IDX[PAWN][them];
            _sub(next, halfkp_w(wk, cap_sq, cp_idx), halfkp_b(bk_mir, cap_sq, cp_idx), H, W);
        } else {
            PieceType captured = pos.piece_on[to];
            if (captured != NO_PIECE) {
                int cp_idx = PIECE_IDX[captured][them];
                _sub(next, halfkp_w(wk, to, cp_idx), halfkp_b(bk_mir, to, cp_idx), H, W);
            }
        }

        // Handle promotion: the pawn at `to` becomes a different piece type
        if (flags == MF_PROMO) {
            PieceType promo = move_promo(m);
            // The pawn add above used pt=PAWN; we remove it and add the promoted piece
            _sub(next, halfkp_w(wk, to, pidx),              halfkp_b(bk_mir, to, pidx),              H, W);
            _add(next, halfkp_w(wk, to, PIECE_IDX[promo][us]), halfkp_b(bk_mir, to, PIECE_IDX[promo][us]), H, W);
        }

        top++;
    }

    // ── Full rebuild for king / castling moves (call AFTER do_move()) ─────────
    void push_full(const Position &pos_after) {
        assert(top + 1 < MAX_STACK);
        _build(pos_after, stack[top + 1]);
        top++;
    }

    // ── Null-move push: just copy the current top (no piece moved) ────────────
    void null_push() {
        assert(top + 1 < MAX_STACK);
        const int H = g_weights.hidden_size;
        memcpy(stack[top + 1].w, stack[top].w, H * sizeof(int16_t));
        memcpy(stack[top + 1].b, stack[top].b, H * sizeof(int16_t));
        top++;
    }

    void pop() {
        assert(top > 0);
        top--;
    }

private:
    // Full accumulator build from a position
    void _build(const Position &pos, AccEntry &e) const {
        const NNUEWeights &W = g_weights;
        const int H = W.hidden_size;
        const int wk     = pos.king_sq[WHITE];
        const int bk_mir = sq_mirror(pos.king_sq[BLACK]);

        // Initialise with bias
        for (int i = 0; i < H; ++i) e.w[i] = e.b[i] = W.ft_b[i];

        // Accumulate every non-king piece
        for (int c = 0; c < 2; ++c) {
            for (int pt = 0; pt < 5; ++pt) {   // PAWN..QUEEN
                Bitboard pieces = pos.bb[c][pt];
                while (pieces) {
                    Square sq  = pop_lsb(pieces);
                    int    idx = PIECE_IDX[pt][c];
                    int    iw  = halfkp_w(wk,     sq, idx);
                    int    ib  = halfkp_b(bk_mir, sq, idx);
                    for (int i = 0; i < H; ++i) {
                        e.w[i] += W.ft_w[iw][i];
                        e.b[i] += W.ft_w[ib][i];
                    }
                }
            }
        }
    }

    // Subtract feature vector for index `iw` / `ib`
    static void _sub(AccEntry &e, int iw, int ib, int H, const NNUEWeights &W) {
        for (int i = 0; i < H; ++i) {
            e.w[i] -= W.ft_w[iw][i];
            e.b[i] -= W.ft_w[ib][i];
        }
    }

    // Add feature vector for index `iw` / `ib`
    static void _add(AccEntry &e, int iw, int ib, int H, const NNUEWeights &W) {
        for (int i = 0; i < H; ++i) {
            e.w[i] += W.ft_w[iw][i];
            e.b[i] += W.ft_w[ib][i];
        }
    }
};

// ─────────────────────────── NNUE forward pass ─────────────────────────────

// Runs the quantised FT→L1→L2→output forward pass on the current accumulator
// frame.  Returns the evaluation in centipawns from the side-to-move's POV.
//
//   stm_white : true if the side to move is White (controls perspective order)
//   bucket    : material-count output bucket [0, num_buckets)

inline int nnue_eval(const AccEntry &acc, bool stm_white, int bucket) {
    const NNUEWeights &W = g_weights;
    const int H = W.hidden_size;

    // ── Feature transformer clipped ReLU — STM perspective first ─────────────
    // Mirrors model.py forward(): first = STM, second = opponent.
    const int16_t *first  = stm_white ? acc.w : acc.b;
    const int16_t *second = stm_white ? acc.b : acc.w;

    int8_t x[MAX_HIDDEN * 2];
    for (int i = 0; i < H; ++i) {
        x[i]     = static_cast<int8_t>(std::clamp(static_cast<int>(first[i]),  0, FT_SCALE));
        x[H + i] = static_cast<int8_t>(std::clamp(static_cast<int>(second[i]), 0, FT_SCALE));
    }

    // ── L1 ────────────────────────────────────────────────────────────────────
    int8_t l1[L1_SIZE];
    for (int o = 0; o < L1_SIZE; ++o) {
        int32_t s = W.l1_b[o];
        for (int i = 0; i < H * 2; ++i)
            s += static_cast<int32_t>(W.l1_w[o][i]) * static_cast<int32_t>(x[i]);
        // Bias was pre-scaled by FT_SCALE*L1_SCALE; divide to land in [0, L1_SCALE]
        l1[o] = static_cast<int8_t>(std::clamp(s / FT_SCALE, 0, L1_SCALE));
    }

    // ── L2 ────────────────────────────────────────────────────────────────────
    int8_t l2[L2_SIZE];
    for (int o = 0; o < L2_SIZE; ++o) {
        int32_t s = W.l2_b[o];
        for (int i = 0; i < L1_SIZE; ++i)
            s += static_cast<int32_t>(W.l2_w[o][i]) * static_cast<int32_t>(l1[i]);
        l2[o] = static_cast<int8_t>(std::clamp(s / L1_SCALE, 0, L2_SCALE));
    }

    // ── Output ────────────────────────────────────────────────────────────────
    int32_t score = W.out_b[bucket];
    for (int i = 0; i < L2_SIZE; ++i)
        score += static_cast<int32_t>(W.out_w[bucket][i]) * static_cast<int32_t>(l2[i]);

    return score / OUT_SCALE;   // centipawns
}

// ─────────────────────────── bucket selection ──────────────────────────────

// Maps the number of non-king pieces on the board to a material bucket index.
// Matches get_bucket() in nnue_engine.cpp and the training data convention.
inline int get_bucket(const Position &pos) {
    int n = pos.piece_count;
    int b = n * g_weights.num_buckets / 32;
    return std::min(b, g_weights.num_buckets - 1);
}
