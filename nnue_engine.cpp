/*
 * nnue_engine.cpp — Self-contained HalfKP NNUE chess engine (UCI)
 *
 * Replaces the Python search_core.c + python-chess approach with a fully
 * native C++ engine that does everything in one binary:
 *
 *   • Bitboard move generator (magic bitboards for sliders)
 *   • Integer NNUE inference matching serialize.py quantisation:
 *       FT  : int16 weights/biases,  activations clamped [0, 127]
 *       L1  : int8  weights, int32 biases, >> FT_SCALE_SHIFT, clamp [0, 127]
 *       L2  : same pattern
 *       Out : int8  weights, int32 biases, / OUT_SCALE → centipawn score
 *   • Incremental HalfKP accumulator with push/pop stack
 *   • Full search: negamax PVS, aspiration windows, iterative deepening,
 *       NMP, LMR, futility, reverse futility, quiescence + delta pruning,
 *       TT (Zobrist), killer moves, history heuristic
 *   • UCI protocol (position, go, ucinewgame, setoption)
 *
 * Build:
 *   Apple Silicon (ARM64 / M1-M4):
 *     clang++ -O3 -march=native -flto -std=c++17 -o nnue_engine nnue_engine.cpp
 *     (-march=native enables NEON automatically; -mavx2/-mfma are x86-only flags)
 *
 *   Linux / Windows x86-64:
 *     g++ -O3 -march=native -mavx2 -mfma -flto -std=c++17 -o nnue_engine nnue_engine.cpp
 *
 * Usage:
 *   ./nnue_engine                          # enters UCI loop
 *   (set NN file via: setoption name NNFile value nn-xxxx.nnue)
 *
 * .nnue format (little-endian, produced by serialize.py):
 *   uint32  version   = 0x00000001
 *   uint32  hash      = SHA-256 first 4 bytes of whole buffer
 *   uint32  desc_len
 *   char[]  description
 *   int16[HALFKP_SIZE * H]  ft_weights  (row-major)
 *   int16[H]                ft_biases
 *   int8[32 * H*2]          l1_weights  (row-major [32, H*2])
 *   int32[32]               l1_biases
 *   int8[32 * 32]           l2_weights
 *   int32[32]               l2_biases
 *   int8[B * 32]            out_weights (row-major [B, 32])
 *   int32[B]                out_biases
 *
 * Quantisation constants (must match serialize.py):
 *   FT_SCALE  = 127   (FT activations ∈ [0, 127])
 *   L1_SCALE  = 64
 *   L2_SCALE  = 64
 *   OUT_SCALE = 0.6   (NOT 600 — see serialize.py's OUT_SCALE comment: the
 *                      output layer's weights run an order of magnitude
 *                      larger than the other layers' because its inputs are
 *                      bounded to [0,1] by clipped_relu. 600 clipped ~97% of
 *                      the output layer to the int8 boundary.)
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <pthread.h>
#include <string>
#include <vector>

// ── ARM NEON SIMD ──────────────────────────────────────────────────────────
// On Apple Silicon and other ARM64 targets (-march=native) we write explicit
// NEON intrinsics for the NNUE forward pass rather than relying on the
// auto-vectoriser, which cannot reliably handle the multi-layer accumulator
// loop pattern. Fallback scalar path is used on non-ARM64 platforms.
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  #include <arm_neon.h>
  #define USE_NEON 1
#else
  #define USE_NEON 0
#endif

// ─────────────────────────── compile-time limits ───────────────────────────

static constexpr int HALFKP_SIZE  = 40960;
static constexpr int MAX_HIDDEN   = 512;
static constexpr int L1_SIZE      = 32;
static constexpr int L2_SIZE      = 32;
static constexpr int MAX_BUCKETS  = 32;
static constexpr int MAX_PLY      = 128;
static constexpr int MAX_MOVES    = 256;
static constexpr int N_SQUARES    = 64;
static constexpr int N_PIECE_TYPES = 10;

// NNUE fixed-point constants (must match serialize.py)
static constexpr int FT_SCALE    = 127;
static constexpr int L1_SCALE    = 64;
static constexpr int L2_SCALE    = 64;
static constexpr double OUT_SCALE = 0.6;  // see the comment above and in serialize.py

// Search constants
static constexpr int INF         = 1'000'000;
static constexpr int MATE_SCORE  = 900'000;
static constexpr int DRAW_SCORE  = 0;
// Sentinel for "no static eval computed" (e.g. in-check nodes). Chosen well
// outside any realistic centipawn or mate score range so it can never be
// mistaken for a real evaluation.
static constexpr int EVAL_NONE   = -32'001;

static constexpr int NMP_MIN_DEPTH  = 3;
[[maybe_unused]] static constexpr int NMP_R_BASE     = 3;
[[maybe_unused]] static constexpr int NMP_R_DIV      = 4;
static constexpr int LMR_MIN_DEPTH  = 3;
static constexpr int LMR_FULL_MOVES = 3;
static constexpr int ASP_WINDOW     = 50;
static constexpr int ASP_MAX_TRIES  = 4;
static constexpr int DELTA_MARGIN   = 200;
// Max check extensions stacked along a single search line. Was previously
// unbounded (every checking move got +1 unconditionally) — see the
// check-extension gate in the move loop for why this exists.
static constexpr int CHECK_EXT_BUDGET = 16;
[[maybe_unused]] static constexpr int RFP_MARGIN     = 120;

static constexpr int FUTILITY_MARGIN[5] = {0, 100, 200, 300, 400};

// Late Move Pruning: maximum quiet moves to try at each depth before giving up.
// Indexed by [depth][improving].  Formula: (3 + depth*depth) * (1 + improving) / 2.
// At depth 1 we try 2 or 4 quiets; at depth 4 we try 9 or 19; above depth 8 LMP
// is disabled (the move count never gets that high in practice before a cut).
static constexpr int LMP_MOVES[9][2] = {
    {0, 0},   // depth 0 — unused
    {2, 4},   // depth 1
    {4, 7},   // depth 2
    {6, 12},  // depth 3
    {9, 19},  // depth 4
    {13, 27}, // depth 5
    {18, 37}, // depth 6
    {24, 49}, // depth 7
    {31, 63}, // depth 8
};

// Piece values (centipawns) indexed by piece type 0..5 (PAWN=0..QUEEN=4,KING=5)
static constexpr int PIECE_VALUE[6] = {100, 320, 330, 500, 900, 20000};

// ─────────────────────────── types ─────────────────────────────────────────

using Bitboard = uint64_t;
using Square   = int;
using Move     = uint32_t;  // from(6)|to(6)|promo(4)|flags(4)

enum Color { WHITE = 0, BLACK = 1 };
enum PieceType { PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5, NO_PIECE=6 };

// Move flags
static constexpr uint32_t MF_NORMAL   = 0;
static constexpr uint32_t MF_CASTLE   = 1;
static constexpr uint32_t MF_EP       = 2;
static constexpr uint32_t MF_PROMO    = 3;

static inline Square    move_from(Move m)   { return m & 63; }
static inline Square    move_to(Move m)     { return (m >> 6) & 63; }
static inline PieceType move_promo(Move m)  { return (PieceType)((m >> 12) & 15); }
static inline uint32_t  move_flags(Move m)  { return (m >> 16) & 15; }
static inline Move make_move(Square f, Square t, PieceType promo=NO_PIECE, uint32_t flags=MF_NORMAL) {
    return f | (t << 6) | ((uint32_t)promo << 12) | (flags << 16);
}

static constexpr Move NULL_MOVE = 0xFFFFFFFF;
static constexpr Move NO_MOVE   = 0;

// ─────────────────────────── bit utilities ─────────────────────────────────

static inline Square lsb(Bitboard b) { return __builtin_ctzll(b); }
[[maybe_unused]] static inline Square msb(Bitboard b) { return 63 ^ __builtin_clzll(b); }
static inline int    popcount(Bitboard b) { return __builtin_popcountll(b); }
static inline Bitboard pop_lsb(Bitboard &b) {
    Square s = lsb(b);
    b &= b - 1;
    return (Bitboard)1 << s;
}

static constexpr Bitboard FILE_A = 0x0101010101010101ULL;
static constexpr Bitboard FILE_H = 0x8080808080808080ULL;
static constexpr Bitboard RANK_1 = 0x00000000000000FFULL;
[[maybe_unused]] static constexpr Bitboard RANK_2 = 0x000000000000FF00ULL;
static constexpr Bitboard RANK_3 = 0x0000000000FF0000ULL;
static constexpr Bitboard RANK_6 = 0x0000FF0000000000ULL;
[[maybe_unused]] static constexpr Bitboard RANK_7 = 0x00FF000000000000ULL;
static constexpr Bitboard RANK_8 = 0xFF00000000000000ULL;

static inline Bitboard sq_bb(Square s) { return (Bitboard)1 << s; }
static inline int      sq_file(Square s) { return s & 7; }
static inline int      sq_rank(Square s) { return s >> 3; }
static inline Square   sq_mirror(Square s) { return s ^ 56; }
static inline Square   make_sq(int file, int rank) { return rank * 8 + file; }

// ─────────────────────────── magic bitboards ───────────────────────────────

struct Magic {
    Bitboard  mask;
    Bitboard  magic;
    Bitboard *attacks;
    int       shift;
    inline Bitboard operator()(Bitboard occ) const {
        return attacks[((occ & mask) * magic) >> shift];
    }
};

static Magic g_rook_magic[64];
static Magic g_bishop_magic[64];
// Rook mask has at most 12 relevant bits (corner squares) → 2^12 = 4096 entries max.
// Bishop mask has at most 9 relevant bits (centre squares) → 2^9 = 512 entries max.
// These are hard upper bounds; no magic index can exceed them.
// Memory: 64*4096*8 = 2.10 MB (rook) + 64*512*8 = 0.26 MB (bishop) = 2.36 MB total.
static Bitboard g_rook_attacks_table[64][4096];
static Bitboard g_bishop_attacks_table[64][512];
static Bitboard g_knight_attacks[64];
static Bitboard g_king_attacks[64];
static Bitboard g_pawn_attacks[2][64];  // [color][sq]

// Sliding attack helpers (for magic initialisation)
static Bitboard sliding_attacks(Square sq, Bitboard occ, const int deltas[][2], int ndelta) {
    Bitboard ret = 0;
    int r = sq_rank(sq), f = sq_file(sq);
    for (int d = 0; d < ndelta; d++) {
        int dr = deltas[d][0], df = deltas[d][1];
        for (int nr = r+dr, nf = f+df;
             nr >= 0 && nr < 8 && nf >= 0 && nf < 8;
             nr += dr, nf += df) {
            Square ns = make_sq(nf, nr);
            ret |= sq_bb(ns);
            if (occ & sq_bb(ns)) break;
        }
    }
    return ret;
}

// Pre-computed magic numbers (known good)
// Rook magics — generated and validated with variable shift (64 - popcount(mask)).
// Zero destructive collisions verified across all 64 squares and all occupancy subsets.
static const Bitboard ROOK_MAGICS[64] = {
    0x2080001620400080ULL,  // sq= 0 shift=52 bits=12
    0x0440029000406004ULL,  // sq= 1 shift=53 bits=11
    0x0080200010008008ULL,  // sq= 2 shift=53 bits=11
    0x0100082010000502ULL,  // sq= 3 shift=53 bits=11
    0x0480040002800800ULL,  // sq= 4 shift=53 bits=11
    0x8880140012008001ULL,  // sq= 5 shift=53 bits=11
    0x42002C0588020029ULL,  // sq= 6 shift=53 bits=11
    0x8200022213048044ULL,  // sq= 7 shift=52 bits=12
    0x0409800180C00020ULL,  // sq= 8 shift=53 bits=11
    0x0012002041020094ULL,  // sq= 9 shift=54 bits=10
    0x5002801000A00080ULL,  // sq=10 shift=54 bits=10
    0x8802002012000B40ULL,  // sq=11 shift=54 bits=10
    0x0000800400800800ULL,  // sq=12 shift=54 bits=10
    0x0002808004000200ULL,  // sq=13 shift=54 bits=10
    0x2401010401000200ULL,  // sq=14 shift=54 bits=10
    0x08C0802080004100ULL,  // sq=15 shift=53 bits=11
    0x0080114001406000ULL,  // sq=16 shift=53 bits=11
    0x00D0004000200048ULL,  // sq=17 shift=54 bits=10
    0x0000410010200109ULL,  // sq=18 shift=54 bits=10
    0x0008008008100080ULL,  // sq=19 shift=54 bits=10
    0x4004008008000481ULL,  // sq=20 shift=54 bits=10
    0x8104008002008004ULL,  // sq=21 shift=54 bits=10
    0x4000040008820150ULL,  // sq=22 shift=54 bits=10
    0x0000020000440081ULL,  // sq=23 shift=53 bits=11
    0x1000800080204002ULL,  // sq=24 shift=53 bits=11
    0x0040008100310040ULL,  // sq=25 shift=54 bits=10
    0x0208104200220080ULL,  // sq=26 shift=54 bits=10
    0x2100100100210008ULL,  // sq=27 shift=54 bits=10
    0x2280110100080004ULL,  // sq=28 shift=54 bits=10
    0x8000020080040080ULL,  // sq=29 shift=54 bits=10
    0x9800888400100102ULL,  // sq=30 shift=54 bits=10
    0x0000802080004100ULL,  // sq=31 shift=53 bits=11
    0x2020004000808000ULL,  // sq=32 shift=53 bits=11
    0x0823C02002401000ULL,  // sq=33 shift=54 bits=10
    0x1611004011002001ULL,  // sq=34 shift=54 bits=10
    0x0000081001002100ULL,  // sq=35 shift=54 bits=10
    0x1010080080800400ULL,  // sq=36 shift=54 bits=10
    0x8000040080800200ULL,  // sq=37 shift=54 bits=10
    0x0300100204008801ULL,  // sq=38 shift=54 bits=10
    0x2812050486000044ULL,  // sq=39 shift=53 bits=11
    0x2000814000218005ULL,  // sq=40 shift=53 bits=11
    0x0410002000404000ULL,  // sq=41 shift=54 bits=10
    0x3810004020010100ULL,  // sq=42 shift=54 bits=10
    0x2001001000210008ULL,  // sq=43 shift=54 bits=10
    0x2048000400088080ULL,  // sq=44 shift=54 bits=10
    0x8000040002008080ULL,  // sq=45 shift=54 bits=10
    0x0800020001008080ULL,  // sq=46 shift=54 bits=10
    0x0180208400520021ULL,  // sq=47 shift=53 bits=11
    0x0080448001002300ULL,  // sq=48 shift=53 bits=11
    0x4204804001002500ULL,  // sq=49 shift=54 bits=10
    0x014120118A420200ULL,  // sq=50 shift=54 bits=10
    0x0100080080100080ULL,  // sq=51 shift=54 bits=10
    0x9008000804008080ULL,  // sq=52 shift=54 bits=10
    0x1000040080020080ULL,  // sq=53 shift=54 bits=10
    0x1284280230010400ULL,  // sq=54 shift=54 bits=10
    0x0804110084204200ULL,  // sq=55 shift=53 bits=11
    0x0042214100508001ULL,  // sq=56 shift=52 bits=12
    0x0300204000108101ULL,  // sq=57 shift=53 bits=11
    0x0100200100100C41ULL,  // sq=58 shift=53 bits=11
    0x0420852100100009ULL,  // sq=59 shift=53 bits=11
    0x008200A41008204AULL,  // sq=60 shift=53 bits=11
    0x0282000448011062ULL,  // sq=61 shift=53 bits=11
    0x0008008228100144ULL,  // sq=62 shift=53 bits=11
    0x0A000C0085005022ULL,  // sq=63 shift=52 bits=12
};

// Bishop magics — generated and validated with variable shift (64 - popcount(mask)).
// Each magic was exhaustively verified: for every occupancy subset of that square's
// mask, the index (occ & mask) * magic >> shift maps to a unique or constructively-
// aliased attack bitboard with zero destructive collisions across all 64 squares.
static const Bitboard BISHOP_MAGICS[64] = {
    0x0018911026004900ULL,  // sq= 0 shift=58 bits=6
    0x0820020082248000ULL,  // sq= 1 shift=59 bits=5
    0x0442043040800800ULL,  // sq= 2 shift=59 bits=5
    0x0084404484000000ULL,  // sq= 3 shift=59 bits=5
    0x0184042000840C10ULL,  // sq= 4 shift=59 bits=5
    0x0001010840082470ULL,  // sq= 5 shift=59 bits=5
    0x001884C820106008ULL,  // sq= 6 shift=59 bits=5
    0x8010108815082000ULL,  // sq= 7 shift=58 bits=6
    0x0000040808212400ULL,  // sq= 8 shift=59 bits=5
    0x8800080220840100ULL,  // sq= 9 shift=59 bits=5
    0x0000240845810800ULL,  // sq=10 shift=59 bits=5
    0x0400022A02008520ULL,  // sq=11 shift=59 bits=5
    0x0200020211004009ULL,  // sq=12 shift=59 bits=5
    0x200A408220200002ULL,  // sq=13 shift=59 bits=5
    0x0000004108A01080ULL,  // sq=14 shift=59 bits=5
    0x1000082406185404ULL,  // sq=15 shift=59 bits=5
    0x2108804110412214ULL,  // sq=16 shift=59 bits=5
    0x4010990204154400ULL,  // sq=17 shift=59 bits=5
    0x0081021204010A00ULL,  // sq=18 shift=57 bits=7
    0x0344041824001020ULL,  // sq=19 shift=57 bits=7
    0x110C800400A01000ULL,  // sq=20 shift=57 bits=7
    0x0101080200822010ULL,  // sq=21 shift=57 bits=7
    0x0002028051142040ULL,  // sq=22 shift=59 bits=5
    0x0090903024040200ULL,  // sq=23 shift=59 bits=5
    0x0410284110329001ULL,  // sq=24 shift=59 bits=5
    0x0210080050810100ULL,  // sq=25 shift=59 bits=5
    0x1402110008014400ULL,  // sq=26 shift=57 bits=7
    0x0060080002081010ULL,  // sq=27 shift=55 bits=9
    0x0380840002020201ULL,  // sq=28 shift=55 bits=9
    0x580202008148060AULL,  // sq=29 shift=57 bits=7
    0x080440C413080640ULL,  // sq=30 shift=59 bits=5
    0x000A00281A090102ULL,  // sq=31 shift=59 bits=5
    0x001014240260089AULL,  // sq=32 shift=59 bits=5
    0x902828040042C420ULL,  // sq=33 shift=59 bits=5
    0x0300180400C20400ULL,  // sq=34 shift=57 bits=7
    0x0089E08400080210ULL,  // sq=35 shift=55 bits=9
    0x2028020400001010ULL,  // sq=36 shift=55 bits=9
    0x2020808602210100ULL,  // sq=37 shift=57 bits=7
    0x0468080045092100ULL,  // sq=38 shift=59 bits=5
    0x0004010048002400ULL,  // sq=39 shift=59 bits=5
    0x00886A0220005000ULL,  // sq=40 shift=59 bits=5
    0x0081009004501000ULL,  // sq=41 shift=59 bits=5
    0x001E010402100100ULL,  // sq=42 shift=57 bits=7
    0x8040122018012108ULL,  // sq=43 shift=57 bits=7
    0x0000080104000841ULL,  // sq=44 shift=57 bits=7
    0x04440804880A2100ULL,  // sq=45 shift=57 bits=7
    0x40082200A2200400ULL,  // sq=46 shift=59 bits=5
    0x0001010408840108ULL,  // sq=47 shift=59 bits=5
    0x0081040120092000ULL,  // sq=48 shift=59 bits=5
    0x4709128A10060080ULL,  // sq=49 shift=59 bits=5
    0x0087310401310020ULL,  // sq=50 shift=59 bits=5
    0x4040080084040000ULL,  // sq=51 shift=59 bits=5
    0x0202419021024408ULL,  // sq=52 shift=59 bits=5
    0x2040418408008804ULL,  // sq=53 shift=59 bits=5
    0x002920244C820802ULL,  // sq=54 shift=59 bits=5
    0x00A0046080810054ULL,  // sq=55 shift=59 bits=5
    0x2002008421111005ULL,  // sq=56 shift=58 bits=6
    0x0029020100821020ULL,  // sq=57 shift=59 bits=5
    0x1102001040441004ULL,  // sq=58 shift=59 bits=5
    0x38A0822082420220ULL,  // sq=59 shift=59 bits=5
    0x1001000088102400ULL,  // sq=60 shift=59 bits=5
    0x0082029242104100ULL,  // sq=61 shift=59 bits=5
    0x0081098890041040ULL,  // sq=62 shift=59 bits=5
    0x60092104058C0500ULL,  // sq=63 shift=58 bits=6
};

static Bitboard rook_mask(Square sq) {
    Bitboard result = 0;
    int r = sq_rank(sq), f = sq_file(sq);
    for (int nr = r + 1; nr <= 6; nr++) result |= sq_bb(make_sq(f, nr));
    for (int nr = r - 1; nr >= 1; nr--) result |= sq_bb(make_sq(f, nr));
    for (int nf = f + 1; nf <= 6; nf++) result |= sq_bb(make_sq(nf, r));
    for (int nf = f - 1; nf >= 1; nf--) result |= sq_bb(make_sq(nf, r));
    return result;
}

static Bitboard bishop_mask(Square sq) {
    Bitboard result = 0;
    int r = sq_rank(sq), f = sq_file(sq);
    for (int nr = r+1, nf = f+1; nr <= 6 && nf <= 6; nr++, nf++) result |= sq_bb(make_sq(nf, nr));
    for (int nr = r+1, nf = f-1; nr <= 6 && nf >= 1; nr++, nf--) result |= sq_bb(make_sq(nf, nr));
    for (int nr = r-1, nf = f+1; nr >= 1 && nf <= 6; nr--, nf++) result |= sq_bb(make_sq(nf, nr));
    for (int nr = r-1, nf = f-1; nr >= 1 && nf >= 1; nr--, nf--) result |= sq_bb(make_sq(nf, nr));
    return result;
}

static void init_magics() {
    static const int ROOK_DELTAS[4][2]   = {{1,0},{-1,0},{0,1},{0,-1}};
    static const int BISHOP_DELTAS[4][2] = {{1,1},{-1,1},{1,-1},{-1,-1}};

    for (Square sq = 0; sq < 64; sq++) {
        // Knight
        {
            int r = sq_rank(sq), f = sq_file(sq);
            Bitboard b = 0;
            static const int D[8][2] = {{2,1},{2,-1},{-2,1},{-2,-1},{1,2},{1,-2},{-1,2},{-1,-2}};
            for (auto &d : D) {
                int nr = r+d[0], nf = f+d[1];
                if (nr>=0&&nr<8&&nf>=0&&nf<8) b |= sq_bb(make_sq(nf,nr));
            }
            g_knight_attacks[sq] = b;
        }
        // King
        {
            int r = sq_rank(sq), f = sq_file(sq);
            Bitboard b = 0;
            for (int dr = -1; dr <= 1; dr++)
                for (int df = -1; df <= 1; df++)
                    if ((dr||df) && r+dr>=0&&r+dr<8&&f+df>=0&&f+df<8)
                        b |= sq_bb(make_sq(f+df, r+dr));
            g_king_attacks[sq] = b;
        }
        // Pawn attacks
        {
            int r = sq_rank(sq), f = sq_file(sq);
            if (r < 7) {
                if (f > 0) g_pawn_attacks[WHITE][sq] |= sq_bb(make_sq(f-1, r+1));
                if (f < 7) g_pawn_attacks[WHITE][sq] |= sq_bb(make_sq(f+1, r+1));
            }
            if (r > 0) {
                if (f > 0) g_pawn_attacks[BLACK][sq] |= sq_bb(make_sq(f-1, r-1));
                if (f < 7) g_pawn_attacks[BLACK][sq] |= sq_bb(make_sq(f+1, r-1));
            }
        }
        // Rook magic
        {
            auto &m = g_rook_magic[sq];
            m.magic   = ROOK_MAGICS[sq];
            m.attacks = g_rook_attacks_table[sq];
            m.mask    = rook_mask(sq);
            m.shift   = 64 - popcount(m.mask);
            // Rook mask has at most 12 bits → max index = 2^12-1 = 4095. Assert fits.
            assert(popcount(m.mask) <= 12 && "rook mask exceeds 12 bits — table overflow");
            Bitboard occ = 0;
            do {
                int idx = (int)(((occ & m.mask) * m.magic) >> m.shift);
                assert(idx >= 0 && idx < 4096 && "rook magic index out of bounds");
                m.attacks[idx] = sliding_attacks(sq, occ, ROOK_DELTAS, 4);
                occ = (occ - 1) & m.mask;
            } while (occ);
        }
        // Bishop magic
        {
            auto &m = g_bishop_magic[sq];
            m.magic   = BISHOP_MAGICS[sq];
            m.attacks = g_bishop_attacks_table[sq];
            m.mask    = bishop_mask(sq);
            m.shift   = 64 - popcount(m.mask);
            // Bishop mask has at most 9 bits → max index = 2^9-1 = 511. Assert fits.
            assert(popcount(m.mask) <= 9 && "bishop mask exceeds 9 bits — table overflow");
            Bitboard occ = 0;
            do {
                int idx = (int)(((occ & m.mask) * m.magic) >> m.shift);
                assert(idx >= 0 && idx < 512 && "bishop magic index out of bounds");
                m.attacks[idx] = sliding_attacks(sq, occ, BISHOP_DELTAS, 4);
                occ = (occ - 1) & m.mask;
            } while (occ);
        }
    }

    // ── Runtime correctness self-test ──────────────────────────────────────
    // Verify every magic produces zero destructive collisions across all
    // occupancy subsets. A collision is destructive when two different
    // occupancies that yield the same magic index produce different attack
    // sets — an impossible-to-detect error at query time that silently
    // returns wrong attacks and corrupts move generation.
    // This runs once at startup (< 1 ms) and aborts immediately on failure.
    for (Square sq = 0; sq < 64; sq++) {
        // Rook
        {
            const auto &m = g_rook_magic[sq];
            static Bitboard verify_table[4096];
            static bool verify_used[4096];
            memset(verify_used, 0, sizeof(bool) * 4096);
            memset(verify_table, 0, sizeof(Bitboard) * 4096);
            Bitboard occ = 0;
            do {
                int idx = (int)(((occ & m.mask) * m.magic) >> m.shift);
                Bitboard atk = sliding_attacks(sq, occ, ROOK_DELTAS, 4);
                if (!verify_used[idx]) {
                    verify_used[idx]  = true;
                    verify_table[idx] = atk;
                } else {
                    assert(verify_table[idx] == atk && "rook magic destructive collision — magic table is corrupt");
                }
                occ = (occ - 1) & m.mask;
            } while (occ);
        }
        // Bishop
        {
            const auto &m = g_bishop_magic[sq];
            static Bitboard verify_table[512];
            static bool verify_used[512];
            memset(verify_used, 0, sizeof(bool) * 512);
            memset(verify_table, 0, sizeof(Bitboard) * 512);
            Bitboard occ = 0;
            do {
                int idx = (int)(((occ & m.mask) * m.magic) >> m.shift);
                Bitboard atk = sliding_attacks(sq, occ, BISHOP_DELTAS, 4);
                if (!verify_used[idx]) {
                    verify_used[idx]  = true;
                    verify_table[idx] = atk;
                } else {
                    assert(verify_table[idx] == atk && "bishop magic destructive collision — magic table is corrupt");
                }
                occ = (occ - 1) & m.mask;
            } while (occ);
        }
    }
}

// Use pre-computed magic bitboard tables (filled by init_magics).
// The Magic::operator() computes: attacks[((occ & mask) * magic) >> shift]
// This is an O(1) lookup — no ray loops, no branches, no board iteration.
static inline Bitboard rook_attacks(Square sq, Bitboard occ) {
    return g_rook_magic[sq](occ);
}
static inline Bitboard bishop_attacks(Square sq, Bitboard occ) {
    return g_bishop_magic[sq](occ);
}
static inline Bitboard queen_attacks(Square sq, Bitboard occ)  { return rook_attacks(sq,occ)|bishop_attacks(sq,occ); }

// ─────────────────────────── Zobrist ───────────────────────────────────────

static uint64_t g_zobrist_piece[2][6][64];  // [color][type][sq]
static uint64_t g_zobrist_ep[64];
static uint64_t g_zobrist_castle[16];
static uint64_t g_zobrist_stm;

static void init_zobrist() {
    // LCG seeded with a constant for reproducibility
    uint64_t s = 0xDEADBEEFCAFE1234ULL;
    auto next = [&]() -> uint64_t {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 0x2545F4914F6CDD1DULL;
    };
    for (int c = 0; c < 2; c++)
        for (int p = 0; p < 6; p++)
            for (int sq = 0; sq < 64; sq++)
                g_zobrist_piece[c][p][sq] = next();
    for (int sq = 0; sq < 64; sq++) g_zobrist_ep[sq] = next();
    for (int i  = 0; i  < 16; i++)  g_zobrist_castle[i] = next();
    g_zobrist_stm = next();
}

// ─────────────────────────── Board ─────────────────────────────────────────

static constexpr int CR_WK = 1, CR_WQ = 2, CR_BK = 4, CR_BQ = 8;

struct StateInfo {
    uint64_t   hash;
    uint64_t   pawn_hash;       // XOR of g_zobrist_piece over pawns only (for correction history)
    int        ep_square;       // -1 = none
    int        castle_rights;   // bitmask
    int        halfmove_clock;
    PieceType  captured_piece;  // for undo
};

struct Board {
    Bitboard   bb[2][6];        // [color][piece_type]
    Bitboard   occupied[2];
    Bitboard   all;
    PieceType  piece_on[64];
    Color      color_on[64];
    int        king_sq[2];
    Color      stm;
    int        piece_count;     // cached non-king piece count for bucket lookup

    // Fixed-size history stack: MAX_PLY search depth + up to 512 game moves.
    // Using a plain array avoids heap allocation on every do_move/undo_move,
    // which was causing millions of malloc/free calls per second in search.
    static constexpr int HIST_MAX = MAX_PLY + 512;
    StateInfo history[HIST_MAX];
    int       history_top;  // index of next free slot (== current depth)
    StateInfo cur;

    // ── setup ──────────────────────────────────────────────────────────────

    void clear() {
        memset(bb, 0, sizeof bb);
        memset(occupied, 0, sizeof occupied);
        all = 0;
        for (int i = 0; i < 64; i++) { piece_on[i] = NO_PIECE; color_on[i] = WHITE; }
        king_sq[WHITE] = king_sq[BLACK] = 0;
        piece_count = 0;
        stm = WHITE;
        cur = {0, 0, -1, 0, 0, NO_PIECE};
        history_top = 0;
    }

    void place(Color c, PieceType pt, Square sq) {
        bb[c][pt] |= sq_bb(sq);
        occupied[c] |= sq_bb(sq);
        all |= sq_bb(sq);
        piece_on[sq] = pt;
        color_on[sq] = c;
        if (pt == KING) king_sq[c] = sq;
        else piece_count++;
        cur.hash ^= g_zobrist_piece[c][pt][sq];
        if (pt == PAWN) cur.pawn_hash ^= g_zobrist_piece[c][pt][sq];
    }

    void remove(Color c, PieceType pt, Square sq) {
        bb[c][pt] &= ~sq_bb(sq);
        occupied[c] &= ~sq_bb(sq);
        all &= ~sq_bb(sq);
        piece_on[sq] = NO_PIECE;
        color_on[sq] = WHITE;  // reset to avoid stale color after castle undo
        if (pt != KING) piece_count--;
        cur.hash ^= g_zobrist_piece[c][pt][sq];
        if (pt == PAWN) cur.pawn_hash ^= g_zobrist_piece[c][pt][sq];
    }

    void set_from_fen(const std::string &fen) {
        clear();
        std::istringstream ss(fen);
        std::string board_part, stm_str, castle_str, ep_str;
        int halfmove, fullmove;
        ss >> board_part >> stm_str >> castle_str >> ep_str >> halfmove >> fullmove;

        int rank = 7, file = 0;
        for (char c : board_part) {
            if (c == '/') { rank--; file = 0; }
            else if (c >= '1' && c <= '8') { file += c - '0'; }
            else {
                Color  col = isupper(c) ? WHITE : BLACK;
                PieceType pt;
                switch (tolower(c)) {
                    case 'p': pt = PAWN;   break;
                    case 'n': pt = KNIGHT; break;
                    case 'b': pt = BISHOP; break;
                    case 'r': pt = ROOK;   break;
                    case 'q': pt = QUEEN;  break;
                    case 'k': pt = KING;   break;
                    default:  pt = NO_PIECE; break;
                }
                if (pt != NO_PIECE) place(col, pt, make_sq(file, rank));
                file++;
            }
        }
        stm = (stm_str == "w") ? WHITE : BLACK;
        if (stm == BLACK) cur.hash ^= g_zobrist_stm;

        cur.castle_rights = 0;
        for (char c : castle_str) {
            if (c == 'K') cur.castle_rights |= CR_WK;
            if (c == 'Q') cur.castle_rights |= CR_WQ;
            if (c == 'k') cur.castle_rights |= CR_BK;
            if (c == 'q') cur.castle_rights |= CR_BQ;
        }
        cur.hash ^= g_zobrist_castle[cur.castle_rights];

        cur.ep_square = -1;
        if (ep_str != "-") {
            int f = ep_str[0] - 'a';
            int r = ep_str[1] - '1';
            cur.ep_square = make_sq(f, r);
            cur.hash ^= g_zobrist_ep[cur.ep_square];
        }
        cur.halfmove_clock = halfmove;
        cur.captured_piece = NO_PIECE;
    }

    // ── attack queries ──────────────────────────────────────────────────────

    bool sq_attacked(Square sq, Color by) const {
        Bitboard occ = all;
        if (g_pawn_attacks[!by][sq]   & bb[by][PAWN])   return true;
        if (g_knight_attacks[sq]       & bb[by][KNIGHT]) return true;
        if (g_king_attacks[sq]         & bb[by][KING])   return true;
        if (bishop_attacks(sq, occ)    & (bb[by][BISHOP]|bb[by][QUEEN])) return true;
        if (rook_attacks(sq, occ)      & (bb[by][ROOK]  |bb[by][QUEEN])) return true;
        return false;
    }

    bool in_check() const { return sq_attacked(king_sq[stm], (Color)!stm); }

    // ── move making ─────────────────────────────────────────────────────────

    void do_move(Move m) {
        history[history_top++] = cur;

        cur.hash ^= g_zobrist_castle[cur.castle_rights];
        if (cur.ep_square >= 0) {
            cur.hash ^= g_zobrist_ep[cur.ep_square];
            cur.ep_square = -1;
        }

        if (m == NULL_MOVE) {
            stm = (Color)!stm;
            cur.hash ^= g_zobrist_stm;
            cur.captured_piece = NO_PIECE;
            cur.hash ^= g_zobrist_castle[cur.castle_rights];
            return;
        }

        Square    from  = move_from(m);
        Square    to    = move_to(m);
        PieceType pt    = piece_on[from];
        Color     us    = stm;
        Color     them  = (Color)!us;
        uint32_t  flags = move_flags(m);
        PieceType promo = move_promo(m);

        cur.captured_piece = NO_PIECE;

        // Capture
        if (flags == MF_EP) {
            Square cap_sq = us == WHITE ? to - 8 : to + 8;
            cur.captured_piece = PAWN;
            remove(them, PAWN, cap_sq);
        } else if (piece_on[to] != NO_PIECE) {
            cur.captured_piece = piece_on[to];
            remove(them, cur.captured_piece, to);
        }

        // Move piece
        remove(us, pt, from);
        PieceType landing = (flags == MF_PROMO) ? promo : pt;
        place(us, landing, to);

        // Castling rook
        if (flags == MF_CASTLE) {
            if (to == 6)  { remove(us, ROOK, 7);  place(us, ROOK, 5);  } // WK
            if (to == 2)  { remove(us, ROOK, 0);  place(us, ROOK, 3);  } // WQ
            if (to == 62) { remove(us, ROOK, 63); place(us, ROOK, 61); } // BK
            if (to == 58) { remove(us, ROOK, 56); place(us, ROOK, 59); } // BQ
        }

        // En passant square
        if (pt == PAWN && abs((int)to - (int)from) == 16) {
            cur.ep_square = us == WHITE ? from + 8 : from - 8;
            cur.hash ^= g_zobrist_ep[cur.ep_square];
        }

        // Castle rights update
        static const int CASTLE_MASK[64] = {
            ~CR_WQ,15,15,15,~(CR_WK|CR_WQ),15,15,~CR_WK,
            15,15,15,15,15,15,15,15,
            15,15,15,15,15,15,15,15,
            15,15,15,15,15,15,15,15,
            15,15,15,15,15,15,15,15,
            15,15,15,15,15,15,15,15,
            15,15,15,15,15,15,15,15,
            ~CR_BQ,15,15,15,~(CR_BK|CR_BQ),15,15,~CR_BK
        };
        cur.castle_rights &= CASTLE_MASK[from] & CASTLE_MASK[to];
        cur.hash ^= g_zobrist_castle[cur.castle_rights];

        cur.halfmove_clock = (pt == PAWN || cur.captured_piece != NO_PIECE)
                             ? 0 : cur.halfmove_clock + 1;

        stm = them;
        cur.hash ^= g_zobrist_stm;
    }

    void undo_move(Move m) {
        stm = (Color)!stm;
        Color us   = stm;
        Color them = (Color)!us;

        if (m != NULL_MOVE) {
            Square    from  = move_from(m);
            Square    to    = move_to(m);
            uint32_t  flags = move_flags(m);
            PieceType promo = move_promo(m);
            PieceType landing = (flags == MF_PROMO) ? promo : piece_on[to];
            // Read captured_piece from cur (set by do_move) BEFORE we restore
            // cur from history — history.back().captured_piece is the state from
            // BEFORE do_move was called (since history.push_back(cur) happens
            // first in do_move, before cur.captured_piece is updated).
            PieceType captured = cur.captured_piece;

            remove(us, landing, to);
            PieceType restore = (flags == MF_PROMO) ? PAWN : landing;
            place(us, restore, from);
            if (restore == KING) king_sq[us] = from;

            if (flags == MF_CASTLE) {
                if (to == 6)  { remove(us, ROOK, 5);  place(us, ROOK, 7);  }
                if (to == 2)  { remove(us, ROOK, 3);  place(us, ROOK, 0);  }
                if (to == 62) { remove(us, ROOK, 61); place(us, ROOK, 63); }
                if (to == 58) { remove(us, ROOK, 59); place(us, ROOK, 56); }
            }

            if (flags == MF_EP) {
                Square cap_sq = us == WHITE ? to - 8 : to + 8;
                place(them, PAWN, cap_sq);
            } else if (captured != NO_PIECE) {
                place(them, captured, to);
            }
        }

        cur = history[--history_top];
    }

    // ── move generation ─────────────────────────────────────────────────────

    int gen_moves(Move *list, bool captures_only = false) {
        int n = 0;
        Color us   = stm;
        Color them = (Color)!us;
        Bitboard occ  = all;
        Bitboard mine = occupied[us];
        Bitboard theirs = occupied[them];
        Bitboard targets = captures_only ? theirs : ~mine;

        // Pawns
        {
            Bitboard pawns = bb[us][PAWN];
            if (us == WHITE) {
                Bitboard push1 = (pawns << 8) & ~occ;
                Bitboard promo_push = push1 & RANK_8;
                push1 &= ~RANK_8;
                Bitboard push2 = (((pawns << 8) & ~occ) & RANK_3) << 8 & ~occ;
                Bitboard cap_r = ((pawns & ~FILE_H) << 9) & theirs;
                Bitboard cap_l = ((pawns & ~FILE_A) << 7) & theirs;
                Bitboard promo_r = cap_r & RANK_8; cap_r &= ~RANK_8;
                Bitboard promo_l = cap_l & RANK_8; cap_l &= ~RANK_8;

                while (!captures_only && push1) { Square to = lsb(push1); pop_lsb(push1); list[n++] = make_move(to-8, to); }
                while (!captures_only && push2) { Square to = lsb(push2); pop_lsb(push2); list[n++] = make_move(to-16, to); }
                while (cap_r) { Square to = lsb(cap_r); pop_lsb(cap_r); list[n++] = make_move(to-9, to); }
                while (cap_l) { Square to = lsb(cap_l); pop_lsb(cap_l); list[n++] = make_move(to-7, to); }
                // Promotions
                auto add_promo = [&](Square from, Square to) {
                    list[n++] = make_move(from, to, QUEEN,  MF_PROMO);
                    list[n++] = make_move(from, to, ROOK,   MF_PROMO);
                    list[n++] = make_move(from, to, BISHOP, MF_PROMO);
                    list[n++] = make_move(from, to, KNIGHT, MF_PROMO);
                };
                while (promo_push) { Square to = lsb(promo_push); pop_lsb(promo_push); add_promo(to-8, to); }
                while (promo_r)    { Square to = lsb(promo_r); pop_lsb(promo_r); add_promo(to-9, to); }
                while (promo_l)    { Square to = lsb(promo_l); pop_lsb(promo_l); add_promo(to-7, to); }
                // EP
                if (cur.ep_square >= 0) {
                    Bitboard ep = sq_bb(cur.ep_square);
                    if (pawns & ((ep >> 7) & ~FILE_A)) list[n++] = make_move(cur.ep_square-7, cur.ep_square, NO_PIECE, MF_EP);
                    if (pawns & ((ep >> 9) & ~FILE_H)) list[n++] = make_move(cur.ep_square-9, cur.ep_square, NO_PIECE, MF_EP);
                }
            } else { // BLACK
                Bitboard push1 = (pawns >> 8) & ~occ;
                Bitboard push2 = (((pawns >> 8) & ~occ) & RANK_6) >> 8 & ~occ;
                Bitboard promo_push = push1 & RANK_1;
                push1 &= ~RANK_1;
                Bitboard cap_r = ((pawns & ~FILE_A) >> 9) & theirs;
                Bitboard cap_l = ((pawns & ~FILE_H) >> 7) & theirs;
                Bitboard promo_r = cap_r & RANK_1; cap_r &= ~RANK_1;
                Bitboard promo_l = cap_l & RANK_1; cap_l &= ~RANK_1;

                while (!captures_only && push1) { Square to = lsb(push1); pop_lsb(push1); list[n++] = make_move(to+8, to); }
                while (!captures_only && push2) { Square to = lsb(push2); pop_lsb(push2); list[n++] = make_move(to+16, to); }
                while (cap_r) { Square to = lsb(cap_r); pop_lsb(cap_r); list[n++] = make_move(to+9, to); }
                while (cap_l) { Square to = lsb(cap_l); pop_lsb(cap_l); list[n++] = make_move(to+7, to); }
                auto add_promo = [&](Square from, Square to) {
                    list[n++] = make_move(from, to, QUEEN,  MF_PROMO);
                    list[n++] = make_move(from, to, ROOK,   MF_PROMO);
                    list[n++] = make_move(from, to, BISHOP, MF_PROMO);
                    list[n++] = make_move(from, to, KNIGHT, MF_PROMO);
                };
                while (promo_push) { Square to = lsb(promo_push); pop_lsb(promo_push); add_promo(to+8, to); }
                while (promo_r)    { Square to = lsb(promo_r); pop_lsb(promo_r); add_promo(to+9, to); }
                while (promo_l)    { Square to = lsb(promo_l); pop_lsb(promo_l); add_promo(to+7, to); }
                if (cur.ep_square >= 0) {
                    Bitboard ep = sq_bb(cur.ep_square);
                    if (pawns & ((ep << 9) & ~FILE_A)) list[n++] = make_move(cur.ep_square+9, cur.ep_square, NO_PIECE, MF_EP);
                    if (pawns & ((ep << 7) & ~FILE_H)) list[n++] = make_move(cur.ep_square+7, cur.ep_square, NO_PIECE, MF_EP);
                }
            }
        }

        // Knights
        {
            Bitboard kn = bb[us][KNIGHT];
            while (kn) {
                Square sq = lsb(kn); pop_lsb(kn);
                Bitboard att = g_knight_attacks[sq] & targets;
                while (att) { Square to = lsb(att); pop_lsb(att); list[n++] = make_move(sq, to); }
            }
        }
        // Bishops
        {
            Bitboard bi = bb[us][BISHOP];
            while (bi) {
                Square sq = lsb(bi); pop_lsb(bi);
                Bitboard att = bishop_attacks(sq, occ) & targets;
                while (att) { Square to = lsb(att); pop_lsb(att); list[n++] = make_move(sq, to); }
            }
        }
        // Rooks
        {
            Bitboard ro = bb[us][ROOK];
            while (ro) {
                Square sq = lsb(ro); pop_lsb(ro);
                Bitboard att = rook_attacks(sq, occ) & targets;
                while (att) { Square to = lsb(att); pop_lsb(att); list[n++] = make_move(sq, to); }
            }
        }
        // Queens
        {
            Bitboard qu = bb[us][QUEEN];
            while (qu) {
                Square sq = lsb(qu); pop_lsb(qu);
                Bitboard att = queen_attacks(sq, occ) & targets;
                while (att) { Square to = lsb(att); pop_lsb(att); list[n++] = make_move(sq, to); }
            }
        }
        // King
        {
            Square sq  = king_sq[us];
            Bitboard att = g_king_attacks[sq] & targets;
            while (att) { Square to = lsb(att); pop_lsb(att); list[n++] = make_move(sq, to); }
            // Castling
            if (!captures_only) {
                if (us == WHITE) {
                    if ((cur.castle_rights & CR_WK) && !(occ & 0x60ULL) &&
                        !sq_attacked(4, BLACK) && !sq_attacked(5, BLACK) && !sq_attacked(6, BLACK))
                        list[n++] = make_move(4, 6, NO_PIECE, MF_CASTLE);
                    if ((cur.castle_rights & CR_WQ) && !(occ & 0x0EULL) &&
                        !sq_attacked(4, BLACK) && !sq_attacked(3, BLACK) && !sq_attacked(2, BLACK))
                        list[n++] = make_move(4, 2, NO_PIECE, MF_CASTLE);
                } else {
                    if ((cur.castle_rights & CR_BK) && !(occ & 0x6000000000000000ULL) &&
                        !sq_attacked(60, WHITE) && !sq_attacked(61, WHITE) && !sq_attacked(62, WHITE))
                        list[n++] = make_move(60, 62, NO_PIECE, MF_CASTLE);
                    if ((cur.castle_rights & CR_BQ) && !(occ & 0x0E00000000000000ULL) &&
                        !sq_attacked(60, WHITE) && !sq_attacked(59, WHITE) && !sq_attacked(58, WHITE))
                        list[n++] = make_move(60, 58, NO_PIECE, MF_CASTLE);
                }
            }
        }

        // Filter illegal moves (leaves king in check).
        // Fast path for normal moves: compute occ after the move and re-check
        // king safety with magic lookups only — no do_move/undo_move overhead.
        // King moves and en-passant use the full do/undo path (rare, complex).
        int legal = 0;
        Square ksq = king_sq[us];
        Bitboard their_diag  = bb[them][BISHOP] | bb[them][QUEEN];
        Bitboard their_orth  = bb[them][ROOK]   | bb[them][QUEEN];

        for (int i = 0; i < n; i++) {
            Move mv         = list[i];
            Square from     = move_from(mv);
            Square to       = move_to(mv);
            uint32_t flags  = move_flags(mv);
            bool is_king    = (piece_on[from] == KING);
            bool is_ep      = (flags == MF_EP);
            bool is_castle  = (flags == MF_CASTLE);

            if (!is_king && !is_ep && !is_castle) {
                // Occupancy after move: remove `from`, remove captured piece at
                // `to` (if any), add piece at `to`.  Net: flip from off, to is
                // already set in `all` if capturing (noop), else set it.
                Bitboard occ_after = (all & ~sq_bb(from)) | sq_bb(to);

                // Slider attackers of king, masking away the captured piece at `to`
                Bitboard diag_att = their_diag & ~sq_bb(to);
                Bitboard orth_att = their_orth & ~sq_bb(to);

                bool safe =
                    !(bishop_attacks(ksq, occ_after) & diag_att) &&
                    !(rook_attacks(ksq, occ_after)   & orth_att) &&
                    !(g_knight_attacks[ksq] & bb[them][KNIGHT] & ~sq_bb(to)) &&
                    !(g_pawn_attacks[us][ksq] & bb[them][PAWN] & ~sq_bb(to)) &&
                    !(g_king_attacks[ksq] & bb[them][KING]);
                if (safe) list[legal++] = mv;
            } else {
                // King move, castling, or en-passant: use full do/undo
                Square check_sq = is_king ? to : ksq;
                do_move(mv);
                if (!sq_attacked(check_sq, them)) list[legal++] = mv;
                undo_move(mv);
            }
        }
        return legal;
    }

    // ── material count for bucket ────────────────────────────────────────────

    int material_count() const { return piece_count; }  // O(1) via cached counter

    // ── repetition / 50-move / insufficient material ────────────────────────

    // Returns true when neither side has mating material.
    // Covers: KK, KBK, KNK, KBKB (same-colour bishops), KBsK (multiple same-colour bishops).
    // Does NOT cover KNNK (technically a win is possible but extremely unlikely, and almost
    // every engine treats it as a draw — this is the universally accepted heuristic).
    bool is_insufficient_material() const {
        // Any pawns, rooks, or queens on the board → mating material exists.
        for (int c = 0; c < 2; c++) {
            if (bb[c][PAWN] | bb[c][ROOK] | bb[c][QUEEN]) return false;
        }
        // At this point we have only kings, knights, and bishops.
        int w_knights = popcount(bb[WHITE][KNIGHT]);
        int b_knights = popcount(bb[BLACK][KNIGHT]);
        int w_bishops = popcount(bb[WHITE][BISHOP]);
        int b_bishops = popcount(bb[BLACK][BISHOP]);
        int w_minors  = w_knights + w_bishops;
        int b_minors  = b_knights + b_bishops;

        // KK
        if (w_minors == 0 && b_minors == 0) return true;

        // KBK or KNK (one minor total)
        if (w_minors + b_minors == 1) return true;

        // KNNvK — two knights one side, lone king other side
        if (w_minors == 2 && w_knights == 2 && b_minors == 0) return true;
        if (b_minors == 2 && b_knights == 2 && w_minors == 0) return true;

        // KBBvK — two bishops one side, lone king other side.
        // Draw only if both bishops are on the same colour squares.
        // Opposite-colour bishops can force mate, so not a draw.
        if (w_minors == 2 && w_bishops == 2 && b_minors == 0) {
            static constexpr Bitboard LIGHT_SQ = 0x55AA55AA55AA55AAULL;
            Bitboard wb = bb[WHITE][BISHOP];
            if ((wb & LIGHT_SQ) == 0) return true;   // both on dark
            if ((wb & ~LIGHT_SQ) == 0) return true;  // both on light
            return false; // opposite colour — mating material
        }
        if (b_minors == 2 && b_bishops == 2 && w_minors == 0) {
            static constexpr Bitboard LIGHT_SQ = 0x55AA55AA55AA55AAULL;
            Bitboard bb_ = bb[BLACK][BISHOP];
            if ((bb_ & LIGHT_SQ) == 0) return true;
            if ((bb_ & ~LIGHT_SQ) == 0) return true;
            return false;
        }

        return false;
    }

    // is_draw(): used at root / game-over check. Requires true threefold (count>=2).
    bool is_draw() const {
        if (cur.halfmove_clock >= 100) return true;
        if (is_insufficient_material()) return true;
        return is_repetition(/*root_distance=*/history_top);
    }

    // is_repetition(root_distance): ply-aware repetition check for search.
    //   root_distance = history.size() at the root of the current search.
    //   A position that first appeared WITHIN the search tree (index >= root_distance)
    //   is a draw on the SECOND occurrence (twofold), because the engine can steer
    //   away from it — treating it as drawn prevents search from being fooled.
    //   A position from GAME history (index < root_distance) requires threefold.
    //   This also fixes the TT interaction: call this before TT cutoff in search.
    bool is_repetition(int root_distance) const {
        int count = 0;
        int sz = history_top;
        for (int i = sz - 2; i >= 0; i -= 2) {
            if (history[i].hash == cur.hash) {
                ++count;
                // If this occurrence is within the search tree, twofold is enough.
                if (i >= root_distance) return true;
                // Otherwise need threefold (two prior occurrences in game history).
                if (count >= 2) return true;
            }
            if (history[i].halfmove_clock == 0) break;
        }
        return false;
    }

    // ── UCI move parsing ─────────────────────────────────────────────────────

    Move parse_uci(const std::string &s) const {
        if (s.size() < 4) return NO_MOVE;
        int from = make_sq(s[0]-'a', s[1]-'1');
        int to   = make_sq(s[2]-'a', s[3]-'1');
        PieceType promo = NO_PIECE;
        if (s.size() >= 5) {
            switch (s[4]) {
                case 'q': promo = QUEEN;  break;
                case 'r': promo = ROOK;   break;
                case 'b': promo = BISHOP; break;
                case 'n': promo = KNIGHT; break;
            }
        }
        // Detect flags
        uint32_t flags = MF_NORMAL;
        if (piece_on[from] == PAWN && to == cur.ep_square) flags = MF_EP;
        if (piece_on[from] == KING && abs((int)from - (int)to) == 2) flags = MF_CASTLE;
        if (promo != NO_PIECE) flags = MF_PROMO;
        return make_move(from, to, promo, flags);
    }

    std::string move_uci(Move m) const {
        static const char FILES[] = "abcdefgh";
        static const char RANKS[] = "12345678";
        static const char PROMO[] = " pnbrqk";
        Square from = move_from(m), to = move_to(m);
        char s[6] = {FILES[sq_file(from)], RANKS[sq_rank(from)],
                     FILES[sq_file(to)],   RANKS[sq_rank(to)], 0, 0};
        PieceType pr = move_promo(m);
        if (move_flags(m) == MF_PROMO) { s[4] = PROMO[pr+1]; s[5] = 0; }
        return std::string(s);
    }
};

// ─────────────────────────── Perft ─────────────────────────────────────────
// Movegen regression test: gen_moves() already filters to strictly legal
// moves (see the "Filter illegal moves" pass inside it), so perft(1) is
// just the legal move count and every deeper level is a straightforward
// recursive sum — no separate legality re-check needed here. Exposed via
// the "perft" UCI debug command below; run against known node counts for
// standard test positions (startpos, Kiwipete, etc.) before trusting any
// movegen change (e.g. staged move generation).
static uint64_t perft(Board &board, int depth) {
    if (depth == 0) return 1;
    Move moves[MAX_MOVES];
    int n = board.gen_moves(moves);
    if (depth == 1) return (uint64_t)n;
    uint64_t total = 0;
    for (int i = 0; i < n; i++) {
        board.do_move(moves[i]);
        total += perft(board, depth - 1);
        board.undo_move(moves[i]);
    }
    return total;
}

// Same as perft() but prints a per-root-move node-count breakdown to stdout
// ("divide"), the standard technique for isolating which specific move a
// movegen bug is hiding under when a plain perft total doesn't match.
static uint64_t perft_divide(Board &board, int depth) {
    Move moves[MAX_MOVES];
    int n = board.gen_moves(moves);
    uint64_t total = 0;
    for (int i = 0; i < n; i++) {
        board.do_move(moves[i]);
        uint64_t cnt = (depth <= 1) ? 1 : perft(board, depth - 1);
        board.undo_move(moves[i]);
        std::cout << board.move_uci(moves[i]) << ": " << cnt << "\n";
        total += cnt;
    }
    return total;
}

// ─────────────────────────── NNUE weights ──────────────────────────────────

struct NNUEWeights {
    // Stored quantised as loaded from .nnue file
    int16_t ft_w[HALFKP_SIZE][MAX_HIDDEN];
    int16_t ft_b[MAX_HIDDEN];
    int8_t  l1_w[MAX_HIDDEN * 2][L1_SIZE];  // [in][out] layout: inner loop over outputs (contiguous)
    int32_t l1_b[L1_SIZE];
    int8_t  l2_w[L2_SIZE][L1_SIZE];
    int32_t l2_b[L2_SIZE];
    int8_t  out_w[MAX_BUCKETS][L2_SIZE];
    int32_t out_b[MAX_BUCKETS];
    int     hidden_size;
    int     num_buckets;
    bool    loaded;

    NNUEWeights() : hidden_size(256), num_buckets(8), loaded(false) {}
};

static NNUEWeights g_weights;

// ─────────────────────────── HalfKP index ──────────────────────────────────
// Must match serialize.py PIECE_TYPE_IDX and halfkp_w/halfkp_b

static const int PIECE_IDX[6][2] = {
    {0,1},{2,3},{4,5},{6,7},{8,9},{-1,-1}  // PAWN..QUEEN; KING unused
};

static inline int halfkp_w(int king_sq, int piece_sq, int pidx) {
    return king_sq * (N_SQUARES * N_PIECE_TYPES) + piece_sq * N_PIECE_TYPES + pidx;
}
static inline int halfkp_b(int king_sq_mirror, int piece_sq, int pidx) {
    return king_sq_mirror * (N_SQUARES * N_PIECE_TYPES)
         + sq_mirror(piece_sq) * N_PIECE_TYPES + (pidx ^ 1);
}

// ─────────────────────────── Accumulator ───────────────────────────────────

struct AccEntry {
    // int16 is sufficient: FT weights/biases are int16, we accumulate ≤32 pieces,
    // so max value is 32 * 32767 = ~1M which fits in int32 but after clamping to
    // [0,127] the live range is tiny. We store as int16 (range ±32767) which is
    // safe because individual weight rows are trained to keep sums in range, and
    // using int16 halves the AccEntry size (512→256 bytes), cutting memcpy cost
    // and doubling how many stack entries fit in L1 cache.
    int16_t w[MAX_HIDDEN];
    int16_t b[MAX_HIDDEN];
};

struct Accumulator {
    AccEntry stack[MAX_PLY + 4];
    int      top;

    void reset(const Board &board) {
        const NNUEWeights &W = g_weights;
        int H = W.hidden_size;
        AccEntry &e = stack[0];
        // Start from bias (ft_b is int16, AccEntry is int16 — direct assign)
        for (int i = 0; i < H; i++) { e.w[i] = W.ft_b[i]; e.b[i] = W.ft_b[i]; }
        // King squares
        int wk = board.king_sq[WHITE];
        int bk = sq_mirror(board.king_sq[BLACK]);
        // Accumulate all pieces
        for (int c = 0; c < 2; c++) {
            for (int pt = 0; pt < 5; pt++) {  // no KING
                Bitboard pieces = board.bb[c][pt];
                while (pieces) {
                    Square sq = lsb(pieces); pop_lsb(pieces);
                    int pidx = PIECE_IDX[pt][c];
                    int iw   = halfkp_w(wk, sq, pidx);
                    int ib   = halfkp_b(bk, sq, pidx);
                    for (int i = 0; i < H; i++) {
                        e.w[i] += W.ft_w[iw][i];
                        e.b[i] += W.ft_w[ib][i];
                    }
                }
            }
        }
        top = 0;
    }

    // Push with incremental update for non-king, non-castle moves
    void push_incremental(const Board &board, Move m) {
        const NNUEWeights &W = g_weights;
        int H = W.hidden_size;
        AccEntry &prev = stack[top];
        AccEntry &next = stack[top + 1];
        // Only copy the live H elements, not the full MAX_HIDDEN buffer.
        // H=256 → 1024 bytes instead of 2048 per node.
        memcpy(next.w, prev.w, H * sizeof(int16_t));
        memcpy(next.b, prev.b, H * sizeof(int16_t));
        top++;

        Square from = move_from(m), to = move_to(m);
        int wk = board.king_sq[WHITE];
        int bk = sq_mirror(board.king_sq[BLACK]);
        Color us  = board.stm;
        PieceType pt = board.piece_on[from];
        int pidx = PIECE_IDX[pt][us];
        PieceType promo = move_promo(m);
        int arriving_pidx = (move_flags(m) == MF_PROMO)
                            ? PIECE_IDX[promo][us] : pidx;

        // Remove from source
        int iw_from = halfkp_w(wk, from, pidx);
        int ib_from = halfkp_b(bk, from, pidx);
#if USE_NEON
        {
            const int16_t *rw = W.ft_w[iw_from], *rb = W.ft_w[ib_from];
            int16_t *nw = next.w, *nb = next.b;
            for (int i = 0; i < H; i += 8) {
                vst1q_s16(nw+i, vsubq_s16(vld1q_s16(nw+i), vld1q_s16(rw+i)));
                vst1q_s16(nb+i, vsubq_s16(vld1q_s16(nb+i), vld1q_s16(rb+i)));
            }
        }
#else
        for (int i = 0; i < H; i++) { next.w[i] -= W.ft_w[iw_from][i]; next.b[i] -= W.ft_w[ib_from][i]; }
#endif

        // Add to destination
        int iw_to = halfkp_w(wk, to, arriving_pidx);
        int ib_to = halfkp_b(bk, to, arriving_pidx);
#if USE_NEON
        {
            const int16_t *rw = W.ft_w[iw_to], *rb = W.ft_w[ib_to];
            int16_t *nw = next.w, *nb = next.b;
            for (int i = 0; i < H; i += 8) {
                vst1q_s16(nw+i, vaddq_s16(vld1q_s16(nw+i), vld1q_s16(rw+i)));
                vst1q_s16(nb+i, vaddq_s16(vld1q_s16(nb+i), vld1q_s16(rb+i)));
            }
        }
#else
        for (int i = 0; i < H; i++) { next.w[i] += W.ft_w[iw_to][i]; next.b[i] += W.ft_w[ib_to][i]; }
#endif

        // Capture (or EP)
        Color them = (Color)!us;
        Square cap_sq = to;
        PieceType cap_pt = board.piece_on[to];
        if (move_flags(m) == MF_EP) {
            cap_sq = (us == WHITE) ? to - 8 : to + 8;
            cap_pt = PAWN;
        }
        if (cap_pt != NO_PIECE) {
            int cpidx  = PIECE_IDX[cap_pt][them];
            int iw_cap = halfkp_w(wk, cap_sq, cpidx);
            int ib_cap = halfkp_b(bk, cap_sq, cpidx);
#if USE_NEON
            {
                const int16_t *rw = W.ft_w[iw_cap], *rb = W.ft_w[ib_cap];
                int16_t *nw = next.w, *nb = next.b;
                for (int i = 0; i < H; i += 8) {
                    vst1q_s16(nw+i, vsubq_s16(vld1q_s16(nw+i), vld1q_s16(rw+i)));
                    vst1q_s16(nb+i, vsubq_s16(vld1q_s16(nb+i), vld1q_s16(rb+i)));
                }
            }
#else
            for (int i = 0; i < H; i++) { next.w[i] -= W.ft_w[iw_cap][i]; next.b[i] -= W.ft_w[ib_cap][i]; }
#endif
        }
    }

    // Full recompute after king move or castling
    void push_full(const Board &board_after) {
        AccEntry &next = stack[top + 1];
        const NNUEWeights &W = g_weights;
        int H = W.hidden_size;
        int wk = board_after.king_sq[WHITE];
        int bk = sq_mirror(board_after.king_sq[BLACK]);
#if USE_NEON
        // Initialise from bias using NEON stores (H=256: 32 × vst1q_s16 per half)
        for (int i = 0; i < H; i += 8) {
            int16x8_t b = vld1q_s16(W.ft_b + i);
            vst1q_s16(next.w + i, b);
            vst1q_s16(next.b + i, b);
        }
#else
        for (int i = 0; i < H; i++) { next.w[i] = W.ft_b[i]; next.b[i] = W.ft_b[i]; }
#endif
        for (int c = 0; c < 2; c++) {
            for (int pt = 0; pt < 5; pt++) {
                Bitboard pieces = board_after.bb[c][pt];
                while (pieces) {
                    Square sq = lsb(pieces); pop_lsb(pieces);
                    int pidx = PIECE_IDX[pt][c];
                    int iw   = halfkp_w(wk, sq, pidx);
                    int ib   = halfkp_b(bk, sq, pidx);
#if USE_NEON
                    {
                        const int16_t *rw = W.ft_w[iw], *rb = W.ft_w[ib];
                        int16_t *nw = next.w, *nb = next.b;
                        for (int i = 0; i < H; i += 8) {
                            vst1q_s16(nw+i, vaddq_s16(vld1q_s16(nw+i), vld1q_s16(rw+i)));
                            vst1q_s16(nb+i, vaddq_s16(vld1q_s16(nb+i), vld1q_s16(rb+i)));
                        }
                    }
#else
                    for (int i = 0; i < H; i++) {
                        next.w[i] += W.ft_w[iw][i];
                        next.b[i] += W.ft_w[ib][i];
                    }
#endif
                }
            }
        }
        top++;
    }

    void null_push() {
        int H = g_weights.hidden_size;
        memcpy(stack[top + 1].w, stack[top].w, H * sizeof(int16_t));
        memcpy(stack[top + 1].b, stack[top].b, H * sizeof(int16_t));
        top++;
    }
    void pop() { top--; }
};

// ─────────────────────────── NNUE evaluation ───────────────────────────────

static int nnue_eval(const AccEntry &acc, bool stm_white, int bucket) {
    const NNUEWeights &W = g_weights;
    int H = W.hidden_size;

    // Clipped ReLU on FT output: clamp [0, FT_SCALE=127].
    // STM perspective first, opponent second (matches training convention).
    const int16_t *first  = stm_white ? acc.w : acc.b;
    const int16_t *second = stm_white ? acc.b : acc.w;

    // Pack both halves into int8 x[H*2], clamped to [0, FT_SCALE].
    // L1_SIZE=32 and L2_SIZE=32 are compile-time constants, so all inner
    // loops below are fixed-width and the compiler/NEON can fully unroll them.
    int8_t x[MAX_HIDDEN * 2];

#if USE_NEON
    // ── ARM NEON path ─────────────────────────────────────────────────────
    // Process 8 int16 lanes per iteration using:
    //   vld1q_s16   – load 8 × int16
    //   vmaxq_s16   – clamp below by 0    (ReLU)
    //   vminq_s16   – clamp above by 127  (clipped ReLU)
    //   vmovn_s16   – narrow int16 → int8 (two halves → one int8x16 register)
    {
        const int16x8_t zero   = vdupq_n_s16(0);
        const int16x8_t maxval = vdupq_n_s16(FT_SCALE);  // 127
        int i = 0;
        // Process two 8-element chunks of `first` and `second` together so we
        // can vmovn_s16 directly into an int8x16_t without intermediate storage.
        for (; i + 8 <= H; i += 8) {
            int16x8_t f = vld1q_s16(first  + i);
            int16x8_t s = vld1q_s16(second + i);
            f = vminq_s16(vmaxq_s16(f, zero), maxval);  // clamp f to [0, FT_SCALE]
            s = vminq_s16(vmaxq_s16(s, zero), maxval);  // clamp s to [0, FT_SCALE]
            // Store clamped int16 → int8 for each half
            // First half: x[i..i+7]
            vst1_s8((int8_t*)(x + i),   vmovn_s16(f));
            // Second half: x[H+i..H+i+7]
            vst1_s8((int8_t*)(x + H + i), vmovn_s16(s));
        }
        // Scalar tail (H is typically 256; loop always completes cleanly)
        for (; i < H; i++) {
            x[i]   = (int8_t)std::clamp((int)first[i],  0, FT_SCALE);
            x[H+i] = (int8_t)std::clamp((int)second[i], 0, FT_SCALE);
        }
    }

    // ── L1: scalar-input × int8-weight vector, accumulated into int32 ──────
    // l1_w layout is [in][out] (transposed at load time): W.l1_w[i] has L1_SIZE=32
    // contiguous int8 weights, one per output neuron.
    //
    // We use vmull_s8 (element-wise int8→int16 multiply) + vaddw_s16 (widen-add
    // into int32) to correctly compute l1_acc[o] += x[i] * W.l1_w[i][o].
    //
    // vdotq_s32 is intentionally NOT used here: with a broadcast scalar input,
    // vdotq_s32 computes x[i]*(w[4j]+w[4j+1]+w[4j+2]+w[4j+3]) per output lane —
    // a group SUM, not individual products — which produces wrong results.
    int32_t l1_acc[L1_SIZE];
    {
        // Initialise 8 × int32x4_t accumulators with biases (covering 32 outputs)
        int32x4_t acc0 = vld1q_s32(W.l1_b +  0);
        int32x4_t acc1 = vld1q_s32(W.l1_b +  4);
        int32x4_t acc2 = vld1q_s32(W.l1_b +  8);
        int32x4_t acc3 = vld1q_s32(W.l1_b + 12);
        int32x4_t acc4 = vld1q_s32(W.l1_b + 16);
        int32x4_t acc5 = vld1q_s32(W.l1_b + 20);
        int32x4_t acc6 = vld1q_s32(W.l1_b + 24);
        int32x4_t acc7 = vld1q_s32(W.l1_b + 28);

        for (int i = 0; i < H * 2; i++) {
            if (x[i] == 0) continue;  // exploit post-ReLU sparsity
            // Broadcast scalar x[i] to 8 lanes of int8x8_t.
            // x[i] is in [0,127] after ReLU clamp, so it's safe as signed int8.
            int8x8_t xi8 = vdup_n_s8(x[i]);
            const int8_t *row = W.l1_w[i];  // 32 int8 weights for this input

            // Load four 8-element groups of weights (32 total)
            int8x8_t w0 = vld1_s8(row +  0);  // outputs  0.. 7
            int8x8_t w1 = vld1_s8(row +  8);  // outputs  8..15
            int8x8_t w2 = vld1_s8(row + 16);  // outputs 16..23
            int8x8_t w3 = vld1_s8(row + 24);  // outputs 24..31

            // vmull_s8: element-wise int8 × int8 → int16x8_t (no summation)
            // This is the correct operation: each output lane i gets xi8[i]*w_k[i].
            int16x8_t p0 = vmull_s8(xi8, w0);
            int16x8_t p1 = vmull_s8(xi8, w1);
            int16x8_t p2 = vmull_s8(xi8, w2);
            int16x8_t p3 = vmull_s8(xi8, w3);

            // vaddw_s16: widen int16x4_t to int32x4_t and add to accumulator
            acc0 = vaddw_s16(acc0, vget_low_s16(p0));   // outputs  0.. 3
            acc1 = vaddw_s16(acc1, vget_high_s16(p0));  // outputs  4.. 7
            acc2 = vaddw_s16(acc2, vget_low_s16(p1));   // outputs  8..11
            acc3 = vaddw_s16(acc3, vget_high_s16(p1));  // outputs 12..15
            acc4 = vaddw_s16(acc4, vget_low_s16(p2));   // outputs 16..19
            acc5 = vaddw_s16(acc5, vget_high_s16(p2));  // outputs 20..23
            acc6 = vaddw_s16(acc6, vget_low_s16(p3));   // outputs 24..27
            acc7 = vaddw_s16(acc7, vget_high_s16(p3));  // outputs 28..31
        }
        vst1q_s32(l1_acc +  0, acc0);
        vst1q_s32(l1_acc +  4, acc1);
        vst1q_s32(l1_acc +  8, acc2);
        vst1q_s32(l1_acc + 12, acc3);
        vst1q_s32(l1_acc + 16, acc4);
        vst1q_s32(l1_acc + 20, acc5);
        vst1q_s32(l1_acc + 24, acc6);
        vst1q_s32(l1_acc + 28, acc7);
    }
#else
    // ── Scalar fallback (x86-64 / non-NEON) ───────────────────────────────
    for (int i = 0; i < H; i++) {
        x[i]   = (int8_t)std::clamp((int)first[i],  0, FT_SCALE);
        x[H+i] = (int8_t)std::clamp((int)second[i], 0, FT_SCALE);
    }

    int32_t l1_acc[L1_SIZE];
    for (int o = 0; o < L1_SIZE; o++) l1_acc[o] = W.l1_b[o];
    for (int i = 0; i < H * 2; i++) {
        if (x[i] == 0) continue;
        int xi = (int32_t)x[i];
        const int8_t *row = W.l1_w[i];
        for (int o = 0; o < L1_SIZE; o++) l1_acc[o] += xi * (int32_t)row[o];
    }
#endif // USE_NEON

    // Clamp L1 output to [0, L1_SCALE=64] after >>7 shift
    int8_t l1[L1_SIZE];
    for (int o = 0; o < L1_SIZE; o++) {
        int32_t v = l1_acc[o] >> 7;
        l1[o] = (int8_t)std::clamp(v, 0, L1_SCALE);
    }

#if USE_NEON
    // Widen-multiply-accumulate a 32-element int8 x int8 dot product using
    // base NEON (vmull_s8 + vpadal, no dependency on the optional ARMv8.4
    // dot-product extension, so it always compiles under -march=native).
    // Integer addition is exactly associative — reordering the summation
    // this way cannot change the result versus the scalar loop.
    auto dot32 = [](const int8_t *a, const int8_t *b) -> int32_t {
        int16x8_t p0 = vmull_s8(vld1_s8(a +  0), vld1_s8(b +  0));
        int16x8_t p1 = vmull_s8(vld1_s8(a +  8), vld1_s8(b +  8));
        int16x8_t p2 = vmull_s8(vld1_s8(a + 16), vld1_s8(b + 16));
        int16x8_t p3 = vmull_s8(vld1_s8(a + 24), vld1_s8(b + 24));
        int32x4_t sum = vpaddlq_s16(p0);
        sum = vpadalq_s16(sum, p1);
        sum = vpadalq_s16(sum, p2);
        sum = vpadalq_s16(sum, p3);
        return vaddvq_s32(sum);
    };

    // L2: 32×32 int8 dot-product
    int8_t l2[L2_SIZE];
    for (int o = 0; o < L2_SIZE; o++) {
        int32_t s = W.l2_b[o] + dot32(W.l2_w[o], l1);
        int32_t v = s / L1_SCALE;
        l2[o] = (int8_t)std::clamp(v, 0, L2_SCALE);
    }

    // Output layer: bucket-specific 32-element dot-product
    int32_t score = W.out_b[bucket] + dot32(W.out_w[bucket], l2);
#else
    // L2: 32×32 int8 dot-product, scalar
    int8_t l2[L2_SIZE];
    for (int o = 0; o < L2_SIZE; o++) {
        int32_t s = W.l2_b[o];
        for (int i = 0; i < L1_SIZE; i++) s += (int32_t)W.l2_w[o][i] * (int32_t)l1[i];
        int32_t v = s / L1_SCALE;
        l2[o] = (int8_t)std::clamp(v, 0, L2_SCALE);
    }

    // Output layer: bucket-specific 32-element dot-product
    int32_t score = W.out_b[bucket];
    for (int i = 0; i < L2_SIZE; i++) score += (int32_t)W.out_w[bucket][i] * (int32_t)l2[i];
#endif
    // Descale by L2_SCALE*OUT_SCALE: the weighted sum above carries both
    // L2_SCALE (from the incoming l2[] activations, themselves scaled by
    // L2_SCALE) and OUT_SCALE (from the output weights) — matching
    // serialize.py's out-bias scale (see its quantise_out()).
    return (int)std::lround(score / (L2_SCALE * OUT_SCALE));  // centipawns
}

// ─────────────────────────── .nnue loader ──────────────────────────────────

static bool load_nnue(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "info string Cannot open " << path << "\n"; return false; }
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read((char*)buf.data(), sz);
    if (!f) { std::cerr << "info string Read error\n"; return false; }

    size_t offset = 0;
    auto read32 = [&]() -> uint32_t {
        uint32_t v;
        memcpy(&v, buf.data()+offset, 4); offset += 4; return v;
    };

    uint32_t version  = read32();
    /* hash */ read32();
    uint32_t desc_len = read32();
    if (version != 0x00000001) {
        std::cerr << "info string Unknown .nnue version " << version << "\n"; return false;
    }
    offset += desc_len;  // skip description

    int H = g_weights.hidden_size;
    int B = g_weights.num_buckets;

    // ft.weight [HALFKP_SIZE, H] int16
    for (int i = 0; i < HALFKP_SIZE; i++)
        for (int j = 0; j < H; j++) {
            int16_t v; memcpy(&v, buf.data()+offset, 2); offset += 2;
            g_weights.ft_w[i][j] = v;
        }
    // ft.bias [H] int16
    for (int i = 0; i < H; i++) {
        int16_t v; memcpy(&v, buf.data()+offset, 2); offset += 2;
        g_weights.ft_b[i] = v;
    }
    // l1.weight [32, H*2] int8 — file is row-major [out][in], transpose to [in][out]
    {
        int8_t tmp[L1_SIZE][MAX_HIDDEN * 2];
        for (int o = 0; o < 32; o++)
            for (int i = 0; i < H*2; i++) { tmp[o][i] = (int8_t)buf[offset++]; }
        for (int i = 0; i < H*2; i++)
            for (int o = 0; o < 32; o++) g_weights.l1_w[i][o] = tmp[o][i];
    }
    // l1.bias [32] int32
    for (int i = 0; i < 32; i++) {
        int32_t v; memcpy(&v, buf.data()+offset, 4); offset += 4;
        g_weights.l1_b[i] = v;
    }
    // l2.weight [32,32] int8
    for (int o = 0; o < 32; o++)
        for (int i = 0; i < 32; i++) { g_weights.l2_w[o][i] = (int8_t)buf[offset++]; }
    // l2.bias [32] int32
    for (int i = 0; i < 32; i++) {
        int32_t v; memcpy(&v, buf.data()+offset, 4); offset += 4;
        g_weights.l2_b[i] = v;
    }
    // out.weight [B,32] int8
    for (int b = 0; b < B; b++)
        for (int i = 0; i < 32; i++) { g_weights.out_w[b][i] = (int8_t)buf[offset++]; }
    // out.bias [B] int32
    for (int b = 0; b < B; b++) {
        int32_t v; memcpy(&v, buf.data()+offset, 4); offset += 4;
        g_weights.out_b[b] = v;
    }

    g_weights.loaded = true;
    std::cerr << "info string Loaded " << path << " H=" << H << " B=" << B << "\n";
    return true;
}

// ─────────────────────────── Transposition table ───────────────────────────
// Uses 3-entry bucket clusters (Stockfish-style) to reduce eviction of deep
// entries on hash collision. Each cluster is exactly one cache line (64 bytes).
// Replacement policy: always replace the entry with the shallowest depth,
// unless an exact match on the key is found (then update in-place).
//
// Cache-line prefetching: tt_prefetch() is called at the START of each node
// before any work is done, so the memory request is in-flight while we compute
// move generation and static eval. On Apple M-series this saves ~80 cycles/node.

struct __attribute__((packed)) TTEntry {
    uint64_t key;    // 8 bytes
    Move     move;   // 4 bytes
    int32_t  score;  // 4 bytes
    int16_t  depth;  // 2 bytes
    uint8_t  flag;   // 1 byte
    uint8_t  age;    // 1 byte  → total: 20 bytes (packed, no implicit padding)
};

static constexpr int TT_CLUSTER_SIZE = 3;

// Each cluster is padded to exactly 64 bytes (one cache line).
struct alignas(64) TTCluster {
    TTEntry entries[TT_CLUSTER_SIZE];
    // 3 × 20 = 60 bytes of entries; 4 bytes of padding to reach 64.
    uint8_t  _pad[64 - TT_CLUSTER_SIZE * sizeof(TTEntry)];
};
static_assert(sizeof(TTCluster) == 64, "TTCluster must be exactly one cache line");

static constexpr uint8_t TT_EXACT = 0, TT_LOWER = 1, TT_UPPER = 2;

// Mate scores are root-relative ("mate in N from the root") while the TT is
// shared across all plies. Storing a root-relative mate score verbatim and
// later reading it back at a different ply silently corrupts mate distances
// (and can make a slower mate look faster than one actually available).
// Convert to a from-this-node distance on store, and back to root-relative
// on load — the standard technique (mirrors Stockfish's value_to_tt / from_tt).
static inline int score_to_tt(int score, int ply) {
    if (score >= MATE_SCORE - MAX_PLY)  return score + ply;
    if (score <= -(MATE_SCORE - MAX_PLY)) return score - ply;
    return score;
}
static inline int score_from_tt(int score, int ply) {
    if (score >= MATE_SCORE - MAX_PLY)  return score - ply;
    if (score <= -(MATE_SCORE - MAX_PLY)) return score + ply;
    return score;
}

static std::vector<TTCluster> g_tt;
static uint64_t               g_tt_mask = 0;  // cluster count - 1
static uint8_t                g_tt_age  = 0;  // incremented each search; used for eviction

static void tt_resize(int mb) {
    // Snap down to the largest power-of-2 cluster count that fits in mb MB.
    size_t clusters = ((size_t)mb * 1024 * 1024) / sizeof(TTCluster);
    size_t pot = 1;
    while (pot * 2 <= clusters) pot *= 2;
    g_tt.assign(pot, TTCluster{});
    g_tt_mask = (uint64_t)(pot - 1);
}

static void tt_clear() {
    for (auto &c : g_tt) memset(&c, 0, sizeof(TTCluster));
}

// Prefetch the cluster that would be accessed for `key`.
// Call this as early as possible in each node — the CPU hides latency while
// we do other work. On ARM64 PLDL1KEEP is emitted by __builtin_prefetch.
static inline void tt_prefetch(uint64_t key) {
    __builtin_prefetch(&g_tt[key & g_tt_mask], 0 /*read*/, 3 /*L1*/);
}

// Probe: return pointer to the matching entry, or nullptr on miss.
static TTEntry* tt_probe(uint64_t key) {
    TTCluster &cluster = g_tt[key & g_tt_mask];
    for (int i = 0; i < TT_CLUSTER_SIZE; i++) {
        if (cluster.entries[i].key == key)
            return &cluster.entries[i];
    }
    return nullptr;
}

// Store: find the best slot in the cluster to overwrite.
// Priority: (1) exact key match (update in-place), (2) age+depth eviction score.
// Eviction score: old entries (large age delta) and shallow entries are cheapest.
// Formula: (age_delta * 4 - depth), higher = better eviction candidate.
static void tt_store(uint64_t key, int depth, int score, uint8_t flag, Move mv) {
    TTCluster &cluster = g_tt[key & g_tt_mask];
    TTEntry   *replace = &cluster.entries[0];
    int        best_evict = INT_MIN;

    for (int i = 0; i < TT_CLUSTER_SIZE; i++) {
        TTEntry &e = cluster.entries[i];
        // Exact key match: always update (may improve depth/flag)
        if (e.key == key) {
            if (depth >= e.depth || flag == TT_EXACT)
                e = {key, mv, (int32_t)score, (int16_t)depth, flag, g_tt_age};
            return;
        }
        // Empty slot: ideal candidate
        if (e.key == 0) { replace = &e; break; }
        // Age-weighted eviction: prefer stale, shallow entries
        int age_delta = (int)(uint8_t)(g_tt_age - e.age);
        int evict_score = age_delta * 4 - (int)e.depth;
        if (evict_score > best_evict) { best_evict = evict_score; replace = &e; }
    }
    *replace = {key, mv, (int32_t)score, (int16_t)depth, flag, g_tt_age};
}

// ─────────────────────────── Killer / history ──────────────────────────────

static constexpr int HIST_MAX = 16384;

// ButterflyHistory [color][from*64+to]
// thread_local: each search thread keeps its own copy (Lazy SMP style — only
// the TT is truly shared between threads). Measured cost of thread_local vs.
// a global here: within noise (<2% NPS) on this platform — see the session
// that introduced multithreading for the measurement.
static thread_local int16_t g_main_history[2][64 * 64];

// ContinuationHistory [inCheck(0/1)][piece(0..5)][to][prev_piece(0..5)][prev_to]
// Covers 1-ply and 2-ply continuation contexts; indexed by (ss-1) and (ss-2).
static thread_local int16_t g_cont_history[2][6][64][6][64];

// CapturePieceToHistory [attacker_piece][to][captured_piece]
static thread_local int16_t g_capture_history[6][64][6];

// Countermove table [piece_type][to_square] → best reply to that move
static thread_local Move g_countermoves[6][64];

// ── Pawn correction history ─────────────────────────────────────────────────
// Tracks the running average gap between static eval and search result,
// indexed by pawn structure (see Board::pawn_hash), to correct systematic
// NNUE eval bias for a given pawn skeleton (e.g. a structure the net
// under/overrates). Standard technique, see
// https://www.chessprogramming.org/Static_Evaluation_Correction_History —
// this is a single-component (pawn-only) version of Stockfish's multi-
// component correction history.
static constexpr int CORRHIST_SIZE  = 16384;  // power of 2
// Stored value is in internal units, NOT centipawns directly — see
// corrhist_read(). CORRHIST_GRAIN converts internal units -> cp on read,
// so the table can converge with fine (sub-1-cp) resolution over many
// updates while the *applied* correction stays capped at a sane magnitude
// (CORRHIST_LIMIT / CORRHIST_GRAIN = 128cp at full saturation) instead of
// being added to static_eval 1:1, which was the bug in the first version
// of this feature: a saturated ±1024cp swing on the pruning-margin eval
// wrecked the search (confirmed via SPRT — candidate lost ~every game).
static constexpr int CORRHIST_LIMIT = 1024;   // internal-unit cap
static constexpr int CORRHIST_GRAIN = 8;      // internal units per cp
static thread_local int16_t g_pawn_corrhist[2][CORRHIST_SIZE];  // [color][pawn_hash & mask]

static inline int corrhist_index(uint64_t pawn_hash) {
    return (int)(pawn_hash & (CORRHIST_SIZE - 1));
}

// The cp-scale correction to actually apply to a static eval.
static inline int corrhist_read(Color c, uint64_t pawn_hash) {
    return g_pawn_corrhist[c][corrhist_index(pawn_hash)] / CORRHIST_GRAIN;
}

// Same self-bounding gravity update as hist_update, but with its own limit
// since correction values are internal-unit-scale, not move-ordering-score-scale.
static inline void corrhist_update(int16_t &h, int bonus) {
    bonus = std::clamp(bonus, -CORRHIST_LIMIT / 4, CORRHIST_LIMIT / 4);
    h += (int16_t)(bonus - (int)h * std::abs(bonus) / CORRHIST_LIMIT);
}

// Stack frame for search — carries continuation history pointers and static eval
struct SearchStack {
    int  staticEval = 0;
    bool inCheck    = false;
    Move currentMove = NO_MOVE;   // move that led to this position
    int  movingPiece = 6;         // piece type (0-5) of the move at this ply; 6 = none
    int  moveCount   = 0;
    int  statScore   = 0;
    // Pointer into g_cont_history for the move played at this ply:
    // [inCheck][piece][to][prev_piece][prev_to]
    int16_t (*contHist)[64] = nullptr; // points to g_cont_history[inCheck][piece][to]
    // Cumulative count of check extensions applied on the path from the
    // root to (and including) the move about to be made at this ply. Reset
    // implicitly to 0 at the root via the sentinel ss[-1]/ss[-2] zero-init.
    // See the check-extension gate below for why this exists.
    int  checkExtensions = 0;
};

static thread_local Move g_killers[MAX_PLY][2];

// SF-style gravity update: h += bonus - h*|bonus|/MAX
static inline void hist_update(int16_t &h, int bonus) {
    h += (int16_t)(bonus - (int)h * std::abs(bonus) / HIST_MAX);
}

static int stat_bonus(int depth) {
    return depth > 14 ? 66 : 6 * depth * depth + 231 * depth - 206;
}

// Called once per game (ucinewgame) — wipes all heuristic state.
static void history_clear() {
    memset(g_killers,         0, sizeof g_killers);
    memset(g_main_history,    0, sizeof g_main_history);
    memset(g_cont_history,    0, sizeof g_cont_history);
    memset(g_capture_history, 0, sizeof g_capture_history);
    memset(g_countermoves,    0, sizeof g_countermoves);
    memset(g_pawn_corrhist,   0, sizeof g_pawn_corrhist);
    g_tt_age = 0;
}

// g_tt_age increments each search so TT eviction prefers stale entries.
// Called once per `go`, by the orchestrating thread only, before any search
// thread is launched — g_tt_age is shared TT bookkeeping, not per-thread
// state, so it must be bumped exactly once regardless of Threads count.
static void search_init() {
    g_tt_age++;
}
// Called once per search thread (main and every Lazy SMP helper) at the
// start of its own iterate() — resets that thread's killers, which are
// ply-indexed and meaningless across searches. History tables accumulate
// across moves in the same game and are per-thread (thread_local); clearing
// them every move was throwing away all learned ordering information (major
// Elo loss) — same reasoning as before, now per-thread instead of global.
static void search_init_thread() {
    memset(g_killers, 0, sizeof g_killers);
}

static void killer_store(int ply, Move m) {
    if (g_killers[ply][0] != m) {
        g_killers[ply][1] = g_killers[ply][0];
        g_killers[ply][0] = m;
    }
}

// Update continuation histories for the move played at ss, conditioning on
// the moves at ss-1 and ss-2 (mirrors SF's update_continuation_histories).
static void update_continuation_histories(SearchStack *ss, int piece, Square to, int bonus) {
    for (int i : {1, 2}) {
        SearchStack *prev = ss - i;
        if (prev->contHist && prev->currentMove != NO_MOVE)
            hist_update(prev->contHist[piece][to], bonus);
    }
}

// Quiet move stats update: butterfly + continuation histories + killer
static void update_quiet_stats(const Board &board, SearchStack *ss, int ply, Move m, int bonus) {
    killer_store(ply, m);
    Color us = board.stm;
    int from = move_from(m), to = move_to(m);
    hist_update(g_main_history[us][from * 64 + to], bonus);
    int piece = (int)board.piece_on[from];
    update_continuation_histories(ss, piece, to, bonus);
}

// ─────────────────────────── SEE (Static Exchange Evaluation) ──────────────
// Returns true if the capture on `to` has SEE >= threshold centipawns.
// Uses the same "gain list" technique as Stockfish's see_ge().
static bool see_ge(const Board &board, Move m, int threshold) {
    if (move_flags(m) != MF_NORMAL) return threshold <= 0; // promotions/EP: assume ok

    Square from = move_from(m), to = move_to(m);
    int swap = PIECE_VALUE[(int)board.piece_on[to]] - threshold;
    if (swap < 0) return false;
    swap = PIECE_VALUE[(int)board.piece_on[from]] - swap;
    if (swap <= 0) return true;

    Bitboard occ    = board.all ^ sq_bb(from) ^ sq_bb(to);
    Bitboard diag   = (board.bb[WHITE][BISHOP] | board.bb[BLACK][BISHOP] |
                       board.bb[WHITE][QUEEN]  | board.bb[BLACK][QUEEN]) & occ;
    Bitboard orth   = (board.bb[WHITE][ROOK]   | board.bb[BLACK][ROOK]   |
                       board.bb[WHITE][QUEEN]  | board.bb[BLACK][QUEEN]) & occ;

    Bitboard attackers =
        (g_pawn_attacks[WHITE][to] & board.bb[BLACK][PAWN]) |
        (g_pawn_attacks[BLACK][to] & board.bb[WHITE][PAWN]) |
        (g_knight_attacks[to]      & (board.bb[WHITE][KNIGHT] | board.bb[BLACK][KNIGHT])) |
        (bishop_attacks(to, occ)   & diag) |
        (rook_attacks(to, occ)     & orth) |
        (g_king_attacks[to]        & (board.bb[WHITE][KING] | board.bb[BLACK][KING]));

    Color stm = (Color)!board.stm; // the side that just captured moves next
    bool res = true;
    while (true) {
        attackers &= occ;
        Bitboard my_atk = attackers & board.occupied[stm];
        if (!my_atk) break;
        res = !res;
        // Find least-valuable attacker
        int pt;
        for (pt = PAWN; pt <= KING; pt++) {
            if (my_atk & board.bb[stm][pt]) break;
        }
        if (pt > KING) break;
        swap = PIECE_VALUE[pt] - swap;
        if ((swap < res) ? res : !res) break; // pruning: outcome already decided
        // Remove the attacker we used
        Bitboard used = my_atk & board.bb[stm][pt];
        occ ^= used & -used; // remove LSB
        // Reveal hidden attackers along diagonals/files
        if (pt == PAWN || pt == BISHOP || pt == QUEEN)
            attackers |= bishop_attacks(to, occ) & diag;
        if (pt == ROOK || pt == QUEEN)
            attackers |= rook_attacks(to, occ) & orth;
        stm = (Color)!stm;
    }
    return res;
}

// ─────────────────────────── Move scoring ──────────────────────────────────

struct ScoredMove {
    Move move;
    int  score;
};

static void score_moves(const Board &board, ScoredMove *ms, int n, int ply,
                        Move tt_move, const SearchStack *ss) {
    Color us = board.stm;
    for (int i = 0; i < n; i++) {
        Move m = ms[i].move;
        if (m == tt_move) { ms[i].score = 10'000'000; continue; }

        Square    from     = move_from(m);
        Square    to       = move_to(m);
        PieceType attacker = board.piece_on[from];
        PieceType victim   = board.piece_on[to];
        bool      is_cap   = (victim != NO_PIECE || move_flags(m) == MF_EP);

        if (is_cap) {
            // MVV base + capture history
            int vv = (victim != NO_PIECE) ? PIECE_VALUE[victim] : 100;
            int cap_hist = g_capture_history[(int)attacker][to][(int)victim];
            ms[i].score = 1'000'000 + vv * 6 + cap_hist;
            continue;
        }
        if (move_flags(m) == MF_PROMO) {
            static constexpr int PROMO_SCORE[7] = {0,0,700000,700100,800000,900000,0};
            ms[i].score = PROMO_SCORE[(int)move_promo(m)];
            continue;
        }
        if (ply < MAX_PLY && m == g_killers[ply][0]) { ms[i].score = 800'000; continue; }
        if (ply < MAX_PLY && m == g_killers[ply][1]) { ms[i].score = 700'000; continue; }

        // Countermove heuristic: bonus for a move that historically refutes
        // the previous move (indexed by [prev_piece][prev_to]).
        if (ss && (ss-1)->currentMove != NO_MOVE && (ss-1)->currentMove != NULL_MOVE) {
            int prev_piece = (ss-1)->movingPiece;
            Square prev_to = move_to((ss-1)->currentMove);
            if (prev_piece < 6 && m == g_countermoves[prev_piece][prev_to]) {
                ms[i].score = 600'000; continue;
            }
        }

        // Butterfly history + 2 continuation history plies
        int score = g_main_history[us][from * 64 + to];
        if (ss) {
            int piece = (int)attacker;
            // ss-1 continuation (weight 2x as in SF)
            if ((ss-1)->contHist) score += 2 * (ss-1)->contHist[piece][to];
            // ss-2 continuation
            if ((ss-2)->contHist) score +=     (ss-2)->contHist[piece][to];
        }
        ms[i].score = score;
    }
}

// Swap the highest-scored remaining move into position i (selection sort step).
// Used instead of a full upfront sort — most nodes cut early so we never visit
// all moves, and a full std::sort on 30 moves is wasted work at those nodes.
static inline void pick_next(ScoredMove *ms, int start, int n) {
    int best = start;
    for (int i = start + 1; i < n; i++)
        if (ms[i].score > ms[best].score) best = i;
    if (best != start) std::swap(ms[start], ms[best]);
}

// ─────────────────────────── Timing ────────────────────────────────────────

using Clock = std::chrono::steady_clock;
// Shared across all search threads: written once by the orchestrating thread
// before any worker thread is launched, then read-only for the duration of
// the search (safe without atomics — pthread_create() establishes a
// happens-before edge for the launching thread's prior writes).
static Clock::time_point g_start_time;
static int64_t           g_time_limit_ms;  // -1 = no limit
static int64_t           g_node_limit;     // -1 = no limit
static int64_t           g_soft_limit_ms = -1;  // -1 = disabled (fixed movetime/depth/infinite searches)
static int               g_num_threads = 1;     // set via UCI "setoption name Threads"

// Per-thread: each search thread (Lazy SMP worker) tracks its own node count
// and time-poll bookkeeping. Node totals across threads are summed only when
// reporting (see the "go" handler) — never read cross-thread during search.
static thread_local int64_t g_nodes;
static thread_local int    g_root_history_size = 0;

// Shared stop flag: every thread must observe the same signal so they all
// stop together (set by UCI "stop", by the time budget, or by one thread
// hitting the node limit). Relaxed ordering is enough — this is a liveness
// flag, not a data-carrying variable, and every thread already re-checks its
// own g_nodes/elapsed_ms() before trusting anything gated on it.
static std::atomic<bool> g_stop{false};

static int64_t elapsed_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - g_start_time).count();
}
// Diagnostic only (see CLAUDE.md "CRITICAL — time-management overrun bug"):
// records the elapsed_ms() value at the last point out_of_time() actually
// sampled the clock, so the UCI loop can report the gap between that last
// poll and when search() finally returned. A large gap localizes the
// overrun to an unpolled stretch of code; a small gap means the overrun is
// elsewhere (allocation, output, or the budget itself). thread_local because
// each search thread polls independently; NNUE_TIME_DEBUG reporting reflects
// only the main (reporting) thread's poll gaps.
static thread_local int64_t g_last_poll_ms = 0;      // elapsed_ms() at the most recent poll
static thread_local int64_t g_prev_poll_ms = 0;      // elapsed_ms() at the poll before that
static thread_local int64_t g_max_poll_gap_ms = 0;   // largest observed gap between consecutive polls this search
static bool g_time_debug = false;  // set once in main() from NNUE_TIME_DEBUG env var

// ── Pruning firing-rate / depth-distribution instrumentation ────────────
// Purely additive diagnostic counters, gated by NNUE_PRUNE_DEBUG so there is
// zero behavior/perf change when unset. Read via the "prunestats" UCI debug
// command (see main loop). Used to re-measure which forward-pruning
// mechanism is saturated vs. leaving nodes on the table, post the
// output-layer quantization fix (the pre-fix firing-rate numbers in
// CLAUDE.md were measured against a badly-distorted eval and are stale).
// thread_local, like the history tables above: with Threads>1 each search
// thread accumulates its own counts, and "prunestats" (see the UCI loop)
// reports whichever thread services the debug command — normally used with
// Threads=1 for analysis, so this doesn't need cross-thread summation.
static bool g_prune_debug = false;  // set once in main() from NNUE_PRUNE_DEBUG env var
static thread_local int64_t g_nonleaf_nodes = 0;
static thread_local int64_t g_nodes_by_depth[64] = {0};
static thread_local int64_t g_rfp_checked = 0, g_rfp_hits = 0;
static thread_local int64_t g_nmp_checked = 0, g_nmp_hits = 0;
static thread_local int64_t g_probcut_checked = 0, g_probcut_hits = 0;
static thread_local int64_t g_futility_skips = 0;
static thread_local int64_t g_lmp_skips = 0;
static thread_local int64_t g_histprune_skips = 0;
// Depth-bucketed breakdown: pooling RFP's hit rate across depth 1-8 is
// meaningless because its margin (234*(depth-improving)) varies 8x across
// that range (0 at depth 1/improving, up to 1872 at depth 8/non-improving).
// These arrays answer "which depth band is actually restrictive" instead.
static thread_local int64_t g_rfp_checked_by_depth[64] = {0}, g_rfp_hits_by_depth[64] = {0};
static thread_local int64_t g_futility_skips_by_depth[64] = {0};
static thread_local int64_t g_lmp_skips_by_depth[64] = {0};
// Simulated RFP hit-rate at several candidate margin coefficients (in place
// of 234), computed alongside the real (unchanged) 234 check -- answers "how
// much node mass would newly be pruned by a tighter coefficient" without
// needing a full near-miss histogram or actually changing search behavior.
static constexpr int RFP_CANDIDATE_COEFS[] = {234, 200, 180, 165, 150, 120, 100, 80};
static constexpr int N_RFP_CANDIDATES = sizeof(RFP_CANDIDATE_COEFS) / sizeof(int);
static thread_local int64_t g_rfp_hits_at_coef[N_RFP_CANDIDATES] = {0};
// Same simulation, broken out by remaining depth -- the flat version above is
// dominated by depth-1 node mass (44% of all nonleaf nodes), where margin is
// already 0 whenever improving==true regardless of coefficient. This answers
// the actual decision question: does a tighter coefficient gain anything in
// the depth 2-4 band (~48% of node mass, margin 234-702cp, genuinely
// restrictive) specifically, not just in aggregate.
static thread_local int64_t g_rfp_hits_at_coef_by_depth[N_RFP_CANDIDATES][64] = {{0}};

static void prune_stats_reset() {
    g_nonleaf_nodes = 0;
    for (auto &c : g_nodes_by_depth) c = 0;
    g_rfp_checked = g_rfp_hits = 0;
    g_nmp_checked = g_nmp_hits = 0;
    g_probcut_checked = g_probcut_hits = 0;
    g_futility_skips = g_lmp_skips = g_histprune_skips = 0;
    for (auto &c : g_rfp_checked_by_depth) c = 0;
    for (auto &c : g_rfp_hits_by_depth) c = 0;
    for (auto &c : g_futility_skips_by_depth) c = 0;
    for (auto &c : g_lmp_skips_by_depth) c = 0;
    for (auto &c : g_rfp_hits_at_coef) c = 0;
    for (auto &row : g_rfp_hits_at_coef_by_depth) for (auto &c : row) c = 0;
}

static bool out_of_time() {
    if (g_stop.load(std::memory_order_relaxed)) return true;
    if (g_node_limit >= 0 && g_nodes >= g_node_limit) return true;
    if (g_time_limit_ms >= 0 && (g_nodes & 4095) == 0) {
        g_prev_poll_ms = g_last_poll_ms;
        g_last_poll_ms = elapsed_ms();
        int64_t gap = g_last_poll_ms - g_prev_poll_ms;
        if (gap > g_max_poll_gap_ms) g_max_poll_gap_ms = gap;
        return g_last_poll_ms >= g_time_limit_ms;
    }
    return false;
}

// ─────────────────────────── Search ────────────────────────────────────────

static int get_bucket(const Board &board) {
    int n = board.material_count();
    int b = n * g_weights.num_buckets / 32;
    return std::min(b, g_weights.num_buckets - 1);
}

static int quiescence(Board &board, Accumulator &acc, int alpha, int beta,
                      SearchStack *ss, int ply) {
    g_nodes++;

    // quiescence recurses on itself and, until this check was added, was
    // never gated on the clock at all — only alpha_beta polled
    // out_of_time(). Under strict time-control enforcement (fastchess, not
    // tournament.py's depth-limited setup) this produced real overruns
    // (observed: single moves running 100ms+ past their budget, and one
    // that didn't return within 45s) that alpha_beta's own poll can't catch
    // until qsearch unwinds. Mirrors alpha_beta's own check-and-return-alpha
    // convention.
    if (__builtin_expect(out_of_time(), 0)) return alpha;

    // Hard safety cap: SearchStack/Accumulator/Board::history are all sized
    // MAX_PLY (+ slack), and checked-evasion chains in qsearch have no depth
    // counter to stop them. Without this cap a pathological line of checks
    // can recurse past the fixed-size stacks — silent memory corruption
    // rather than a clean failure. This mirrors Stockfish's ss->ply >= MAX_PLY
    // leaf cutoff and should essentially never trigger in real games.
    if (ply >= MAX_PLY - 2)
        return nnue_eval(acc.stack[acc.top], board.stm == WHITE, get_bucket(board));

    bool in_check = board.in_check();

    // TT probe. Quiescence entries are stored at depth 0, which is exactly
    // the depth main search uses when it drops into qsearch, so a qsearch
    // entry is a valid cutoff source for any depth<=0 probe there too.
    uint64_t key = board.cur.hash;
    tt_prefetch(key);
    Move     tt_move = NO_MOVE;
    TTEntry *entry   = tt_probe(key);
    bool     tt_hit  = (entry != nullptr);
    int      tt_value = 0;
    if (tt_hit) {
        if (entry->move != NO_MOVE) {
            Move em = entry->move;
            Square ef = move_from(em), et = move_to(em);
            if (ef < 64 && et < 64 &&
                board.piece_on[ef] != NO_PIECE &&
                (board.occupied[board.stm] & sq_bb(ef)) &&
                !(board.occupied[board.stm] & sq_bb(et)))
                tt_move = em;
        }
        tt_value = score_from_tt(entry->score, ply);
        if (entry->flag == TT_EXACT) return tt_value;
        if (entry->flag == TT_LOWER && tt_value >= beta)  return tt_value;
        if (entry->flag == TT_UPPER && tt_value <= alpha) return tt_value;
    }

    int stm_w  = board.stm == WHITE;
    int bucket = get_bucket(board);
    int stand_pat = nnue_eval(acc.stack[acc.top], stm_w, bucket);
    {
        int corr = corrhist_read(board.stm, board.cur.pawn_hash);
        stand_pat = std::clamp(stand_pat + corr, -MATE_SCORE + MAX_PLY, MATE_SCORE - MAX_PLY);
    }
    // Refine the raw eval with a compatible TT bound, same idea as in the
    // main search: a stored bound that's tighter than the fresh eval is a
    // better basis for the stand-pat / pruning decisions below.
    if (tt_hit && ((entry->flag == TT_LOWER && tt_value > stand_pat) ||
                   (entry->flag == TT_UPPER && tt_value < stand_pat)))
        stand_pat = tt_value;

    int raw_alpha = alpha;

    // When in check we must try all evasions — no stand-pat cutoff.
    if (!in_check) {
        if (stand_pat >= beta) {
            tt_store(key, 0, score_to_tt(stand_pat, ply), TT_LOWER, NO_MOVE);
            return beta;
        }
        if (stand_pat + 900 + DELTA_MARGIN < alpha) return alpha;
        if (stand_pat > alpha) alpha = stand_pat;
    }

    // Generate captures only when not in check; all moves when in check.
    Move moves[MAX_MOVES];
    int n = board.gen_moves(moves, /*captures_only=*/!in_check);

    // If in check and no moves exist it's checkmate — ply-adjusted so a mate
    // found deep in the qsearch tree is correctly valued as slower than one
    // found near the root (fixes a flat -MATE_SCORE that previously ignored
    // how deep the mate actually was).
    if (in_check && n == 0) return -(MATE_SCORE - ply);

    ScoredMove ms[MAX_MOVES];
    for (int i = 0; i < n; i++) ms[i] = {moves[i], 0};
    score_moves(board, ms, n, ply, tt_move, ss);

    int  best_score = alpha;   // fail-hard baseline (already stand-pat clamped above)
    Move best_move  = NO_MOVE;

    for (int i = 0; i < n; i++) {
        pick_next(ms, i, n);
        Move m  = ms[i].move;

        // SEE pruning: skip moves with clearly negative exchange (only when not in check)
        if (!in_check && !see_ge(board, m, 0)) continue;

        bool king_moved = (board.piece_on[move_from(m)] == KING);
        bool is_castle  = (move_flags(m) == MF_CASTLE);
        if (is_castle || king_moved) {
            board.do_move(m);
            acc.push_full(board);
        } else {
            acc.push_incremental(board, m);
            board.do_move(m);
        }

        int score = -quiescence(board, acc, -beta, -alpha, ss + 1, ply + 1);
        board.undo_move(m);
        acc.pop();

        if (score > best_score) {
            best_score = score;
            best_move  = m;
        }
        if (score >= beta) {
            tt_store(key, 0, score_to_tt(score, ply), TT_LOWER, m);
            return beta;
        }
        if (score > alpha) alpha = score;
    }

    uint8_t flag = (best_score > raw_alpha) ? TT_EXACT : TT_UPPER;
    tt_store(key, 0, score_to_tt(best_score, ply), flag, best_move);
    return best_score;
}

// LMR reduction table [depth][moveCount], initialized at search startup
static int g_lmr[MAX_PLY][MAX_MOVES];
static void init_lmr() {
    for (int d = 1; d < MAX_PLY; d++)
        for (int m = 1; m < MAX_MOVES; m++)
            g_lmr[d][m] = (int)(std::log(d) * std::log(m) / 1.675 + 0.5);
}

// Forward declaration
static int alpha_beta(Board &board, Accumulator &acc,
                      int depth, int alpha, int beta,
                      int ply, bool allow_null, bool cut_node,
                      Move *pv, int &pv_len, SearchStack *ss,
                      Move excludedMove = NO_MOVE);

static int alpha_beta(Board &board, Accumulator &acc,
                      int depth, int alpha, int beta,
                      int ply, bool allow_null, bool cut_node,
                      Move *pv, int &pv_len, SearchStack *ss,
                      Move excludedMove) {
    g_nodes++;
    pv_len = 0;

    if (__builtin_expect(out_of_time(), 0)) return alpha;

    // Hard safety cap: extension chains (checks, singular) have no absolute
    // bound on how much they can push ply beyond depth, and SearchStack /
    // Accumulator / Board::history are fixed-size. Treat as a leaf rather
    // than risk overrunning those stacks.
    if (ply >= MAX_PLY - 2)
        return nnue_eval(acc.stack[acc.top], board.stm == WHITE, get_bucket(board));

    if (board.cur.halfmove_clock >= 100 || board.is_insufficient_material())
        return DRAW_SCORE;
    if (ply > 0 && board.is_repetition(g_root_history_size)) return DRAW_SCORE;

    bool is_pv    = (beta - alpha > 1);

    ss->inCheck    = false;
    ss->staticEval = EVAL_NONE;

    // Mate distance pruning: even a mate on the very next move can't beat a
    // shorter mate already guaranteed by an ancestor, and we can never do
    // better than being mated right now. Tightening the window here lets
    // many nodes near a forced mate cut immediately.
    alpha = std::max(alpha, -MATE_SCORE + ply);
    beta  = std::min(beta,   MATE_SCORE - ply - 1);
    if (alpha >= beta) return alpha;

    // Prefetch the TT cluster for this position's hash into L1 cache NOW,
    // before any other work, so the memory is ready by the time tt_probe reads it.
    uint64_t key = board.cur.hash;
    tt_prefetch(key);

    // TT probe — do before in_check() so TT hits skip the sq_attacked call
    Move tt_move   = NO_MOVE;
    TTEntry *entry = tt_probe(key);
    int tt_value   = 0;
    bool tt_hit    = (entry != nullptr);
    if (tt_hit) {
        if (entry->move != NO_MOVE) {
            Move em = entry->move;
            Square ef = move_from(em), et = move_to(em);
            if (ef < 64 && et < 64 &&
                board.piece_on[ef] != NO_PIECE &&
                (board.occupied[board.stm] & sq_bb(ef)) &&
                !(board.occupied[board.stm] & sq_bb(et)))
                tt_move = em;
        }
        tt_value = score_from_tt(entry->score, ply);
        // Cutoffs based on the stored bound are valid regardless of whether a
        // legal move was stored.  The old guard (tt_move != NO_MOVE) was causing
        // missed cutoffs at all-nodes and cut-nodes whose stored move failed
        // legality validation above.
        // Skipped entirely during a singular-extension verification search
        // (excludedMove set): that search must actually explore moves rather
        // than bounce straight back out via the very TT entry it's probing.
        if (excludedMove == NO_MOVE && entry->depth >= depth && !is_pv) {
            if (entry->flag == TT_EXACT) {
                if (tt_move != NO_MOVE) { pv[0] = tt_move; pv_len = 1; }
                return tt_value;
            }
            if (entry->flag == TT_LOWER && tt_value >= beta)  return tt_value;
            if (entry->flag == TT_UPPER && tt_value <= alpha) return tt_value;
        }
    }

    if (depth <= 0) return quiescence(board, acc, alpha, beta, ss, ply);

    // Internal Iterative Reduction: no TT move at high depth → reduce by 1
    if (depth >= 4 && tt_move == NO_MOVE) depth--;

    bool in_check = board.in_check();
    ss->inCheck   = in_check;

    int stm_w  = board.stm == WHITE;
    int bucket = get_bucket(board);
    // Skip the NNUE forward pass entirely when in check: every consumer of
    // static_eval below (RFP, NMP, ProbCut, futility) is already gated on
    // !in_check, so the only remaining use is `improving`, which treats
    // EVAL_NONE as "unknown" explicitly.
    int static_eval = in_check ? EVAL_NONE : nnue_eval(acc.stack[acc.top], stm_w, bucket);
    // Apply pawn correction history: a running per-pawn-structure adjustment
    // toward this position's typical (search - static eval) gap, correcting
    // systematic NNUE bias for that structure. See declaration for details.
    if (!in_check) {
        int corr = corrhist_read(board.stm, board.cur.pawn_hash);
        static_eval = std::clamp(static_eval + corr, -MATE_SCORE + MAX_PLY, MATE_SCORE - MAX_PLY);
    }
    ss->staticEval  = static_eval;

    // Refine the eval with a compatible TT bound (Stockfish-style): a stored
    // bound tighter than the fresh eval is a better basis for the pruning
    // decisions below. Skip near mate scores — those aren't comparable to a
    // plain positional eval.
    if (!in_check && tt_hit && std::abs(tt_value) < MATE_SCORE - MAX_PLY &&
        ((entry->flag == TT_LOWER && tt_value > static_eval) ||
         (entry->flag == TT_UPPER && tt_value < static_eval)))
        static_eval = tt_value;

    // "Improving": position better than 2 (or, failing that, 4) plies ago —
    // used to scale pruning margins. EVAL_NONE marks plies with no valid
    // static eval (in-check, or an early-returned node) so they're skipped
    // rather than misread as an eval of exactly 0.
    bool improving;
    if (in_check) {
        improving = false;
    } else if (ply >= 2 && (ss-2)->staticEval != EVAL_NONE) {
        improving = static_eval > (ss-2)->staticEval;
    } else if (ply >= 4 && (ss-4)->staticEval != EVAL_NONE) {
        improving = static_eval > (ss-4)->staticEval;
    } else {
        improving = true;
    }

    if (g_prune_debug) {
        g_nonleaf_nodes++;
        g_nodes_by_depth[std::clamp(depth, 0, 63)]++;
    }

    if (!in_check && excludedMove == NO_MOVE) {
        // Reverse futility pruning (SF Step 7)
        if (!is_pv && depth < 9) {
            if (g_prune_debug) {
                g_rfp_checked++;
                g_rfp_checked_by_depth[std::clamp(depth, 0, 63)]++;
                if (static_eval < 900000) {
                    int cd = std::clamp(depth, 0, 63);
                    for (int ci = 0; ci < N_RFP_CANDIDATES; ci++)
                        if (static_eval - RFP_CANDIDATE_COEFS[ci] * (depth - (int)improving) >= beta) {
                            g_rfp_hits_at_coef[ci]++;
                            g_rfp_hits_at_coef_by_depth[ci][cd]++;
                        }
                }
            }
            if (static_eval - 100 * (depth - (int)improving) >= beta &&
                static_eval < 900000) {
                if (g_prune_debug) {
                    g_rfp_hits++;
                    g_rfp_hits_by_depth[std::clamp(depth, 0, 63)]++;
                }
                return static_eval;
            }
        }

        // Null move pruning (SF Step 8)
        if (allow_null && depth >= NMP_MIN_DEPTH && static_eval >= beta &&
            !is_pv && board.bb[board.stm][QUEEN] | board.bb[board.stm][ROOK]) {
            if (g_prune_debug) g_nmp_checked++;
            int R = (1062 + 68 * depth) / 256 + std::min((static_eval - beta) / 190, 3);
            acc.null_push();
            board.do_move(NULL_MOVE);
            ss->currentMove = NULL_MOVE;
            Move child_pv[MAX_PLY]; int child_pv_len = 0;
            int null_score = -alpha_beta(board, acc, depth - 1 - R, -beta, -beta + 1,
                                         ply + 1, false, !cut_node,
                                         child_pv, child_pv_len, ss + 1);
            board.undo_move(NULL_MOVE);
            acc.pop();
            if (null_score >= beta) {
                if (g_prune_debug) g_nmp_hits++;
                return (null_score >= 900000) ? beta : null_score;
            }
        }

        // ProbCut (SF Step 9): if a good capture passes a raised-beta qsearch, prune.
        int prob_cut_beta = beta + 209 - 44 * (int)improving;
        if (!is_pv && depth > 4 && std::abs(beta) < 900000 &&
            !(tt_hit && entry->depth >= depth - 3 && tt_value < prob_cut_beta)) {
            if (g_prune_debug) g_probcut_checked++;
            // Try a few captures with SEE >= prob_cut_beta - static_eval
            Move caps[MAX_MOVES];
            int ncaps = board.gen_moves(caps, /*captures_only=*/true);
            ScoredMove cms[MAX_MOVES];
            for (int i = 0; i < ncaps; i++) cms[i] = {caps[i], 0};
            score_moves(board, cms, ncaps, ply, tt_move, ss);
            int prob_cut_count = 0;
            for (int i = 0; i < ncaps && prob_cut_count < 3; i++) {
                pick_next(cms, i, ncaps);
                Move m = cms[i].move;
                if (!see_ge(board, m, prob_cut_beta - static_eval)) continue;
                prob_cut_count++;
                bool king_moved = (board.piece_on[move_from(m)] == KING);
                bool is_castle  = (move_flags(m) == MF_CASTLE);
                int  pc_piece   = (int)board.piece_on[move_from(m)];  // before do_move
                if (is_castle || king_moved) {
                    board.do_move(m);
                    acc.push_full(board);
                } else {
                    acc.push_incremental(board, m);
                    board.do_move(m);
                }
                ss->currentMove  = m;
                ss->movingPiece  = pc_piece;
                Move cp[MAX_PLY]; int cpl = 0;
                int pc_val = -quiescence(board, acc, -prob_cut_beta, -prob_cut_beta + 1, ss + 1, ply + 1);
                // Same ungated-re-search hazard as the LMR re-search below:
                // don't chase an aborted qsearch's score with a fresh,
                // node-count-unaligned alpha_beta call.
                if (pc_val >= prob_cut_beta && !out_of_time())
                    pc_val = -alpha_beta(board, acc, depth - 4, -prob_cut_beta, -prob_cut_beta + 1,
                                         ply + 1, false, !cut_node, cp, cpl, ss + 1);
                board.undo_move(m);
                acc.pop();
                if (pc_val >= prob_cut_beta) {
                    if (g_prune_debug) g_probcut_hits++;
                    tt_store(key, depth - 3, score_to_tt(pc_val, ply), TT_LOWER, m);
                    return pc_val;
                }
            }
        }
    }

    bool futil = (!in_check && !is_pv && depth <= 4 &&
                  static_eval + FUTILITY_MARGIN[std::min(depth,4)] <= alpha);

    // ── Singular Extension (SF Step 11) ───────────────────────────────────
    // If we have a TT move at sufficient depth, check whether it is "singular"
    // (vastly better than all alternatives). If a reduced-depth search that
    // *excludes* the TT move fails to reach a window centred below tt_value,
    // no alternative comes close — the TT move is singular and deserves a
    // +1 extension. If that search fails HIGH instead (multi-cut), several
    // alternatives already beat beta, so the whole subtree can be pruned.
    //
    // The verification search below must run at the SAME ply/position (no
    // move has been made) with the TT move excluded from consideration —
    // passing excludedMove down disables the TT cutoff and TT store for that
    // call so it actually explores alternative moves instead of bouncing
    // straight back out through the very entry it's trying to verify.
    //
    // Conditions (following Stockfish):
    //   • We have a TT move with depth ≥ depth − 3
    //   • depth ≥ 8
    //   • Not a PV node (too expensive to search twice there)
    //   • Not in check (check extensions handled separately)
    //   • Not already inside another singular verification search
    //   • The TT bound is TT_LOWER (at least a lower bound, i.e. it raised alpha)
    int singular_ext = 0;
    if (!in_check && !is_pv && depth >= 8 && excludedMove == NO_MOVE &&
        tt_hit && tt_move != NO_MOVE &&
        entry->flag == TT_LOWER &&
        entry->depth >= depth - 3 &&
        std::abs(tt_value) < MATE_SCORE - MAX_PLY) {

        // Singular beta = tt_value - depth * 2  (SF formula)
        int s_beta  = tt_value - depth * 2;
        int s_depth = (depth - 1) / 2;

        Move s_pv[MAX_PLY]; int s_pv_len = 0;
        int s_val = alpha_beta(board, acc, s_depth, s_beta - 1, s_beta,
                               ply, /*allow_null=*/false, cut_node,
                               s_pv, s_pv_len, ss, /*excludedMove=*/tt_move);

        if (s_val < s_beta) {
            // TT move is singular — extend it.
            // Double extension (SF-style): if the TT move is far better than all
            // alternatives (s_val << s_beta), it's so uniquely good we extend by 2.
            singular_ext = (s_val < s_beta - 17) ? 2 : 1;
        } else if (s_beta >= beta) {
            // Multi-cut heuristic: many alternatives already beat beta → prune.
            return s_beta;
        }
    }

    // Generate moves
    Move moves[MAX_MOVES];
    int n = board.gen_moves(moves);
    if (n == 0) return in_check ? -(MATE_SCORE - ply) : DRAW_SCORE;

    ScoredMove ms[MAX_MOVES];
    for (int i = 0; i < n; i++) ms[i] = {moves[i], 0};
    score_moves(board, ms, n, ply, tt_move, ss);

    int  best_score      = -INF;
    Move best_move       = NO_MOVE;
    // Distinct from best_move: best_move is set on any improvement over the
    // initial -INF (so it's set by the very first move tried, essentially
    // always non-NO_MOVE — correct for TT move-ordering hints even at a
    // fail-low node). The correction-history gate below needs specifically
    // "did any move raise alpha", which only this tracks.
    Move alpha_raising_move = NO_MOVE;
    int  orig_alpha      = alpha;
    int  moves_done      = 0;
    int  quiet_count     = 0;   // counts quiet moves tried (for LMP)
    bool search_aborted  = false;
    Color moving_side    = board.stm;

    // Track quiets and captures tried, for history penalisation on beta-cutoff
    Move quiets_tried[64];  int n_quiets  = 0;
    Move caps_tried[32];    int n_caps    = 0;

    for (int i = 0; i < n; i++) {
        pick_next(ms, i, n);
        Move m      = ms[i].move;
        if (m == excludedMove) continue;
        Square from = move_from(m);
        Square to   = move_to(m);
        PieceType pt    = board.piece_on[from];
        PieceType victim= board.piece_on[to];
        bool is_capture = (victim != NO_PIECE || move_flags(m) == MF_EP);
        bool is_prom    = (move_flags(m) == MF_PROMO);
        bool is_castle  = (move_flags(m) == MF_CASTLE);
        bool king_moved = (pt == KING);

        if (futil && moves_done > 0 && !is_capture && !is_prom) {
            if (g_prune_debug) {
                g_futility_skips++;
                g_futility_skips_by_depth[std::clamp(depth, 0, 63)]++;
            }
            continue;
        }

        // ── Late Move Pruning (LMP) ──────────────────────────────────────────
        // At low depths, once we've tried enough quiets the remaining ones are
        // unlikely to raise alpha — skip them entirely.  Saving ~30% of nodes
        // at shallow depths.  Only applies when not in check, not a PV node,
        // and not near a mate score so we don't prune potential mate-savers.
        if (!in_check && !is_pv && !is_capture && !is_prom &&
            moves_done > 0 && best_score > -(MATE_SCORE - MAX_PLY) &&
            depth <= 8) {
            int lmp_limit = LMP_MOVES[depth][improving];
            if (quiet_count >= lmp_limit) {
                if (g_prune_debug) {
                    g_lmp_skips++;
                    g_lmp_skips_by_depth[std::clamp(depth, 0, 63)]++;
                }
                continue;
            }
        }

        // ── History-based quiet pruning ──────────────────────────────────────
        // At low depths, skip quiets whose combined history score is strongly
        // negative — they are persistently bad moves the engine has learned to
        // avoid.  Saves meaningful nodes without requiring SEE.
        if (!in_check && !is_pv && !is_capture && !is_prom &&
            moves_done > 0 && depth <= 6 &&
            best_score > -(MATE_SCORE - MAX_PLY)) {
            int hist = g_main_history[moving_side][from * 64 + to];
            if ((ss-1)->contHist) hist += (ss-1)->contHist[(int)pt][to];
            if ((ss-2)->contHist) hist += (ss-2)->contHist[(int)pt][to];
            // Threshold scales with depth: deeper searches are more lenient.
            if (hist < -1024 * depth) {
                if (g_prune_debug) g_histprune_skips++;
                continue;
            }
        }

        // SEE-based pruning for late moves (SF Step 12)
        if (!in_check && moves_done > 0 && best_score > -(MATE_SCORE - MAX_PLY)) {
            if (is_capture || is_prom) {
                // Losing captures at low depth: SEE < 0 → skip
                if (depth < 3 && !see_ge(board, m, 0)) continue;
            } else {
                // Quiet: SEE negative means likely bad
                int see_thresh = -(30 - std::min(depth, 18)) * depth * depth;
                if (!see_ge(board, m, see_thresh)) continue;
            }
        }

        int ext = 0;
        // Apply singular extension to the TT move (which is sorted first).
        if (m == tt_move) ext = singular_ext;

        // SEE for the check-extension gate below, must be computed on the
        // pre-move board (see_ge simulates the exchange sequence starting
        // from `m` on the current position) — can't be deferred past do_move.
        bool check_ext_see_ok = see_ge(board, m, 0);

        // Push accumulator BEFORE do_move
        if (is_castle || king_moved) {
            board.do_move(m);
            acc.push_full(board);
        } else {
            acc.push_incremental(board, m);
            board.do_move(m);
        }

        bool gives_check = board.in_check();
        // Check extension, gated (was previously unconditional +1 for every
        // checking move, no SEE gate, no per-line budget — confirmed via
        // node-count testing to cost 23-57% extra nodes at fixed depth in
        // 3 of 4 positions tested, see CLAUDE.md). Only extend a check that
        // isn't a losing sacrifice (SEE >= 0), and only up to a fixed budget
        // of check extensions stacked along this search line, to bound the
        // worst-case cost of a long forced-check sequence.
        bool applied_check_ext = false;
        int inherited_check_ext = (ss - 1)->checkExtensions;
        if (gives_check && ext == 0 &&
            inherited_check_ext < CHECK_EXT_BUDGET && check_ext_see_ok) {
            ext = 1;
            applied_check_ext = true;
        }
        ss->checkExtensions = inherited_check_ext + (applied_check_ext ? 1 : 0);

        // Set up continuation history pointer for this node
        ss->currentMove  = m;
        ss->movingPiece  = (int)pt;  // used by countermove heuristic at child nodes
        int piece = (int)pt;
        ss->contHist = (pt != NO_PIECE)
            ? g_cont_history[(int)gives_check][piece][to]
            : nullptr;

        Move child_pv[MAX_PLY]; int child_pv_len = 0;
        int score;

        if (moves_done == 0) {
            score = -alpha_beta(board, acc, depth - 1 + ext, -beta, -alpha,
                                ply + 1, true, false, child_pv, child_pv_len, ss + 1);
        } else {
            // LMR (SF Step 15)
            int r = 0;
            if (depth >= LMR_MIN_DEPTH && moves_done >= LMR_FULL_MOVES &&
                !is_capture && !gives_check && !is_prom && !in_check) {
                r = g_lmr[std::min(depth, MAX_PLY-1)][std::min(moves_done, MAX_MOVES-1)];
                // Stat score adjustment (SF: r -= statScore/14382)
                // Use butterfly history + continuation history from *parent* plies
                // (ss-1 and ss-2), which are the move contexts for this position.
                // ss->contHist is not yet valid here (it belongs to the child move).
                int stat_score = g_main_history[moving_side][from * 64 + to];
                if ((ss-1)->contHist) stat_score += 2 * (ss-1)->contHist[piece][to];
                if ((ss-2)->contHist) stat_score +=     (ss-2)->contHist[piece][to];
                r -= stat_score / 14382;
                // Increase for cut nodes
                if (cut_node) r += 2;
                // PV node gets less reduction
                if (is_pv)   r -= 1;
                r = std::clamp(r, 0, depth - 1);
            }
            score = -alpha_beta(board, acc, depth - 1 - r + ext, -alpha - 1, -alpha,
                                ply + 1, true, true, child_pv, child_pv_len, ss + 1);
            // Re-search at full depth if LMR failed high.
            // Gated on !out_of_time(): a "fail high" here can be a spurious
            // artifact of the reduced-depth search itself having been aborted
            // deep inside by a timeout (it returns its local alpha bound,
            // which after negation across plies can look like score > alpha
            // with no real information behind it). Without this guard, that
            // triggers a brand-new *full-depth* search that starts at a fresh
            // node count not aligned to the polling boundary, so it runs
            // unchecked for up to ~4096 nodes before the clock is read again
            // — measured as the dominant cause of the CLAUDE.md-documented
            // time-overrun bug (SPRT losses on time despite generous margin).
            if (score > alpha && r > 0 && !out_of_time()) {
                child_pv_len = 0;
                score = -alpha_beta(board, acc, depth - 1 + ext, -alpha - 1, -alpha,
                                    ply + 1, true, !cut_node, child_pv, child_pv_len, ss + 1);
                // Update continuation history based on LMR result
                int bonus = (score > alpha) ? stat_bonus(depth - 1) : -stat_bonus(depth - 1);
                update_continuation_histories(ss, piece, to, bonus);
            }
            // Same rationale as above: don't launch an ungated full-window
            // re-search on a score that may only reflect an aborted subtree.
            if (is_pv && score > alpha && score < beta && !out_of_time()) {
                child_pv_len = 0;
                score = -alpha_beta(board, acc, depth - 1 + ext, -beta, -alpha,
                                    ply + 1, true, false, child_pv, child_pv_len, ss + 1);
            }
        }

        board.undo_move(m);
        acc.pop();
        moves_done++;

        if (__builtin_expect(out_of_time(), 0)) { search_aborted = true; break; }

        if (score > best_score) {
            best_score = score;
            best_move  = m;
            if (score > alpha) {
                alpha_raising_move = m;
                alpha  = score;
                pv[0]  = m;
                if (child_pv_len > 0 && child_pv_len < MAX_PLY - 1)
                    memcpy(pv + 1, child_pv, sizeof(Move) * child_pv_len);
                pv_len = child_pv_len + 1;
            }
        }

        if (__builtin_expect(score >= beta, 0)) {
            int bonus = stat_bonus(depth);
            if (!is_capture && !is_prom) {
                // Good quiet: reward butterfly + continuation, penalise all tried quiets
                update_quiet_stats(board, ss, ply, m, bonus);
                // Countermove: record this quiet as the reply to the previous move
                if ((ss-1)->currentMove != NO_MOVE && (ss-1)->currentMove != NULL_MOVE) {
                    int prev_piece = (ss-1)->movingPiece;
                    Square prev_to = move_to((ss-1)->currentMove);
                    if (prev_piece < 6) g_countermoves[prev_piece][prev_to] = m;
                }
                for (int j = 0; j < n_quiets; j++) {
                    Move qm = quiets_tried[j];
                    int qfrom = move_from(qm), qto = move_to(qm);
                    hist_update(g_main_history[moving_side][qfrom * 64 + qto], -bonus);
                    int qpiece = (int)board.piece_on[qfrom];
                    update_continuation_histories(ss, qpiece, qto, -bonus);
                }
            } else {
                // Good capture: reward capture history
                PieceType cap = (victim != NO_PIECE) ? victim : PAWN;
                hist_update(g_capture_history[(int)pt][to][(int)cap], bonus);
                for (int j = 0; j < n_caps; j++) {
                    Move cm = caps_tried[j];
                    int cto = move_to(cm);
                    PieceType cpt  = board.piece_on[move_from(cm)];
                    PieceType cvic = board.piece_on[cto];
                    if (cvic != NO_PIECE)
                        hist_update(g_capture_history[(int)cpt][cto][(int)cvic], -bonus);
                }
            }
            break;
        }

        // Record tried moves for penalisation on cutoff
        if (!is_capture && !is_prom) {
            if (n_quiets < 64) quiets_tried[n_quiets++] = m;
            quiet_count++;
        }
        if (is_capture && n_caps < 32) caps_tried[n_caps++] = m;
    }

    // An aborted search only explored a prefix of the move list, so
    // best_score is not a valid bound for this depth — storing it would
    // corrupt the TT for the rest of the game (the TT persists across moves
    // until ucinewgame). Likewise, a singular-extension verification search
    // (excludedMove set) must not overwrite the entry it's verifying against.
    if (!search_aborted && excludedMove == NO_MOVE) {
        uint8_t flag = (orig_alpha < best_score && best_score < beta) ? TT_EXACT :
                       (best_score >= beta)                            ? TT_LOWER : TT_UPPER;
        tt_store(key, depth, score_to_tt(best_score, ply), flag, best_move);
    }

    // Correction history update — teach this pawn structure's slot to close
    // the gap between static eval and what the search actually found.
    // Gated the same way Stockfish gates it: not in check or aborted or a
    // singular-verification call, no mate score (those reflect tactics, not
    // positional eval bias), the alpha-raising move (if any) isn't a capture
    // (captures change the pawn structure's context, not informative about
    // *this* structure), and the error direction matches whether some move
    // improved on alpha at all — guards against reinforcing noise from an
    // all-pruned fail-low. Deliberately uses alpha_raising_move, not
    // best_move: best_move is set by the first move tried regardless of
    // whether it raised alpha (correct for TT move-ordering hints, wrong
    // here — using it made this gate always-true and let the table only
    // ever ratchet upward, confirmed via SPRT: candidate lost ~90% of games).
    bool alpha_raiser_is_capture = alpha_raising_move != NO_MOVE &&
        (board.piece_on[move_to(alpha_raising_move)] != NO_PIECE ||
         move_flags(alpha_raising_move) == MF_EP);
    if (!in_check && !search_aborted && excludedMove == NO_MOVE &&
        std::abs(best_score) < MATE_SCORE - MAX_PLY && !alpha_raiser_is_capture &&
        (best_score > ss->staticEval) == (alpha_raising_move != NO_MOVE)) {
        // bonus is computed in cp, then converted to internal units (see
        // CORRHIST_GRAIN) before feeding the gravity update.
        int bonus = (best_score - ss->staticEval) * depth / 8 * CORRHIST_GRAIN;
        corrhist_update(g_pawn_corrhist[board.stm][corrhist_index(board.cur.pawn_hash)], bonus);
    }

    return best_score;
}

// ─────────────────────────── Engine ────────────────────────────────────────

struct Engine {
    Board       board;
    std::string nn_file;

    Engine() { board.set_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"); }

    void set_nnue(const std::string &path) {
        nn_file = path;
        load_nnue(path);
    }

    // Runs one thread's iterative-deepening loop on its own private copy of
    // the position (`board` taken by value — Lazy SMP: every search thread,
    // main and helpers alike, gets an independent Board+Accumulator; the
    // only state genuinely shared between threads is the TT, plus the
    // read-only-during-search time/stop globals set by search() before any
    // thread is launched). `report` gates the "info depth ..." stdout lines
    // and the out-params — only the main thread's result is used as the
    // engine's actual move; helper threads search purely to warm the shared
    // TT and are otherwise silent, per the standard Lazy SMP design (no
    // cross-thread best-move voting in this version).
    static void iterate(Board board, int depth, int64_t soft_limit_ms, bool report,
                         Move &out_best_move, int &out_best_score) {
        search_init_thread();

        Accumulator acc;
        acc.reset(board);

        g_nodes            = 0;
        g_last_poll_ms      = 0;
        g_prev_poll_ms      = 0;
        g_max_poll_gap_ms   = 0;

        Move best_move  = NO_MOVE;
        int  best_score = 0;
        int  prev_score = 0;

        g_root_history_size = board.history_top;

        {
            int bkt = get_bucket(board);
            prev_score = nnue_eval(acc.stack[0], board.stm == WHITE, bkt);
        }

        // SearchStack — oversized so ss-2 is always valid even at ply 0
        SearchStack ss_storage[MAX_PLY + 4] = {};
        SearchStack *ss = ss_storage + 2; // ss[-2] and ss[-1] are sentinel zeroes

        Move pv[MAX_PLY];
        int  pv_len = 0;

        // Best-move instability tracking: each time the best move changes
        // between consecutive completed iterations (from a depth where that
        // means something, not the noisy shallow depths), stretch the soft
        // limit a bit — an unstable position is exactly when cutting the
        // search short is most likely to return the wrong move. Always
        // capped by the hard limit via out_of_time().
        int64_t effective_soft = soft_limit_ms;

        for (int d = 1; d <= depth; d++) {
            if (out_of_time()) break;
            // Soft-limit early exit: once the normal allocation for this
            // move is used up and the line hasn't been unstable, don't
            // start another iteration. Only active for wtime/btime-derived
            // budgets (soft_limit_ms >= 0) — fixed `go movetime`/`go
            // depth`/`go infinite` searches use the full hard limit.
            if (soft_limit_ms >= 0 && d > 1 && elapsed_ms() >= effective_soft) break;

            Move iter_pv[MAX_PLY]; int iter_pv_len = 0;
            int score;

            if (d >= 4) {
                int window = ASP_WINDOW;
                int alpha  = prev_score - window;
                int beta   = prev_score + window;
                for (int t = 0; t < ASP_MAX_TRIES; t++) {
                    iter_pv_len = 0;
                    score = alpha_beta(board, acc, d, alpha, beta, 0, true, false,
                                       iter_pv, iter_pv_len, ss);
                    if (out_of_time()) break;
                    if (score <= alpha) {
                        // Fail-low: widen only the lower bound; keep beta tight.
                        alpha = std::max(alpha - window, -INF);
                        window += window / 4 + 5;
                    } else if (score >= beta) {
                        // Fail-high: widen only the upper bound; keep alpha tight.
                        beta = std::min(beta + window, INF);
                        window += window / 4 + 5;
                    } else {
                        break;  // score inside window — done
                    }
                }
                if (out_of_time() && iter_pv_len == 0) break;
                // Fall back to full-width search if the window never converged
                // (i.e. score still outside initial ±ASP_WINDOW range after max retries).
                if (!out_of_time() &&
                    (score <= prev_score - ASP_WINDOW * ASP_MAX_TRIES ||
                     score >= prev_score + ASP_WINDOW * ASP_MAX_TRIES)) {
                    iter_pv_len = 0;
                    score = alpha_beta(board, acc, d, -INF, INF, 0, true, false,
                                       iter_pv, iter_pv_len, ss);
                }
            } else {
                score = alpha_beta(board, acc, d, -INF, INF, 0, true, false,
                                   iter_pv, iter_pv_len, ss);
            }

            if (out_of_time() && d > 1) break;

            if (iter_pv_len > 0) {
                // Instability: the best move changed from the previous
                // completed iteration. Ignore the shallow depths, where this
                // is just noise rather than a signal worth reacting to.
                if (soft_limit_ms >= 0 && d >= 5 &&
                    best_move != NO_MOVE && iter_pv[0] != best_move) {
                    effective_soft = std::min(g_time_limit_ms,
                                               (int64_t)(effective_soft * 1.3));
                }
                best_move  = iter_pv[0];
                best_score = score;
                prev_score = score;
                memcpy(pv, iter_pv, sizeof(Move) * iter_pv_len);
                pv_len = iter_pv_len;
            }

            if (report) {
                int64_t t_ms = elapsed_ms();
                int64_t nps  = (t_ms > 0) ? g_nodes * 1000 / t_ms : g_nodes;
                std::string pv_str;
                for (int i = 0; i < std::min(pv_len, 6); i++) {
                    if (i) pv_str += ' ';
                    pv_str += board.move_uci(pv[i]);
                }
                std::cout << "info depth " << d
                          << " score cp " << best_score
                          << " nodes "    << g_nodes
                          << " nps "      << nps
                          << " time "     << t_ms
                          << " pv "       << pv_str
                          << "\n" << std::flush;
            }
        }

        if (best_move == NO_MOVE) {
            Move moves[MAX_MOVES];
            int n = board.gen_moves(moves);
            if (n) best_move = moves[0];
        }
        out_best_move  = best_move;
        out_best_score = best_score;
    }

    // soft_limit_ms: the "normal" time allocation for this move — once used
    // up, iterative deepening won't start another depth unless the best
    // move has been unstable (see below). -1 disables soft-limit early exit
    // (fixed `go movetime`/`go depth`/`go infinite` — those should use the
    // full hard limit, not stop early). hard_limit_ms is the absolute cap
    // (what out_of_time() enforces, same role the old single `movetime_ms`
    // parameter played) — for -1 soft_limit_ms callers, hard_limit_ms is
    // that same fixed budget.
    //
    // Lazy SMP: spawns (Threads-1) helper threads, each running its own
    // independent iterate() call sharing only the TT and the time/stop
    // globals below (set here, before any thread starts, then read-only for
    // the duration of the search). The calling thread runs its own iterate()
    // as the "main" (reporting) thread; its result is the one returned. See
    // CLAUDE.md's multithreading section for the design rationale.
    std::pair<Move,int> search(int depth, int64_t soft_limit_ms, int64_t hard_limit_ms) {
        if (!g_weights.loaded) {
            Move moves[MAX_MOVES];
            int n = board.gen_moves(moves);
            return {n > 0 ? moves[0] : NO_MOVE, 0};
        }

        g_stop.store(false, std::memory_order_relaxed);
        g_start_time    = Clock::now();
        g_time_limit_ms = hard_limit_ms;
        g_soft_limit_ms = soft_limit_ms;

        search_init();  // shared TT-age bump — once per `go`, not once per thread

        int nthreads = std::max(1, g_num_threads);

        // macOS (and several other libc's) give a plain std::thread/pthread
        // only a ~512KB default stack, vs. ~8MB for the process's main
        // thread — confirmed by measurement on this platform (8176KB vs.
        // 524KB) and by a real crash: the first Lazy SMP SPRT run hit a
        // helper-thread SIGBUS ("Thread stack size exceeded due to
        // excessive recursion") inside alpha_beta's normal recursion depth,
        // something 100+ single-threaded games never triggered because the
        // main thread's stack was always large enough. Explicit
        // pthread_attr_setstacksize (used directly instead of std::thread,
        // which has no portable way to request a non-default stack size)
        // gives every helper thread the same generous stack the main
        // thread already had.
        static constexpr size_t HELPER_STACK_SIZE = 16 * 1024 * 1024;

        std::vector<pthread_t> helpers;
        helpers.reserve(nthreads - 1);
        std::vector<HelperArgs> helper_args(nthreads > 0 ? nthreads - 1 : 0);
        for (int i = 1; i < nthreads; i++) {
            helper_args[i - 1] = {this, depth, soft_limit_ms};

            pthread_attr_t attr;
            pthread_attr_init(&attr);
            pthread_attr_setstacksize(&attr, HELPER_STACK_SIZE);

            pthread_t tid;
            pthread_create(&tid, &attr, &Engine::helper_thread_main, &helper_args[i - 1]);
            pthread_attr_destroy(&attr);
            helpers.push_back(tid);
        }

        Move best_move  = NO_MOVE;
        int  best_score = 0;
        iterate(board, depth, soft_limit_ms, /*report=*/true, best_move, best_score);

        for (pthread_t tid : helpers) pthread_join(tid, nullptr);

        return {best_move, best_score};
    }

private:
    struct HelperArgs {
        Engine  *engine;
        int      depth;
        int64_t  soft_limit_ms;
    };

    static void *helper_thread_main(void *raw) {
        HelperArgs *args = static_cast<HelperArgs *>(raw);
        Move discard_move; int discard_score;
        iterate(args->engine->board, args->depth, args->soft_limit_ms,
                /*report=*/false, discard_move, discard_score);
        return nullptr;
    }
};

// ─────────────────────────── Polyglot opening book ─────────────────────────
// Standard Polyglot book support (UCI "OwnBook"/"BookFile" options), so an
// opening book can be exercised through plain UCI -- and therefore through
// fastchess/SPRT -- unlike lichess-bot's book, which lives entirely in
// lichess-bot's Python layer and is structurally invisible to a UCI-only
// tool. See CLAUDE.md's opening-book section for why both exist.
//
// The 781-entry random table below is the standard Polyglot Zobrist array
// (Fabien Letouzey's original constants, used by every Polyglot-compatible
// tool). Copied verbatim from this repo's installed python-chess
// (chess/polyglot.py's POLYGLOT_RANDOM_ARRAY) rather than retyped from any
// external source, so it's byte-for-byte the same table python-chess (and
// therefore build_polyglot_book.py, which wrote our .bin files) already
// uses -- verified by direct hash comparison, see verify_polyglot_hash.py.
static constexpr uint64_t POLYGLOT_RANDOM_ARRAY[781] = {
    0x9D39247E33776D41ULL, 0x2AF7398005AAA5C7ULL, 0x44DB015024623547ULL, 0x9C15F73E62A76AE2ULL,
    0x75834465489C0C89ULL, 0x3290AC3A203001BFULL, 0x0FBBAD1F61042279ULL, 0xE83A908FF2FB60CAULL,
    0x0D7E765D58755C10ULL, 0x1A083822CEAFE02DULL, 0x9605D5F0E25EC3B0ULL, 0xD021FF5CD13A2ED5ULL,
    0x40BDF15D4A672E32ULL, 0x011355146FD56395ULL, 0x5DB4832046F3D9E5ULL, 0x239F8B2D7FF719CCULL,
    0x05D1A1AE85B49AA1ULL, 0x679F848F6E8FC971ULL, 0x7449BBFF801FED0BULL, 0x7D11CDB1C3B7ADF0ULL,
    0x82C7709E781EB7CCULL, 0xF3218F1C9510786CULL, 0x331478F3AF51BBE6ULL, 0x4BB38DE5E7219443ULL,
    0xAA649C6EBCFD50FCULL, 0x8DBD98A352AFD40BULL, 0x87D2074B81D79217ULL, 0x19F3C751D3E92AE1ULL,
    0xB4AB30F062B19ABFULL, 0x7B0500AC42047AC4ULL, 0xC9452CA81A09D85DULL, 0x24AA6C514DA27500ULL,
    0x4C9F34427501B447ULL, 0x14A68FD73C910841ULL, 0xA71B9B83461CBD93ULL, 0x03488B95B0F1850FULL,
    0x637B2B34FF93C040ULL, 0x09D1BC9A3DD90A94ULL, 0x3575668334A1DD3BULL, 0x735E2B97A4C45A23ULL,
    0x18727070F1BD400BULL, 0x1FCBACD259BF02E7ULL, 0xD310A7C2CE9B6555ULL, 0xBF983FE0FE5D8244ULL,
    0x9F74D14F7454A824ULL, 0x51EBDC4AB9BA3035ULL, 0x5C82C505DB9AB0FAULL, 0xFCF7FE8A3430B241ULL,
    0x3253A729B9BA3DDEULL, 0x8C74C368081B3075ULL, 0xB9BC6C87167C33E7ULL, 0x7EF48F2B83024E20ULL,
    0x11D505D4C351BD7FULL, 0x6568FCA92C76A243ULL, 0x4DE0B0F40F32A7B8ULL, 0x96D693460CC37E5DULL,
    0x42E240CB63689F2FULL, 0x6D2BDCDAE2919661ULL, 0x42880B0236E4D951ULL, 0x5F0F4A5898171BB6ULL,
    0x39F890F579F92F88ULL, 0x93C5B5F47356388BULL, 0x63DC359D8D231B78ULL, 0xEC16CA8AEA98AD76ULL,
    0x5355F900C2A82DC7ULL, 0x07FB9F855A997142ULL, 0x5093417AA8A7ED5EULL, 0x7BCBC38DA25A7F3CULL,
    0x19FC8A768CF4B6D4ULL, 0x637A7780DECFC0D9ULL, 0x8249A47AEE0E41F7ULL, 0x79AD695501E7D1E8ULL,
    0x14ACBAF4777D5776ULL, 0xF145B6BECCDEA195ULL, 0xDABF2AC8201752FCULL, 0x24C3C94DF9C8D3F6ULL,
    0xBB6E2924F03912EAULL, 0x0CE26C0B95C980D9ULL, 0xA49CD132BFBF7CC4ULL, 0xE99D662AF4243939ULL,
    0x27E6AD7891165C3FULL, 0x8535F040B9744FF1ULL, 0x54B3F4FA5F40D873ULL, 0x72B12C32127FED2BULL,
    0xEE954D3C7B411F47ULL, 0x9A85AC909A24EAA1ULL, 0x70AC4CD9F04F21F5ULL, 0xF9B89D3E99A075C2ULL,
    0x87B3E2B2B5C907B1ULL, 0xA366E5B8C54F48B8ULL, 0xAE4A9346CC3F7CF2ULL, 0x1920C04D47267BBDULL,
    0x87BF02C6B49E2AE9ULL, 0x092237AC237F3859ULL, 0xFF07F64EF8ED14D0ULL, 0x8DE8DCA9F03CC54EULL,
    0x9C1633264DB49C89ULL, 0xB3F22C3D0B0B38EDULL, 0x390E5FB44D01144BULL, 0x5BFEA5B4712768E9ULL,
    0x1E1032911FA78984ULL, 0x9A74ACB964E78CB3ULL, 0x4F80F7A035DAFB04ULL, 0x6304D09A0B3738C4ULL,
    0x2171E64683023A08ULL, 0x5B9B63EB9CEFF80CULL, 0x506AACF489889342ULL, 0x1881AFC9A3A701D6ULL,
    0x6503080440750644ULL, 0xDFD395339CDBF4A7ULL, 0xEF927DBCF00C20F2ULL, 0x7B32F7D1E03680ECULL,
    0xB9FD7620E7316243ULL, 0x05A7E8A57DB91B77ULL, 0xB5889C6E15630A75ULL, 0x4A750A09CE9573F7ULL,
    0xCF464CEC899A2F8AULL, 0xF538639CE705B824ULL, 0x3C79A0FF5580EF7FULL, 0xEDE6C87F8477609DULL,
    0x799E81F05BC93F31ULL, 0x86536B8CF3428A8CULL, 0x97D7374C60087B73ULL, 0xA246637CFF328532ULL,
    0x043FCAE60CC0EBA0ULL, 0x920E449535DD359EULL, 0x70EB093B15B290CCULL, 0x73A1921916591CBDULL,
    0x56436C9FE1A1AA8DULL, 0xEFAC4B70633B8F81ULL, 0xBB215798D45DF7AFULL, 0x45F20042F24F1768ULL,
    0x930F80F4E8EB7462ULL, 0xFF6712FFCFD75EA1ULL, 0xAE623FD67468AA70ULL, 0xDD2C5BC84BC8D8FCULL,
    0x7EED120D54CF2DD9ULL, 0x22FE545401165F1CULL, 0xC91800E98FB99929ULL, 0x808BD68E6AC10365ULL,
    0xDEC468145B7605F6ULL, 0x1BEDE3A3AEF53302ULL, 0x43539603D6C55602ULL, 0xAA969B5C691CCB7AULL,
    0xA87832D392EFEE56ULL, 0x65942C7B3C7E11AEULL, 0xDED2D633CAD004F6ULL, 0x21F08570F420E565ULL,
    0xB415938D7DA94E3CULL, 0x91B859E59ECB6350ULL, 0x10CFF333E0ED804AULL, 0x28AED140BE0BB7DDULL,
    0xC5CC1D89724FA456ULL, 0x5648F680F11A2741ULL, 0x2D255069F0B7DAB3ULL, 0x9BC5A38EF729ABD4ULL,
    0xEF2F054308F6A2BCULL, 0xAF2042F5CC5C2858ULL, 0x480412BAB7F5BE2AULL, 0xAEF3AF4A563DFE43ULL,
    0x19AFE59AE451497FULL, 0x52593803DFF1E840ULL, 0xF4F076E65F2CE6F0ULL, 0x11379625747D5AF3ULL,
    0xBCE5D2248682C115ULL, 0x9DA4243DE836994FULL, 0x066F70B33FE09017ULL, 0x4DC4DE189B671A1CULL,
    0x51039AB7712457C3ULL, 0xC07A3F80C31FB4B4ULL, 0xB46EE9C5E64A6E7CULL, 0xB3819A42ABE61C87ULL,
    0x21A007933A522A20ULL, 0x2DF16F761598AA4FULL, 0x763C4A1371B368FDULL, 0xF793C46702E086A0ULL,
    0xD7288E012AEB8D31ULL, 0xDE336A2A4BC1C44BULL, 0x0BF692B38D079F23ULL, 0x2C604A7A177326B3ULL,
    0x4850E73E03EB6064ULL, 0xCFC447F1E53C8E1BULL, 0xB05CA3F564268D99ULL, 0x9AE182C8BC9474E8ULL,
    0xA4FC4BD4FC5558CAULL, 0xE755178D58FC4E76ULL, 0x69B97DB1A4C03DFEULL, 0xF9B5B7C4ACC67C96ULL,
    0xFC6A82D64B8655FBULL, 0x9C684CB6C4D24417ULL, 0x8EC97D2917456ED0ULL, 0x6703DF9D2924E97EULL,
    0xC547F57E42A7444EULL, 0x78E37644E7CAD29EULL, 0xFE9A44E9362F05FAULL, 0x08BD35CC38336615ULL,
    0x9315E5EB3A129ACEULL, 0x94061B871E04DF75ULL, 0xDF1D9F9D784BA010ULL, 0x3BBA57B68871B59DULL,
    0xD2B7ADEEDED1F73FULL, 0xF7A255D83BC373F8ULL, 0xD7F4F2448C0CEB81ULL, 0xD95BE88CD210FFA7ULL,
    0x336F52F8FF4728E7ULL, 0xA74049DAC312AC71ULL, 0xA2F61BB6E437FDB5ULL, 0x4F2A5CB07F6A35B3ULL,
    0x87D380BDA5BF7859ULL, 0x16B9F7E06C453A21ULL, 0x7BA2484C8A0FD54EULL, 0xF3A678CAD9A2E38CULL,
    0x39B0BF7DDE437BA2ULL, 0xFCAF55C1BF8A4424ULL, 0x18FCF680573FA594ULL, 0x4C0563B89F495AC3ULL,
    0x40E087931A00930DULL, 0x8CFFA9412EB642C1ULL, 0x68CA39053261169FULL, 0x7A1EE967D27579E2ULL,
    0x9D1D60E5076F5B6FULL, 0x3810E399B6F65BA2ULL, 0x32095B6D4AB5F9B1ULL, 0x35CAB62109DD038AULL,
    0xA90B24499FCFAFB1ULL, 0x77A225A07CC2C6BDULL, 0x513E5E634C70E331ULL, 0x4361C0CA3F692F12ULL,
    0xD941ACA44B20A45BULL, 0x528F7C8602C5807BULL, 0x52AB92BEB9613989ULL, 0x9D1DFA2EFC557F73ULL,
    0x722FF175F572C348ULL, 0x1D1260A51107FE97ULL, 0x7A249A57EC0C9BA2ULL, 0x04208FE9E8F7F2D6ULL,
    0x5A110C6058B920A0ULL, 0x0CD9A497658A5698ULL, 0x56FD23C8F9715A4CULL, 0x284C847B9D887AAEULL,
    0x04FEABFBBDB619CBULL, 0x742E1E651C60BA83ULL, 0x9A9632E65904AD3CULL, 0x881B82A13B51B9E2ULL,
    0x506E6744CD974924ULL, 0xB0183DB56FFC6A79ULL, 0x0ED9B915C66ED37EULL, 0x5E11E86D5873D484ULL,
    0xF678647E3519AC6EULL, 0x1B85D488D0F20CC5ULL, 0xDAB9FE6525D89021ULL, 0x0D151D86ADB73615ULL,
    0xA865A54EDCC0F019ULL, 0x93C42566AEF98FFBULL, 0x99E7AFEABE000731ULL, 0x48CBFF086DDF285AULL,
    0x7F9B6AF1EBF78BAFULL, 0x58627E1A149BBA21ULL, 0x2CD16E2ABD791E33ULL, 0xD363EFF5F0977996ULL,
    0x0CE2A38C344A6EEDULL, 0x1A804AADB9CFA741ULL, 0x907F30421D78C5DEULL, 0x501F65EDB3034D07ULL,
    0x37624AE5A48FA6E9ULL, 0x957BAF61700CFF4EULL, 0x3A6C27934E31188AULL, 0xD49503536ABCA345ULL,
    0x088E049589C432E0ULL, 0xF943AEE7FEBF21B8ULL, 0x6C3B8E3E336139D3ULL, 0x364F6FFA464EE52EULL,
    0xD60F6DCEDC314222ULL, 0x56963B0DCA418FC0ULL, 0x16F50EDF91E513AFULL, 0xEF1955914B609F93ULL,
    0x565601C0364E3228ULL, 0xECB53939887E8175ULL, 0xBAC7A9A18531294BULL, 0xB344C470397BBA52ULL,
    0x65D34954DAF3CEBDULL, 0xB4B81B3FA97511E2ULL, 0xB422061193D6F6A7ULL, 0x071582401C38434DULL,
    0x7A13F18BBEDC4FF5ULL, 0xBC4097B116C524D2ULL, 0x59B97885E2F2EA28ULL, 0x99170A5DC3115544ULL,
    0x6F423357E7C6A9F9ULL, 0x325928EE6E6F8794ULL, 0xD0E4366228B03343ULL, 0x565C31F7DE89EA27ULL,
    0x30F5611484119414ULL, 0xD873DB391292ED4FULL, 0x7BD94E1D8E17DEBCULL, 0xC7D9F16864A76E94ULL,
    0x947AE053EE56E63CULL, 0xC8C93882F9475F5FULL, 0x3A9BF55BA91F81CAULL, 0xD9A11FBB3D9808E4ULL,
    0x0FD22063EDC29FCAULL, 0xB3F256D8ACA0B0B9ULL, 0xB03031A8B4516E84ULL, 0x35DD37D5871448AFULL,
    0xE9F6082B05542E4EULL, 0xEBFAFA33D7254B59ULL, 0x9255ABB50D532280ULL, 0xB9AB4CE57F2D34F3ULL,
    0x693501D628297551ULL, 0xC62C58F97DD949BFULL, 0xCD454F8F19C5126AULL, 0xBBE83F4ECC2BDECBULL,
    0xDC842B7E2819E230ULL, 0xBA89142E007503B8ULL, 0xA3BC941D0A5061CBULL, 0xE9F6760E32CD8021ULL,
    0x09C7E552BC76492FULL, 0x852F54934DA55CC9ULL, 0x8107FCCF064FCF56ULL, 0x098954D51FFF6580ULL,
    0x23B70EDB1955C4BFULL, 0xC330DE426430F69DULL, 0x4715ED43E8A45C0AULL, 0xA8D7E4DAB780A08DULL,
    0x0572B974F03CE0BBULL, 0xB57D2E985E1419C7ULL, 0xE8D9ECBE2CF3D73FULL, 0x2FE4B17170E59750ULL,
    0x11317BA87905E790ULL, 0x7FBF21EC8A1F45ECULL, 0x1725CABFCB045B00ULL, 0x964E915CD5E2B207ULL,
    0x3E2B8BCBF016D66DULL, 0xBE7444E39328A0ACULL, 0xF85B2B4FBCDE44B7ULL, 0x49353FEA39BA63B1ULL,
    0x1DD01AAFCD53486AULL, 0x1FCA8A92FD719F85ULL, 0xFC7C95D827357AFAULL, 0x18A6A990C8B35EBDULL,
    0xCCCB7005C6B9C28DULL, 0x3BDBB92C43B17F26ULL, 0xAA70B5B4F89695A2ULL, 0xE94C39A54A98307FULL,
    0xB7A0B174CFF6F36EULL, 0xD4DBA84729AF48ADULL, 0x2E18BC1AD9704A68ULL, 0x2DE0966DAF2F8B1CULL,
    0xB9C11D5B1E43A07EULL, 0x64972D68DEE33360ULL, 0x94628D38D0C20584ULL, 0xDBC0D2B6AB90A559ULL,
    0xD2733C4335C6A72FULL, 0x7E75D99D94A70F4DULL, 0x6CED1983376FA72BULL, 0x97FCAACBF030BC24ULL,
    0x7B77497B32503B12ULL, 0x8547EDDFB81CCB94ULL, 0x79999CDFF70902CBULL, 0xCFFE1939438E9B24ULL,
    0x829626E3892D95D7ULL, 0x92FAE24291F2B3F1ULL, 0x63E22C147B9C3403ULL, 0xC678B6D860284A1CULL,
    0x5873888850659AE7ULL, 0x0981DCD296A8736DULL, 0x9F65789A6509A440ULL, 0x9FF38FED72E9052FULL,
    0xE479EE5B9930578CULL, 0xE7F28ECD2D49EECDULL, 0x56C074A581EA17FEULL, 0x5544F7D774B14AEFULL,
    0x7B3F0195FC6F290FULL, 0x12153635B2C0CF57ULL, 0x7F5126DBBA5E0CA7ULL, 0x7A76956C3EAFB413ULL,
    0x3D5774A11D31AB39ULL, 0x8A1B083821F40CB4ULL, 0x7B4A38E32537DF62ULL, 0x950113646D1D6E03ULL,
    0x4DA8979A0041E8A9ULL, 0x3BC36E078F7515D7ULL, 0x5D0A12F27AD310D1ULL, 0x7F9D1A2E1EBE1327ULL,
    0xDA3A361B1C5157B1ULL, 0xDCDD7D20903D0C25ULL, 0x36833336D068F707ULL, 0xCE68341F79893389ULL,
    0xAB9090168DD05F34ULL, 0x43954B3252DC25E5ULL, 0xB438C2B67F98E5E9ULL, 0x10DCD78E3851A492ULL,
    0xDBC27AB5447822BFULL, 0x9B3CDB65F82CA382ULL, 0xB67B7896167B4C84ULL, 0xBFCED1B0048EAC50ULL,
    0xA9119B60369FFEBDULL, 0x1FFF7AC80904BF45ULL, 0xAC12FB171817EEE7ULL, 0xAF08DA9177DDA93DULL,
    0x1B0CAB936E65C744ULL, 0xB559EB1D04E5E932ULL, 0xC37B45B3F8D6F2BAULL, 0xC3A9DC228CAAC9E9ULL,
    0xF3B8B6675A6507FFULL, 0x9FC477DE4ED681DAULL, 0x67378D8ECCEF96CBULL, 0x6DD856D94D259236ULL,
    0xA319CE15B0B4DB31ULL, 0x073973751F12DD5EULL, 0x8A8E849EB32781A5ULL, 0xE1925C71285279F5ULL,
    0x74C04BF1790C0EFEULL, 0x4DDA48153C94938AULL, 0x9D266D6A1CC0542CULL, 0x7440FB816508C4FEULL,
    0x13328503DF48229FULL, 0xD6BF7BAEE43CAC40ULL, 0x4838D65F6EF6748FULL, 0x1E152328F3318DEAULL,
    0x8F8419A348F296BFULL, 0x72C8834A5957B511ULL, 0xD7A023A73260B45CULL, 0x94EBC8ABCFB56DAEULL,
    0x9FC10D0F989993E0ULL, 0xDE68A2355B93CAE6ULL, 0xA44CFE79AE538BBEULL, 0x9D1D84FCCE371425ULL,
    0x51D2B1AB2DDFB636ULL, 0x2FD7E4B9E72CD38CULL, 0x65CA5B96B7552210ULL, 0xDD69A0D8AB3B546DULL,
    0x604D51B25FBF70E2ULL, 0x73AA8A564FB7AC9EULL, 0x1A8C1E992B941148ULL, 0xAAC40A2703D9BEA0ULL,
    0x764DBEAE7FA4F3A6ULL, 0x1E99B96E70A9BE8BULL, 0x2C5E9DEB57EF4743ULL, 0x3A938FEE32D29981ULL,
    0x26E6DB8FFDF5ADFEULL, 0x469356C504EC9F9DULL, 0xC8763C5B08D1908CULL, 0x3F6C6AF859D80055ULL,
    0x7F7CC39420A3A545ULL, 0x9BFB227EBDF4C5CEULL, 0x89039D79D6FC5C5CULL, 0x8FE88B57305E2AB6ULL,
    0xA09E8C8C35AB96DEULL, 0xFA7E393983325753ULL, 0xD6B6D0ECC617C699ULL, 0xDFEA21EA9E7557E3ULL,
    0xB67C1FA481680AF8ULL, 0xCA1E3785A9E724E5ULL, 0x1CFC8BED0D681639ULL, 0xD18D8549D140CAEAULL,
    0x4ED0FE7E9DC91335ULL, 0xE4DBF0634473F5D2ULL, 0x1761F93A44D5AEFEULL, 0x53898E4C3910DA55ULL,
    0x734DE8181F6EC39AULL, 0x2680B122BAA28D97ULL, 0x298AF231C85BAFABULL, 0x7983EED3740847D5ULL,
    0x66C1A2A1A60CD889ULL, 0x9E17E49642A3E4C1ULL, 0xEDB454E7BADC0805ULL, 0x50B704CAB602C329ULL,
    0x4CC317FB9CDDD023ULL, 0x66B4835D9EAFEA22ULL, 0x219B97E26FFC81BDULL, 0x261E4E4C0A333A9DULL,
    0x1FE2CCA76517DB90ULL, 0xD7504DFA8816EDBBULL, 0xB9571FA04DC089C8ULL, 0x1DDC0325259B27DEULL,
    0xCF3F4688801EB9AAULL, 0xF4F5D05C10CAB243ULL, 0x38B6525C21A42B0EULL, 0x36F60E2BA4FA6800ULL,
    0xEB3593803173E0CEULL, 0x9C4CD6257C5A3603ULL, 0xAF0C317D32ADAA8AULL, 0x258E5A80C7204C4BULL,
    0x8B889D624D44885DULL, 0xF4D14597E660F855ULL, 0xD4347F66EC8941C3ULL, 0xE699ED85B0DFB40DULL,
    0x2472F6207C2D0484ULL, 0xC2A1E7B5B459AEB5ULL, 0xAB4F6451CC1D45ECULL, 0x63767572AE3D6174ULL,
    0xA59E0BD101731A28ULL, 0x116D0016CB948F09ULL, 0x2CF9C8CA052F6E9FULL, 0x0B090A7560A968E3ULL,
    0xABEEDDB2DDE06FF1ULL, 0x58EFC10B06A2068DULL, 0xC6E57A78FBD986E0ULL, 0x2EAB8CA63CE802D7ULL,
    0x14A195640116F336ULL, 0x7C0828DD624EC390ULL, 0xD74BBE77E6116AC7ULL, 0x804456AF10F5FB53ULL,
    0xEBE9EA2ADF4321C7ULL, 0x03219A39EE587A30ULL, 0x49787FEF17AF9924ULL, 0xA1E9300CD8520548ULL,
    0x5B45E522E4B1B4EFULL, 0xB49C3B3995091A36ULL, 0xD4490AD526F14431ULL, 0x12A8F216AF9418C2ULL,
    0x001F837CC7350524ULL, 0x1877B51E57A764D5ULL, 0xA2853B80F17F58EEULL, 0x993E1DE72D36D310ULL,
    0xB3598080CE64A656ULL, 0x252F59CF0D9F04BBULL, 0xD23C8E176D113600ULL, 0x1BDA0492E7E4586EULL,
    0x21E0BD5026C619BFULL, 0x3B097ADAF088F94EULL, 0x8D14DEDB30BE846EULL, 0xF95CFFA23AF5F6F4ULL,
    0x3871700761B3F743ULL, 0xCA672B91E9E4FA16ULL, 0x64C8E531BFF53B55ULL, 0x241260ED4AD1E87DULL,
    0x106C09B972D2E822ULL, 0x7FBA195410E5CA30ULL, 0x7884D9BC6CB569D8ULL, 0x0647DFEDCD894A29ULL,
    0x63573FF03E224774ULL, 0x4FC8E9560F91B123ULL, 0x1DB956E450275779ULL, 0xB8D91274B9E9D4FBULL,
    0xA2EBEE47E2FBFCE1ULL, 0xD9F1F30CCD97FB09ULL, 0xEFED53D75FD64E6BULL, 0x2E6D02C36017F67FULL,
    0xA9AA4D20DB084E9BULL, 0xB64BE8D8B25396C1ULL, 0x70CB6AF7C2D5BCF0ULL, 0x98F076A4F7A2322EULL,
    0xBF84470805E69B5FULL, 0x94C3251F06F90CF3ULL, 0x3E003E616A6591E9ULL, 0xB925A6CD0421AFF3ULL,
    0x61BDD1307C66E300ULL, 0xBF8D5108E27E0D48ULL, 0x240AB57A8B888B20ULL, 0xFC87614BAF287E07ULL,
    0xEF02CDD06FFDB432ULL, 0xA1082C0466DF6C0AULL, 0x8215E577001332C8ULL, 0xD39BB9C3A48DB6CFULL,
    0x2738259634305C14ULL, 0x61CF4F94C97DF93DULL, 0x1B6BACA2AE4E125BULL, 0x758F450C88572E0BULL,
    0x959F587D507A8359ULL, 0xB063E962E045F54DULL, 0x60E8ED72C0DFF5D1ULL, 0x7B64978555326F9FULL,
    0xFD080D236DA814BAULL, 0x8C90FD9B083F4558ULL, 0x106F72FE81E2C590ULL, 0x7976033A39F7D952ULL,
    0xA4EC0132764CA04BULL, 0x733EA705FAE4FA77ULL, 0xB4D8F77BC3E56167ULL, 0x9E21F4F903B33FD9ULL,
    0x9D765E419FB69F6DULL, 0xD30C088BA61EA5EFULL, 0x5D94337FBFAF7F5BULL, 0x1A4E4822EB4D7A59ULL,
    0x6FFE73E81B637FB3ULL, 0xDDF957BC36D8B9CAULL, 0x64D0E29EEA8838B3ULL, 0x08DD9BDFD96B9F63ULL,
    0x087E79E5A57D1D13ULL, 0xE328E230E3E2B3FBULL, 0x1C2559E30F0946BEULL, 0x720BF5F26F4D2EAAULL,
    0xB0774D261CC609DBULL, 0x443F64EC5A371195ULL, 0x4112CF68649A260EULL, 0xD813F2FAB7F5C5CAULL,
    0x660D3257380841EEULL, 0x59AC2C7873F910A3ULL, 0xE846963877671A17ULL, 0x93B633ABFA3469F8ULL,
    0xC0C0F5A60EF4CDCFULL, 0xCAF21ECD4377B28CULL, 0x57277707199B8175ULL, 0x506C11B9D90E8B1DULL,
    0xD83CC2687A19255FULL, 0x4A29C6465A314CD1ULL, 0xED2DF21216235097ULL, 0xB5635C95FF7296E2ULL,
    0x22AF003AB672E811ULL, 0x52E762596BF68235ULL, 0x9AEBA33AC6ECC6B0ULL, 0x944F6DE09134DFB6ULL,
    0x6C47BEC883A7DE39ULL, 0x6AD047C430A12104ULL, 0xA5B1CFDBA0AB4067ULL, 0x7C45D833AFF07862ULL,
    0x5092EF950A16DA0BULL, 0x9338E69C052B8E7BULL, 0x455A4B4CFE30E3F5ULL, 0x6B02E63195AD0CF8ULL,
    0x6B17B224BAD6BF27ULL, 0xD1E0CCD25BB9C169ULL, 0xDE0C89A556B9AE70ULL, 0x50065E535A213CF6ULL,
    0x9C1169FA2777B874ULL, 0x78EDEFD694AF1EEDULL, 0x6DC93D9526A50E68ULL, 0xEE97F453F06791EDULL,
    0x32AB0EDB696703D3ULL, 0x3A6853C7E70757A7ULL, 0x31865CED6120F37DULL, 0x67FEF95D92607890ULL,
    0x1F2B1D1F15F6DC9CULL, 0xB69E38A8965C6B65ULL, 0xAA9119FF184CCCF4ULL, 0xF43C732873F24C13ULL,
    0xFB4A3D794A9A80D2ULL, 0x3550C2321FD6109CULL, 0x371F77E76BB8417EULL, 0x6BFA9AAE5EC05779ULL,
    0xCD04F3FF001A4778ULL, 0xE3273522064480CAULL, 0x9F91508BFFCFC14AULL, 0x049A7F41061A9E60ULL,
    0xFCB6BE43A9F2FE9BULL, 0x08DE8A1C7797DA9BULL, 0x8F9887E6078735A1ULL, 0xB5B4071DBFC73A66ULL,
    0x230E343DFBA08D33ULL, 0x43ED7F5A0FAE657DULL, 0x3A88A0FBBCB05C63ULL, 0x21874B8B4D2DBC4FULL,
    0x1BDEA12E35F6A8C9ULL, 0x53C065C6C8E63528ULL, 0xE34A1D250E7A8D6BULL, 0xD6B04D3B7651DD7EULL,
    0x5E90277E7CB39E2DULL, 0x2C046F22062DC67DULL, 0xB10BB459132D0A26ULL, 0x3FA9DDFB67E2F199ULL,
    0x0E09B88E1914F7AFULL, 0x10E8B35AF3EEAB37ULL, 0x9EEDECA8E272B933ULL, 0xD4C718BC4AE8AE5FULL,
    0x81536D601170FC20ULL, 0x91B534F885818A06ULL, 0xEC8177F83F900978ULL, 0x190E714FADA5156EULL,
    0xB592BF39B0364963ULL, 0x89C350C893AE7DC1ULL, 0xAC042E70F8B383F2ULL, 0xB49B52E587A1EE60ULL,
    0xFB152FE3FF26DA89ULL, 0x3E666E6F69AE2C15ULL, 0x3B544EBE544C19F9ULL, 0xE805A1E290CF2456ULL,
    0x24B33C9D7ED25117ULL, 0xE74733427B72F0C1ULL, 0x0A804D18B7097475ULL, 0x57E3306D881EDB4FULL,
    0x4AE7D6A36EB5DBCBULL, 0x2D8D5432157064C8ULL, 0xD1E649DE1E7F268BULL, 0x8A328A1CEDFE552CULL,
    0x07A3AEC79624C7DAULL, 0x84547DDC3E203C94ULL, 0x990A98FD5071D263ULL, 0x1A4FF12616EEFC89ULL,
    0xF6F7FD1431714200ULL, 0x30C05B1BA332F41CULL, 0x8D2636B81555A786ULL, 0x46C9FEB55D120902ULL,
    0xCCEC0A73B49C9921ULL, 0x4E9D2827355FC492ULL, 0x19EBB029435DCB0FULL, 0x4659D2B743848A2CULL,
    0x963EF2C96B33BE31ULL, 0x74F85198B05A2E7DULL, 0x5A0F544DD2B1FB18ULL, 0x03727073C2E134B1ULL,
    0xC7F6AA2DE59AEA61ULL, 0x352787BAA0D7C22FULL, 0x9853EAB63B5E0B35ULL, 0xABBDCDD7ED5C0860ULL,
    0xCF05DAF5AC8D77B0ULL, 0x49CAD48CEBF4A71EULL, 0x7A4C10EC2158C4A6ULL, 0xD9E92AA246BF719EULL,
    0x13AE978D09FE5557ULL, 0x730499AF921549FFULL, 0x4E4B705B92903BA4ULL, 0xFF577222C14F0A3AULL,
    0x55B6344CF97AAFAEULL, 0xB862225B055B6960ULL, 0xCAC09AFBDDD2CDB4ULL, 0xDAF8E9829FE96B5FULL,
    0xB5FDFC5D3132C498ULL, 0x310CB380DB6F7503ULL, 0xE87FBB46217A360EULL, 0x2102AE466EBB1148ULL,
    0xF8549E1A3AA5E00DULL, 0x07A69AFDCC42261AULL, 0xC4C118BFE78FEAAEULL, 0xF9F4892ED96BD438ULL,
    0x1AF3DBE25D8F45DAULL, 0xF5B4B0B0D2DEEEB4ULL, 0x962ACEEFA82E1C84ULL, 0x046E3ECAAF453CE9ULL,
    0xF05D129681949A4CULL, 0x964781CE734B3C84ULL, 0x9C2ED44081CE5FBDULL, 0x522E23F3925E319EULL,
    0x177E00F9FC32F791ULL, 0x2BC60A63A6F3B3F2ULL, 0x222BBFAE61725606ULL, 0x486289DDCC3D6780ULL,
    0x7DC7785B8EFDFC80ULL, 0x8AF38731C02BA980ULL, 0x1FAB64EA29A2DDF7ULL, 0xE4D9429322CD065AULL,
    0x9DA058C67844F20CULL, 0x24C0E332B70019B0ULL, 0x233003B5A6CFE6ADULL, 0xD586BD01C5C217F6ULL,
    0x5E5637885F29BC2BULL, 0x7EBA726D8C94094BULL, 0x0A56A5F0BFE39272ULL, 0xD79476A84EE20D06ULL,
    0x9E4C1269BAA4BF37ULL, 0x17EFEE45B0DEE640ULL, 0x1D95B0A5FCF90BC6ULL, 0x93CBE0B699C2585DULL,
    0x65FA4F227A2B6D79ULL, 0xD5F9E858292504D5ULL, 0xC2B5A03F71471A6FULL, 0x59300222B4561E00ULL,
    0xCE2F8642CA0712DCULL, 0x7CA9723FBB2E8988ULL, 0x2785338347F2BA08ULL, 0xC61BB3A141E50E8CULL,
    0x150F361DAB9DEC26ULL, 0x9F6A419D382595F4ULL, 0x64A53DC924FE7AC9ULL, 0x142DE49FFF7A7C3DULL,
    0x0C335248857FA9E7ULL, 0x0A9C32D5EAE45305ULL, 0xE6C42178C4BBB92EULL, 0x71F1CE2490D20B07ULL,
    0xF1BCC3D275AFE51AULL, 0xE728E8C83C334074ULL, 0x96FBF83A12884624ULL, 0x81A1549FD6573DA5ULL,
    0x5FA7867CAF35E149ULL, 0x56986E2EF3ED091BULL, 0x917F1DD5F8886C61ULL, 0xD20D8C88C8FFE65FULL,
    0x31D71DCE64B2C310ULL, 0xF165B587DF898190ULL, 0xA57E6339DD2CF3A0ULL, 0x1EF6E6DBB1961EC9ULL,
    0x70CC73D90BC26E24ULL, 0xE21A6B35DF0C3AD7ULL, 0x003A93D8B2806962ULL, 0x1C99DED33CB890A1ULL,
    0xCF3145DE0ADD4289ULL, 0xD0E4427A5514FB72ULL, 0x77C621CC9FB3A483ULL, 0x67A34DAC4356550BULL,
    0xF8D626AAAF278509ULL
};

struct PolyglotBookEntry {
    uint64_t key;
    uint16_t raw_move;
    uint16_t weight;
};

static std::vector<PolyglotBookEntry> g_book;
static bool                           g_own_book = false;
static bool                           g_book_loaded = false;
static std::mt19937_64                g_book_rng(std::random_device{}());

static inline uint64_t be64(const unsigned char *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v = (v << 8) | p[i];
    return v;
}
static inline uint16_t be16(const unsigned char *p) {
    return (uint16_t)((p[0] << 8) | p[1]);
}

// Loads a standard Polyglot .bin file (16 bytes/entry: key, move, weight,
// learn, all big-endian -- see build_polyglot_book.py). Entries are assumed
// sorted by key ascending (true of any book produced by that script, and of
// every real-world Polyglot book); sorted defensively anyway since the
// lookup below relies on it.
static bool load_polyglot_book(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "info string Failed to open book file " << path << "\n"; return false; }
    std::vector<unsigned char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf.size() % 16 != 0) {
        std::cerr << "info string Book file " << path << " has size not a multiple of 16, ignoring\n";
        return false;
    }
    g_book.clear();
    g_book.reserve(buf.size() / 16);
    for (size_t off = 0; off + 16 <= buf.size(); off += 16) {
        PolyglotBookEntry e;
        e.key      = be64(&buf[off]);
        e.raw_move = be16(&buf[off + 8]);
        e.weight   = be16(&buf[off + 10]);
        g_book.push_back(e);
    }
    std::sort(g_book.begin(), g_book.end(),
              [](const PolyglotBookEntry &a, const PolyglotBookEntry &b) { return a.key < b.key; });
    std::cerr << "info string Loaded book " << path << " (" << g_book.size() << " entries)\n";
    return !g_book.empty();
}

// Polyglot Zobrist hash of the current position. Piece-index convention:
// index = our_piece_type*2 + (1 - color), matching python-chess's
// (python_piece_type-1)*2 + (1 if white else 0) -- our PieceType enum
// (PAWN=0..KING=5) is exactly python-chess's minus 1, so this simplifies
// to that form directly. Verified byte-for-byte against
// chess.polyglot.zobrist_hash() on real positions incl. castling rights
// and en-passant-eligible positions -- see verify_polyglot_hash.py.
static uint64_t polyglot_hash(const Board &board) {
    uint64_t h = 0;
    for (int c = 0; c < 2; c++) {
        int pivot = (c == WHITE) ? 1 : 0;
        for (int pt = 0; pt < 6; pt++) {
            Bitboard bb = board.bb[c][pt];
            while (bb) {
                Square sq = lsb(bb);
                bb &= bb - 1;
                int piece_index = pt * 2 + pivot;
                h ^= POLYGLOT_RANDOM_ARRAY[64 * piece_index + sq];
            }
        }
    }
    if (board.cur.castle_rights & CR_WK) h ^= POLYGLOT_RANDOM_ARRAY[768];
    if (board.cur.castle_rights & CR_WQ) h ^= POLYGLOT_RANDOM_ARRAY[769];
    if (board.cur.castle_rights & CR_BK) h ^= POLYGLOT_RANDOM_ARRAY[770];
    if (board.cur.castle_rights & CR_BQ) h ^= POLYGLOT_RANDOM_ARRAY[771];
    if (board.cur.ep_square >= 0) {
        int epsq = board.cur.ep_square;
        int capturer_rank_sq = (board.stm == WHITE) ? epsq - 8 : epsq + 8;
        Bitboard adjacent = 0;
        if (sq_file(capturer_rank_sq) > 0) adjacent |= sq_bb(capturer_rank_sq - 1);
        if (sq_file(capturer_rank_sq) < 7) adjacent |= sq_bb(capturer_rank_sq + 1);
        if (adjacent & board.bb[board.stm][PAWN])
            h ^= POLYGLOT_RANDOM_ARRAY[772 + sq_file(epsq)];
    }
    if (board.stm == WHITE) h ^= POLYGLOT_RANDOM_ARRAY[780];
    return h;
}

// Looks up the current position in the loaded book and, if found, returns a
// weighted-random legal move (mirrors lichess-bot's "weighted_random"
// selection so the two book paths behave the same way). The raw Polyglot
// move only encodes from/to/promotion, not special-move flags (castle/ep),
// so the decoded from/to/promo is matched against a freshly generated legal
// move list and the *generated* move object (with correct flags) is
// returned -- this also means a corrupt or mismatched book entry can never
// produce an illegal move, it just falls through to NO_MOVE (-> search).
static Move probe_book(Board &board) {
    if (!g_book_loaded || g_book.empty()) return NO_MOVE;
    uint64_t key = polyglot_hash(board);
    auto lo = std::lower_bound(g_book.begin(), g_book.end(), key,
                                [](const PolyglotBookEntry &e, uint64_t k) { return e.key < k; });
    if (lo == g_book.end() || lo->key != key) return NO_MOVE;
    auto hi = lo;
    while (hi != g_book.end() && hi->key == key) ++hi;

    Move legal[MAX_MOVES];
    int n_legal = board.gen_moves(legal);

    uint32_t total_weight = 0;
    for (auto it = lo; it != hi; ++it) total_weight += it->weight;
    if (total_weight == 0) return NO_MOVE;

    std::uniform_int_distribution<uint32_t> dist(0, total_weight - 1);
    uint32_t pick = dist(g_book_rng);
    for (auto it = lo; it != hi; ++it) {
        if (pick < it->weight) {
            Square to   = it->raw_move & 63;
            Square from = (it->raw_move >> 6) & 63;
            int promo_part = (it->raw_move >> 12) & 7;  // == our PieceType enum directly (see header comment)
            for (int i = 0; i < n_legal; i++) {
                Move m = legal[i];
                PieceType mp = (move_flags(m) == MF_PROMO) ? move_promo(m) : NO_PIECE;
                int want_promo = promo_part ? promo_part : (int)NO_PIECE;
                if (move_from(m) == from && move_to(m) == to && (int)mp == want_promo) return m;
            }
            return NO_MOVE;  // decoded move not legal here -- corrupt/mismatched entry, ignore
        }
        pick -= it->weight;
    }
    return NO_MOVE;
}


// ─────────────────────────── UCI loop ──────────────────────────────────────

int main() {
    g_time_debug = std::getenv("NNUE_TIME_DEBUG") != nullptr;
    g_prune_debug = std::getenv("NNUE_PRUNE_DEBUG") != nullptr;
    init_magics();
    init_zobrist();
    init_lmr();       // static table; only needs to be built once
    tt_resize(128);   // default 128 MB TT; overridden by setoption Hash

    Engine engine;

    std::string line;

    std::cout << "id name NNUEEngine\nid author nnue-trainer\n";
    std::cout << "option name NNFile type string default <empty>\n";
    std::cout << "option name Hash type spin default 128 min 1 max 2048\n";
    std::cout << "option name Threads type spin default 1 min 1 max 64\n";
    std::cout << "option name OwnBook type check default false\n";
    std::cout << "option name BookFile type string default <empty>\n";
    std::cout << "uciok\n" << std::flush;

    int    depth      = 64;
    int64_t movetime  = -1;

    while (std::getline(std::cin, line)) {
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;

        if (cmd == "uci") {
            std::cout << "id name NNUEEngine\nid author nnue-trainer\n";
            std::cout << "option name NNFile type string default <empty>\n";
            std::cout << "option name Hash type spin default 128 min 1 max 2048\n";
            std::cout << "option name Threads type spin default 1 min 1 max 64\n";
    std::cout << "option name OwnBook type check default false\n";
    std::cout << "option name BookFile type string default <empty>\n";
            std::cout << "uciok\n" << std::flush;

        } else if (cmd == "isready") {
            std::cout << "readyok\n" << std::flush;

        } else if (cmd == "ucinewgame") {
            engine.board.set_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            tt_clear();
            history_clear();

        } else if (cmd == "setoption") {
            std::string tok, name, value;
            ss >> tok;  // "name"
            ss >> name;
            std::string n2; ss >> n2; if (n2 != "value") name += " " + n2;
            ss >> value;
            if (name == "NNFile" && !value.empty() && value != "<empty>") {
                engine.set_nnue(value);
            }
            if (name == "Hash") {
                tt_resize(std::stoi(value));
            }
            if (name == "Threads") {
                g_num_threads = std::max(1, std::stoi(value));
            }
            if (name == "OwnBook") {
                g_own_book = (value == "true");
            }
            if (name == "BookFile" && !value.empty() && value != "<empty>") {
                g_book_loaded = load_polyglot_book(value);
            }

        } else if (cmd == "position") {
            std::string token;
            ss >> token;
            if (token == "startpos") {
                engine.board.set_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                ss >> token;  // optionally "moves"
            } else if (token == "fen") {
                std::string fen;
                for (int i = 0; i < 6; i++) {
                    std::string part; ss >> part;
                    if (i) fen += ' ';
                    if (part == "moves") { token = part; break; }
                    fen += part;
                    token = "";
                }
                engine.board.set_from_fen(fen);
                ss >> token;  // "moves" or nothing
            }
            if (token == "moves" || (std::string(line).find("moves") != std::string::npos && token.empty())) {
                std::string mv;
                while (ss >> mv) engine.board.do_move(engine.board.parse_uci(mv));
            }

        } else if (cmd == "go") {
            auto t_go_recv = Clock::now();
            if (g_time_debug) std::cerr << "[TIMEDBG] go_recv raw=\"" << line << "\"\n";
            movetime = -1;
            depth    = 64;
            int64_t go_nodes = -1;
            bool infinite = false;
            int wtime = -1, btime = -1, winc = 0, binc = 0, movestogo = 30;
            std::string tok;
            while (ss >> tok) {
                if      (tok == "movetime")  ss >> movetime;
                else if (tok == "depth")     ss >> depth;
                else if (tok == "nodes")     ss >> go_nodes;
                else if (tok == "infinite")  infinite = true;
                else if (tok == "wtime")     ss >> wtime;
                else if (tok == "btime")     ss >> btime;
                else if (tok == "winc")      ss >> winc;
                else if (tok == "binc")      ss >> binc;
                else if (tok == "movestogo") ss >> movestogo;
            }
            // go infinite: search until "stop" — no time or node limit
            if (infinite) { movetime = -1; go_nodes = -1; }
            // Time management: soft_limit is the "normal" per-move
            // allocation (the same formula as before); hard_limit is a
            // generous cap that the soft-limit/instability-extension logic
            // in search() is allowed to grow into, bounded so a single move
            // can never eat more than half the remaining clock. -1 disables
            // the soft-limit-early-exit path entirely (used for fixed
            // `go movetime`/`go depth`/`go infinite`, where the full budget
            // should always be used, not cut short).
            int64_t soft_limit = -1, hard_limit = movetime;
            if (!infinite && movetime < 0 && go_nodes < 0 && (wtime >= 0 || btime >= 0)) {
                int myTime = (engine.board.stm == WHITE) ? wtime : btime;
                int myInc  = (engine.board.stm == WHITE) ? winc  : binc;
                if (myTime >= 0) {
                    // safety_cap bounds both soft and hard at a quarter of the
                    // remaining clock (was half -- see CLAUDE.md's "Lichess time
                    // losses" section for the original derivation of this
                    // formula's shape). Capping soft *before* deriving hard from
                    // it (rather than capping hard and then flooring back up to
                    // soft, as an older version did) means hard = min(cap,
                    // soft*2.5) is always >= soft automatically -- no separate
                    // max-with-soft step needed, and critically, hard can never
                    // end up above safety_cap the way an old max(hard_limit,
                    // soft_limit) step let it.
                    //
                    // Tightened from /2 and *4 to /4 and *2.5: real-game logs
                    // (see CLAUDE.md, "time limit too aggressive in the
                    // opening/early middlegame" follow-up) showed the
                    // best-move-instability stretch (effective_soft *= 1.3 per
                    // event, see iterate() above) routinely walking a move's
                    // spend out to 3-4x the nominal soft budget in the opening,
                    // where near-equal quiet moves flip the shallow-depth best
                    // move often -- draining the clock early and leaving the
                    // engine in a time scramble by move 20-25 of a 3-minute
                    // game. This doesn't change the *pacing* formula
                    // (myTime/movestogo + myInc*0.8, still the target "normal"
                    // per-move allocation) -- it only tightens how far a single
                    // move's instability/iteration-overshoot is allowed to blow
                    // past that normal allocation.
                    int64_t safety_cap = std::max((int64_t)10, (int64_t)myTime / 4);
                    soft_limit = std::min(safety_cap,
                        std::max((int64_t)50, (int64_t)(myTime / movestogo + myInc * 0.8)));
                    hard_limit = std::min(safety_cap, (int64_t)(soft_limit * 2.5));
                }
            }
            // Book consult: same gating condition as the timed-game branch
            // above (real games only -- wtime/btime present, not a fixed
            // `go depth`/`go movetime`/analysis query), matching how
            // lichess-bot's own book only fires during real play.
            if (g_own_book && !infinite && movetime < 0 && go_nodes < 0 && (wtime >= 0 || btime >= 0)) {
                Move book_mv = probe_book(engine.board);
                if (book_mv != NO_MOVE) {
                    std::cout << "bestmove " << engine.board.move_uci(book_mv) << "\n" << std::flush;
                    continue;
                }
            }
            g_node_limit = go_nodes;   // -1 if not set; search() will clear and re-apply
            auto [mv, score] = engine.search(depth, soft_limit, hard_limit);
            auto t_after_search = Clock::now();
            int64_t search_wall_ms = elapsed_ms();  // uses g_start_time set inside search()
            std::string best = (mv != NO_MOVE) ? engine.board.move_uci(mv) : "0000";
            std::cout << "bestmove " << best << "\n" << std::flush;
            if (g_time_debug) {
                auto t_after_print = Clock::now();
                int64_t recv_to_return_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    t_after_search - t_go_recv).count();
                int64_t print_overhead_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    t_after_print - t_after_search).count();
                int64_t unpolled_gap_ms = search_wall_ms - g_last_poll_ms;
                int64_t final_poll_interval_ms = g_last_poll_ms - g_prev_poll_ms;
                std::cerr << "[TIMEDBG] budget=" << hard_limit
                          << " soft=" << soft_limit
                          << " search_wall=" << search_wall_ms
                          << " recv_to_return=" << recv_to_return_ms
                          << " print_overhead=" << print_overhead_ms
                          << " last_poll_to_return_gap=" << unpolled_gap_ms
                          << " final_poll_interval=" << final_poll_interval_ms
                          << " max_poll_gap=" << g_max_poll_gap_ms
                          << " overrun=" << (search_wall_ms - hard_limit) << "\n" << std::flush;
            }

        } else if (cmd == "perft") {
            // Movegen regression test, no search involved. Usage:
            //   perft <depth>          -> total legal-move-tree node count
            //   perft divide <depth>   -> per-root-move breakdown + total
            // Compare against known node counts for standard test positions
            // (see CLAUDE.md) before trusting any movegen change.
            std::string sub; ss >> sub;
            int pd; uint64_t total;
            auto t0 = std::chrono::steady_clock::now();
            if (sub == "divide") {
                ss >> pd;
                total = perft_divide(engine.board, pd);
            } else {
                pd = std::stoi(sub);
                total = perft(engine.board, pd);
            }
            auto t1 = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "perft " << pd << " nodes " << total
                      << " time " << secs << "s"
                      << " nps " << (secs > 0 ? (uint64_t)(total / secs) : total)
                      << "\n" << std::flush;

        } else if (cmd == "polyhash") {
            // Diagnostic-only: print the Polyglot Zobrist hash of the current
            // position, for cross-checking against python-chess's
            // chess.polyglot.zobrist_hash() -- see verify_polyglot_hash.py.
            // Purely additive.
            char buf[32];
            std::snprintf(buf, sizeof buf, "%016llx", (unsigned long long)polyglot_hash(engine.board));
            std::cout << "polyhash " << buf << "\n" << std::flush;

        } else if (cmd == "eval") {
            // Diagnostic-only: print the raw static NNUE eval (no search) of
            // the current position, side-to-move relative, in centipawns.
            // Purely additive — does not alter behavior of any other command.
            if (!g_weights.loaded) {
                std::cout << "info string no net loaded\n" << std::flush;
            } else {
                Accumulator acc;
                acc.reset(engine.board);
                int bkt = get_bucket(engine.board);
                int ev  = nnue_eval(acc.stack[acc.top],
                                    engine.board.stm == WHITE, bkt);
                std::cout << "eval " << ev << " bucket " << bkt << "\n" << std::flush;
            }

        } else if (cmd == "prunestats") {
            // Diagnostic-only: dump cumulative pruning firing-rate / node-by-
            // depth counters accumulated since the last reset (or process
            // start). Only accumulates when NNUE_PRUNE_DEBUG is set in the
            // environment; otherwise all counters stay at 0. "prunestats
            // reset" zeroes them without printing, for use as a per-position
            // boundary in a driver script. Purely additive.
            std::string sub; ss >> sub;
            if (sub == "reset") {
                prune_stats_reset();
            } else {
                std::cout << "prunestats nonleaf_nodes " << g_nonleaf_nodes
                          << " rfp_checked " << g_rfp_checked << " rfp_hits " << g_rfp_hits
                          << " nmp_checked " << g_nmp_checked << " nmp_hits " << g_nmp_hits
                          << " probcut_checked " << g_probcut_checked << " probcut_hits " << g_probcut_hits
                          << " futility_skips " << g_futility_skips
                          << " lmp_skips " << g_lmp_skips
                          << " histprune_skips " << g_histprune_skips
                          << "\n";
                std::cout << "prunestats_depth";
                for (int d = 0; d < 64; d++)
                    if (g_nodes_by_depth[d]) std::cout << " " << d << ":" << g_nodes_by_depth[d];
                std::cout << "\n";

                std::cout << "prunestats_rfp_by_depth";
                for (int d = 0; d < 64; d++)
                    if (g_rfp_checked_by_depth[d])
                        std::cout << " " << d << ":" << g_rfp_hits_by_depth[d] << "/" << g_rfp_checked_by_depth[d];
                std::cout << "\n";

                std::cout << "prunestats_futility_by_depth";
                for (int d = 0; d < 64; d++)
                    if (g_futility_skips_by_depth[d]) std::cout << " " << d << ":" << g_futility_skips_by_depth[d];
                std::cout << "\n";

                std::cout << "prunestats_lmp_by_depth";
                for (int d = 0; d < 64; d++)
                    if (g_lmp_skips_by_depth[d]) std::cout << " " << d << ":" << g_lmp_skips_by_depth[d];
                std::cout << "\n";

                std::cout << "prunestats_rfp_coefs";
                for (int ci = 0; ci < N_RFP_CANDIDATES; ci++)
                    std::cout << " " << RFP_CANDIDATE_COEFS[ci] << ":" << g_rfp_hits_at_coef[ci];
                std::cout << "\n";

                for (int ci = 0; ci < N_RFP_CANDIDATES; ci++) {
                    std::cout << "prunestats_rfp_coef_by_depth " << RFP_CANDIDATE_COEFS[ci];
                    for (int d = 0; d < 64; d++)
                        if (g_rfp_hits_at_coef_by_depth[ci][d])
                            std::cout << " " << d << ":" << g_rfp_hits_at_coef_by_depth[ci][d];
                    std::cout << "\n";
                }
                std::cout << "prunestats_end\n" << std::flush;
            }

        } else if (cmd == "stop") {
            // Signal the search to abort on its next out_of_time() poll.
            // Using a dedicated flag avoids clobbering g_time_limit_ms, which
            // was the bug: setting it to 0 caused the *next* search to exit
            // immediately before doing any work (depth-1 with a random move).
            g_stop = true;

        } else if (cmd == "quit") {
            break;

        } else if (cmd == "d") {
            // Debug: print board
            static const char PC[] = ".PNBRQKpnbrqk";
            for (int r = 7; r >= 0; r--) {
                for (int f = 0; f < 8; f++) {
                    Square sq = make_sq(f, r);
                    PieceType pt = engine.board.piece_on[sq];
                    if (pt == NO_PIECE) std::cout << ". ";
                    else {
                        Color c = engine.board.color_on[sq];
                        int idx = (c == WHITE) ? pt + 1 : pt + 7;
                        std::cout << PC[idx] << ' ';
                    }
                }
                std::cout << '\n';
            }
            std::cout << (engine.board.stm == WHITE ? "White" : "Black") << " to move\n";
            std::cout << "hash " << std::hex << engine.board.cur.hash
                      << " pawn_hash " << engine.board.cur.pawn_hash
                      << std::dec << "\n" << std::flush;
        }
    }
    return 0;
}
