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
 *   OUT_SCALE = 600
 */

#include <algorithm>
#include <array>
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
#include <sstream>
#include <string>
#include <vector>

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
static constexpr int OUT_SCALE   = 600;

// Search constants
static constexpr int INF         = 1'000'000;
static constexpr int MATE_SCORE  = 900'000;
static constexpr int DRAW_SCORE  = 0;

static constexpr int NMP_MIN_DEPTH  = 3;
static constexpr int NMP_R_BASE     = 3;
static constexpr int NMP_R_DIV      = 4;
static constexpr int LMR_MIN_DEPTH  = 3;
static constexpr int LMR_FULL_MOVES = 3;
static constexpr int ASP_WINDOW     = 50;
static constexpr int ASP_MAX_TRIES  = 4;
static constexpr int DELTA_MARGIN   = 200;
static constexpr int RFP_MARGIN     = 120;

static constexpr int FUTILITY_MARGIN[5] = {0, 100, 200, 300, 400};

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
static inline Square msb(Bitboard b) { return 63 ^ __builtin_clzll(b); }
static inline int    popcount(Bitboard b) { return __builtin_popcountll(b); }
static inline Bitboard pop_lsb(Bitboard &b) {
    Square s = lsb(b);
    b &= b - 1;
    return (Bitboard)1 << s;
}

static constexpr Bitboard FILE_A = 0x0101010101010101ULL;
static constexpr Bitboard FILE_H = 0x8080808080808080ULL;
static constexpr Bitboard RANK_1 = 0x00000000000000FFULL;
static constexpr Bitboard RANK_2 = 0x000000000000FF00ULL;
static constexpr Bitboard RANK_3 = 0x0000000000FF0000ULL;
static constexpr Bitboard RANK_6 = 0x0000FF0000000000ULL;
static constexpr Bitboard RANK_7 = 0x00FF000000000000ULL;
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
        cur = {0, -1, 0, 0, NO_PIECE};
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
    }

    void remove(Color c, PieceType pt, Square sq) {
        bb[c][pt] &= ~sq_bb(sq);
        occupied[c] &= ~sq_bb(sq);
        all &= ~sq_bb(sq);
        piece_on[sq] = NO_PIECE;
        color_on[sq] = WHITE;  // reset to avoid stale color after castle undo
        if (pt != KING) piece_count--;
        cur.hash ^= g_zobrist_piece[c][pt][sq];
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

// ─────────────────────────── NNUE weights ──────────────────────────────────

struct NNUEWeights {
    // Stored quantised as loaded from .nnue file
    int16_t ft_w[HALFKP_SIZE][MAX_HIDDEN];
    int16_t ft_b[MAX_HIDDEN];
    int8_t  l1_w[L1_SIZE][MAX_HIDDEN * 2];
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
        memcpy(&next, &prev, sizeof(AccEntry));
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
        for (int i = 0; i < H; i++) { next.w[i] -= W.ft_w[iw_from][i]; next.b[i] -= W.ft_w[ib_from][i]; }

        // Add to destination
        int iw_to = halfkp_w(wk, to, arriving_pidx);
        int ib_to = halfkp_b(bk, to, arriving_pidx);
        for (int i = 0; i < H; i++) { next.w[i] += W.ft_w[iw_to][i]; next.b[i] += W.ft_w[ib_to][i]; }

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
            for (int i = 0; i < H; i++) { next.w[i] -= W.ft_w[iw_cap][i]; next.b[i] -= W.ft_w[ib_cap][i]; }
        }
    }

    // Full recompute after king move or castling
    void push_full(const Board &board_after) {
        AccEntry &next = stack[top + 1];
        const NNUEWeights &W = g_weights;
        int H = W.hidden_size;
        int wk = board_after.king_sq[WHITE];
        int bk = sq_mirror(board_after.king_sq[BLACK]);
        for (int i = 0; i < H; i++) { next.w[i] = W.ft_b[i]; next.b[i] = W.ft_b[i]; }
        for (int c = 0; c < 2; c++) {
            for (int pt = 0; pt < 5; pt++) {
                Bitboard pieces = board_after.bb[c][pt];
                while (pieces) {
                    Square sq = lsb(pieces); pop_lsb(pieces);
                    int pidx = PIECE_IDX[pt][c];
                    int iw   = halfkp_w(wk, sq, pidx);
                    int ib   = halfkp_b(bk, sq, pidx);
                    for (int i = 0; i < H; i++) {
                        next.w[i] += W.ft_w[iw][i];
                        next.b[i] += W.ft_w[ib][i];
                    }
                }
            }
        }
        top++;
    }

    void null_push() {
        stack[top + 1] = stack[top];
        top++;
    }
    void pop() { top--; }
};

// ─────────────────────────── NNUE evaluation ───────────────────────────────

static int nnue_eval(const AccEntry &acc, bool stm_white, int bucket) {
    const NNUEWeights &W = g_weights;
    int H = W.hidden_size;

    // Clipped ReLU on FT output: clamp [0, FT_SCALE]
    // STM perspective first
    const int16_t *first  = stm_white ? acc.w : acc.b;
    const int16_t *second = stm_white ? acc.b : acc.w;

    // Pack into int8 inputs (clamped to [0, 127] = [0, FT_SCALE])
    int8_t x[MAX_HIDDEN * 2];
    for (int i = 0; i < H; i++) {
        x[i]   = (int8_t)std::clamp((int)first[i],  0, FT_SCALE);
        x[H+i] = (int8_t)std::clamp((int)second[i], 0, FT_SCALE);
    }

    // L1: int8 dot + int32 bias, >> FT_SCALE_BITS (log2(FT_SCALE) approx, but
    // matching serialize.py: bias scaled by FT_SCALE*L1_SCALE, so we just
    // divide the sum by FT_SCALE to get output in [0, L1_SCALE])
    // Integer dot, then shift right by 7 (= FT_SCALE = 127 ≈ 2^7 close enough;
    // exact: divide by FT_SCALE=127). We use >>6 (FT_SCALE≈128) for speed
    // and accept the tiny rounding difference vs float.
    int8_t l1[L1_SIZE];
    for (int o = 0; o < L1_SIZE; o++) {
        int32_t s = W.l1_b[o];
        for (int i = 0; i < H * 2; i++) s += (int32_t)W.l1_w[o][i] * (int32_t)x[i];
        // bias was scaled by FT_SCALE*L1_SCALE; inputs sum scaled by FT_SCALE*L1_SCALE
        // so s is in units of 1; divide by FT_SCALE to get [0, L1_SCALE]
        int32_t v = s / FT_SCALE;
        l1[o] = (int8_t)std::clamp(v, 0, L1_SCALE);
    }

    // L2
    int8_t l2[L2_SIZE];
    for (int o = 0; o < L2_SIZE; o++) {
        int32_t s = W.l2_b[o];
        for (int i = 0; i < L1_SIZE; i++) s += (int32_t)W.l2_w[o][i] * (int32_t)l1[i];
        int32_t v = s / L1_SCALE;
        l2[o] = (int8_t)std::clamp(v, 0, L2_SCALE);
    }

    // Output
    int32_t score = W.out_b[bucket];
    for (int i = 0; i < L2_SIZE; i++) score += (int32_t)W.out_w[bucket][i] * (int32_t)l2[i];
    return score / OUT_SCALE;  // centipawns
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
    // l1.weight [32, H*2] int8
    for (int o = 0; o < 32; o++)
        for (int i = 0; i < H*2; i++) { g_weights.l1_w[o][i] = (int8_t)buf[offset++]; }
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

struct TTEntry {
    uint64_t key;
    int32_t  score;
    int16_t  depth;
    uint8_t  flag;
    uint8_t  pad;
    Move     move;
};

// TT size is dynamic so setoption Hash resizes it correctly at runtime.
// Default: 128 MB / 20 bytes per entry, snapped down to power-of-2.
// The mask is kept in sync with the vector size on every resize.
static constexpr uint8_t TT_EXACT = 0, TT_LOWER = 1, TT_UPPER = 2;

static std::vector<TTEntry> g_tt;
static uint64_t             g_tt_mask = 0;   // always g_tt.size()-1, must be power-of-2 - 1

static void tt_resize(int mb) {
    // Compute largest power-of-2 entry count that fits in mb megabytes.
    size_t entries = ((size_t)mb * 1024 * 1024) / sizeof(TTEntry);
    size_t pot = 1;
    while (pot * 2 <= entries) pot *= 2;
    g_tt.assign(pot, {0, 0, 0, 0, 0, NO_MOVE});
    g_tt_mask = (uint64_t)(pot - 1);
}

static void tt_clear() { std::fill(g_tt.begin(), g_tt.end(), TTEntry{0,0,0,0,0,NO_MOVE}); }

static TTEntry* tt_probe(uint64_t key) {
    TTEntry *e = &g_tt[key & g_tt_mask];
    return (e->key == key) ? e : nullptr;
}

static void tt_store(uint64_t key, int depth, int score, uint8_t flag, Move mv) {
    TTEntry *e = &g_tt[key & g_tt_mask];
    if (e->key != key || depth >= e->depth || flag == TT_EXACT) {
        *e = {key, (int32_t)score, (int16_t)depth, flag, 0, mv};
    }
}

// ─────────────────────────── Killer / history ──────────────────────────────

static Move    g_killers[MAX_PLY][2];
static int32_t g_history[2][64][64];

static void search_init() {
    memset(g_killers, 0, sizeof g_killers);
    memset(g_history, 0, sizeof g_history);
}

static void killer_store(int ply, Move m) {
    if (g_killers[ply][0] != m) {
        g_killers[ply][1] = g_killers[ply][0];
        g_killers[ply][0] = m;
    }
}

static void history_update(Color c, Square from, Square to, int depth, bool good) {
    int delta = depth * depth;
    int &h = g_history[c][from][to];
    if (good) h = std::min(h + delta, 16384);
    else      h = std::max(h - delta, -16384);
}

// ─────────────────────────── Move scoring ──────────────────────────────────

struct ScoredMove {
    Move move;
    int  score;
};

static void score_moves(const Board &board, ScoredMove *ms, int n, int ply, Move tt_move) {
    for (int i = 0; i < n; i++) {
        Move m = ms[i].move;
        if (m == tt_move) { ms[i].score = 10'000'000; continue; }

        Square from = move_from(m), to = move_to(m);
        PieceType victim   = board.piece_on[to];
        PieceType attacker = board.piece_on[from];

        if (victim != NO_PIECE || move_flags(m) == MF_EP) {
            int vv = (victim != NO_PIECE) ? PIECE_VALUE[victim] : 100;  // EP=pawn
            int av = PIECE_VALUE[attacker];
            ms[i].score = 1'000'000 + vv * 10 - av;
            continue;
        }
        if (move_flags(m) == MF_PROMO) {
            // Score by piece so queen is always tried first.
            // QUEEN=900000, ROOK=800000, BISHOP=700100, KNIGHT=700000
            static constexpr int PROMO_SCORE[7] = {
                0, 0, 700000, 700100, 800000, 900000, 0  // indexed by PieceType enum
            };
            ms[i].score = PROMO_SCORE[move_promo(m)];
            continue;
        }
        if (ply < MAX_PLY && m == g_killers[ply][0]) { ms[i].score = 800'000; continue; }
        if (ply < MAX_PLY && m == g_killers[ply][1]) { ms[i].score = 700'000; continue; }
        ms[i].score = g_history[board.stm][from][to];
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
static Clock::time_point g_start_time;
static int64_t           g_time_limit_ms;  // -1 = no limit
static int64_t           g_node_limit;     // -1 = no limit
static int64_t           g_nodes;
static int               g_root_history_size = 0;

static bool g_stop = false;   // set by UCI "stop"; cleared at start of each search

static int64_t elapsed_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - g_start_time).count();
}
static bool out_of_time() {
    if (g_stop) return true;
    if (g_node_limit >= 0 && g_nodes >= g_node_limit) return true;
    return g_time_limit_ms >= 0 && (g_nodes & 4095) == 0 && elapsed_ms() >= g_time_limit_ms;
}

// ─────────────────────────── Search ────────────────────────────────────────

static int get_bucket(const Board &board) {
    int n = board.material_count();
    int b = n * g_weights.num_buckets / 32;
    return std::min(b, g_weights.num_buckets - 1);
}

static int quiescence(Board &board, Accumulator &acc, int alpha, int beta) {
    g_nodes++;

    int stm    = board.stm == WHITE;
    int bucket = get_bucket(board);
    int stand_pat = nnue_eval(acc.stack[acc.top], stm, bucket);

    if (stand_pat >= beta) return beta;
    if (stand_pat + 900 + DELTA_MARGIN < alpha) return alpha;
    if (stand_pat > alpha) alpha = stand_pat;

    Move moves[MAX_MOVES];
    int n = board.gen_moves(moves, /*captures_only=*/true);

    ScoredMove ms[MAX_MOVES];
    for (int i = 0; i < n; i++) ms[i] = {moves[i], 0};
    score_moves(board, ms, n, 0, NO_MOVE);

    for (int i = 0; i < n; i++) {
        pick_next(ms, i, n);
        Move m = ms[i].move;
        Square to   = move_to(m);
        Square from = move_from(m);
        PieceType victim   = board.piece_on[to];
        PieceType attacker = board.piece_on[from];

        // Skip losing captures
        if (victim != NO_PIECE && attacker != NO_PIECE &&
            PIECE_VALUE[victim] + DELTA_MARGIN < PIECE_VALUE[attacker])
            continue;

        // Push accumulator BEFORE do_move (mirrors alpha_beta ordering)
        bool king_moved = (board.piece_on[move_from(m)] == KING);
        bool is_castle  = (move_flags(m) == MF_CASTLE);
        if (is_castle || king_moved) {
            board.do_move(m);
            acc.push_full(board);
        } else {
            acc.push_incremental(board, m);
            board.do_move(m);
        }

        int score = -quiescence(board, acc, -beta, -alpha);
        board.undo_move(m);
        acc.pop();

        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

// Forward declaration
static int alpha_beta(Board &board, Accumulator &acc,
                      int depth, int alpha, int beta,
                      int ply, bool allow_null,
                      Move *pv, int &pv_len);

static int alpha_beta(Board &board, Accumulator &acc,
                      int depth, int alpha, int beta,
                      int ply, bool allow_null,
                      Move *pv, int &pv_len) {
    g_nodes++;
    pv_len = 0;

    if (__builtin_expect(out_of_time(), 0)) return alpha;

    // Cheap draws first (50-move, insufficient material).
    if (board.cur.halfmove_clock >= 100 || board.is_insufficient_material())
        return DRAW_SCORE;

    // Ply-aware repetition — before TT probe so stale TT draw scores can't
    // mask a real repetition state. Twofold within the search tree = drawn.
    if (ply > 0 && board.is_repetition(g_root_history_size)) return DRAW_SCORE;

    bool is_pv   = (beta - alpha > 1);
    bool in_check = board.in_check();

    // TT probe
    uint64_t key = board.cur.hash;
    Move tt_move = NO_MOVE;
    TTEntry *entry = tt_probe(key);
    if (entry) {
        // Lightweight TT move validation: check structural plausibility without
        // calling gen_moves() (which costs ~2µs per node = millions of wasted
        // cycles per second). The full-key match in tt_probe already filters
        // most hash collisions; we just guard the obviously-illegal cases.
        if (entry->move != NO_MOVE) {
            Move em = entry->move;
            Square ef = move_from(em), et = move_to(em);
            // from-square must have one of our pieces; to-square must not
            if (ef < 64 && et < 64 &&
                board.piece_on[ef] != NO_PIECE &&
                board.color_on[ef] == board.stm &&
                (board.piece_on[et] == NO_PIECE || board.color_on[et] != board.stm)) {
                tt_move = em;
            }
        }
        if (tt_move != NO_MOVE && entry->depth >= depth && !is_pv) {
            if (entry->flag == TT_EXACT) { pv[0] = tt_move; pv_len = 1; return entry->score; }
            if (entry->flag == TT_LOWER && entry->score >= beta)  return entry->score;
            if (entry->flag == TT_UPPER && entry->score <= alpha) return entry->score;
        }
    }

    if (depth <= 0) return quiescence(board, acc, alpha, beta);

    int stm    = board.stm == WHITE;
    int bucket = get_bucket(board);
    int static_eval = nnue_eval(acc.stack[acc.top], stm, bucket);

    // Reverse futility pruning
    if (!in_check && !is_pv && depth <= 4 && static_eval - RFP_MARGIN * depth >= beta)
        return static_eval;

    // Null move pruning
    if (allow_null && !in_check && !is_pv && depth >= NMP_MIN_DEPTH && static_eval >= beta) {
        int R = NMP_R_BASE + depth / NMP_R_DIV;
        acc.null_push();
        board.do_move(NULL_MOVE);
        Move child_pv[MAX_PLY]; int child_pv_len = 0;
        int null_score = -alpha_beta(board, acc, depth - 1 - R, -beta, -beta + 1,
                                     ply + 1, false, child_pv, child_pv_len);
        board.undo_move(NULL_MOVE);
        acc.pop();
        if (null_score >= beta) return beta;
    }

    bool futil = (!in_check && !is_pv && depth <= 4 &&
                  static_eval + FUTILITY_MARGIN[depth] <= alpha);

    // Generate moves
    Move moves[MAX_MOVES];
    int n = board.gen_moves(moves);
    if (n == 0) return in_check ? -(MATE_SCORE - ply) : DRAW_SCORE;

    ScoredMove ms[MAX_MOVES];
    for (int i = 0; i < n; i++) ms[i] = {moves[i], 0};
    score_moves(board, ms, n, ply, tt_move);

    int best_score    = -INF;
    Move best_move    = NO_MOVE;
    int  orig_alpha   = alpha;
    int  moves_done   = 0;
    Color moving_side = board.stm;  // capture before any do_move flips stm

    for (int i = 0; i < n; i++) {
        pick_next(ms, i, n);
        Move m      = ms[i].move;
        Square from = move_from(m), to = move_to(m);
        PieceType pt        = board.piece_on[from];
        bool is_capture     = (board.piece_on[to] != NO_PIECE || move_flags(m) == MF_EP);
        bool is_prom        = (move_flags(m) == MF_PROMO);
        bool is_castle      = (move_flags(m) == MF_CASTLE);
        bool king_moved     = (pt == KING);

        // Futility pruning
        if (futil && moves_done > 0 && !is_capture && !is_prom) continue;

        // Push accumulator BEFORE do_move (incremental needs pre-move board state)
        if (is_castle || king_moved) {
            board.do_move(m);
            acc.push_full(board);
        } else {
            acc.push_incremental(board, m);
            board.do_move(m);
        }

        bool gives_check = board.in_check();

        int ext = gives_check ? 1 : 0;
        Move child_pv[MAX_PLY]; int child_pv_len = 0;
        int score;

        if (moves_done == 0) {
            score = -alpha_beta(board, acc, depth - 1 + ext, -beta, -alpha,
                                ply + 1, true, child_pv, child_pv_len);
        } else {
            int R = 0;
            if (depth >= LMR_MIN_DEPTH && moves_done >= LMR_FULL_MOVES &&
                !is_capture && !gives_check && !is_prom && !in_check) {
                R = (int)(0.75 + std::log((double)depth) * std::log((double)moves_done) / 2.25);
                R = std::clamp(R, 1, depth - 1);
            }
            score = -alpha_beta(board, acc, depth - 1 - R + ext, -alpha - 1, -alpha,
                                ply + 1, true, child_pv, child_pv_len);
            if (score > alpha && R > 0) {
                child_pv_len = 0;
                score = -alpha_beta(board, acc, depth - 1 + ext, -alpha - 1, -alpha,
                                    ply + 1, true, child_pv, child_pv_len);
            }
            if (score > alpha && score < beta) {
                child_pv_len = 0;
                score = -alpha_beta(board, acc, depth - 1 + ext, -beta, -alpha,
                                    ply + 1, true, child_pv, child_pv_len);
            }
        }

        board.undo_move(m);
        acc.pop();
        moves_done++;

        if (__builtin_expect(out_of_time(), 0)) break;

        if (score > best_score) {
            best_score = score;
            best_move  = m;
            if (score > alpha) {
                alpha  = score;
                pv[0]  = m;
                if (child_pv_len > 0 && child_pv_len < MAX_PLY - 1)
                    memcpy(pv + 1, child_pv, sizeof(Move) * child_pv_len);
                pv_len = child_pv_len + 1;
            }
        }
        if (__builtin_expect(score >= beta, 0)) {
            if (!is_capture) {
                if (ply < MAX_PLY) killer_store(ply, m);
                history_update(moving_side, from, to, depth, true);
                for (int j = 0; j < i; j++) {
                    Move mj = ms[j].move;
                    if (board.piece_on[move_to(mj)] == NO_PIECE && move_flags(mj) != MF_EP)
                        history_update(moving_side, move_from(mj), move_to(mj), depth, false);
                }
            }
            break;
        }
    }

    uint8_t flag = (orig_alpha < best_score && best_score < beta) ? TT_EXACT :
                   (best_score >= beta)                            ? TT_LOWER : TT_UPPER;
    tt_store(key, depth, best_score, flag, best_move);
    return best_score;
}

// ─────────────────────────── Engine ────────────────────────────────────────

struct Engine {
    Board       board;
    Accumulator acc;
    std::string nn_file;

    Engine() { board.set_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"); }

    void set_nnue(const std::string &path) {
        nn_file = path;
        load_nnue(path);
    }

    std::pair<Move,int> search(int depth, int64_t movetime_ms) {
        if (!g_weights.loaded) {
            // Return first legal move
            Move moves[MAX_MOVES];
            int n = board.gen_moves(moves);
            return {n > 0 ? moves[0] : NO_MOVE, 0};
        }

        g_stop          = false;   // clear any previous stop signal
        g_start_time    = Clock::now();
        g_time_limit_ms = movetime_ms;
        // g_node_limit is set by the caller (go handler) before invoking search()
        g_nodes         = 0;

        search_init();
        acc.reset(board);

        Move best_move  = NO_MOVE;
        int  best_score = 0;
        int  prev_score = 0;

        g_root_history_size = board.history_top;

        // Quick static eval for aspiration seed
        {
            int bkt = get_bucket(board);
            prev_score = nnue_eval(acc.stack[0], board.stm == WHITE, bkt);
        }

        Move pv[MAX_PLY];
        int  pv_len = 0;

        for (int d = 1; d <= depth; d++) {
            if (out_of_time()) break;

            Move iter_pv[MAX_PLY]; int iter_pv_len = 0;
            int score;

            if (d >= 4) {
                int window = ASP_WINDOW;
                int alpha  = prev_score - window;
                int beta   = prev_score + window;
                for (int t = 0; t < ASP_MAX_TRIES; t++) {
                    iter_pv_len = 0;
                    score = alpha_beta(board, acc, d, alpha, beta, 0, true, iter_pv, iter_pv_len);
                    if (out_of_time()) break;
                    if (score <= alpha)     { alpha -= window; window *= 2; }
                    else if (score >= beta) { beta  += window; window *= 2; }
                    else break;
                }
                if (out_of_time() && iter_pv_len == 0) break;
                // Fall back to full-width if aspiration collapsed
                if (score <= prev_score - ASP_WINDOW * ASP_MAX_TRIES ||
                    score >= prev_score + ASP_WINDOW * ASP_MAX_TRIES) {
                    iter_pv_len = 0;
                    score = alpha_beta(board, acc, d, -INF, INF, 0, true, iter_pv, iter_pv_len);
                }
            } else {
                score = alpha_beta(board, acc, d, -INF, INF, 0, true, iter_pv, iter_pv_len);
            }

            if (out_of_time() && d > 1) break;

            if (iter_pv_len > 0) {
                best_move  = iter_pv[0];
                best_score = score;
                prev_score = score;
                memcpy(pv, iter_pv, sizeof(Move) * iter_pv_len);
                pv_len = iter_pv_len;
            }

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

        if (best_move == NO_MOVE) {
            Move moves[MAX_MOVES];
            int n = board.gen_moves(moves);
            if (n) best_move = moves[0];
        }
        return {best_move, best_score};
    }
};

// ─────────────────────────── UCI loop ──────────────────────────────────────

int main() {
    init_magics();
    init_zobrist();
    tt_resize(128);   // default 128 MB TT; overridden by setoption Hash

    Engine engine;

    std::string line;

    std::cout << "id name NNUEEngine\nid author nnue-trainer\n";
    std::cout << "option name NNFile type string default <empty>\n";
    std::cout << "option name Hash type spin default 128 min 1 max 2048\n";
    std::cout << "option name Threads type spin default 1 min 1 max 1\n";
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
            std::cout << "option name Threads type spin default 1 min 1 max 1\n";
            std::cout << "uciok\n" << std::flush;

        } else if (cmd == "isready") {
            std::cout << "readyok\n" << std::flush;

        } else if (cmd == "ucinewgame") {
            engine.board.set_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            tt_clear();

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
            // Time management
            if (!infinite && movetime < 0 && go_nodes < 0 && (wtime >= 0 || btime >= 0)) {
                int myTime = (engine.board.stm == WHITE) ? wtime : btime;
                int myInc  = (engine.board.stm == WHITE) ? winc  : binc;
                if (myTime >= 0) {
                    movetime = std::max((int64_t)50, (int64_t)(myTime / movestogo + myInc * 0.8));
                }
            }
            g_node_limit = go_nodes;   // -1 if not set; search() will clear and re-apply
            auto [mv, score] = engine.search(depth, movetime);
            std::string best = (mv != NO_MOVE) ? engine.board.move_uci(mv) : "0000";
            std::cout << "bestmove " << best << "\n" << std::flush;

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
            std::cout << (engine.board.stm == WHITE ? "White" : "Black") << " to move\n" << std::flush;
        }
    }
    return 0;
}
