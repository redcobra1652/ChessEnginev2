/*
 * types.h — Primitive types, constants, move encoding, and bit utilities.
 *
 * All other engine headers include this file first. Nothing here depends on
 * the rest of the engine, so it compiles in isolation.
 *
 * Move encoding (32-bit):
 *   bits  5: 0  — from square (0-63)
 *   bits 11: 6  — to square   (0-63)
 *   bits 15:12  — promotion piece type (PieceType enum; NO_PIECE if not a promo)
 *   bits 19:16  — move flags  (MF_NORMAL / MF_CASTLE / MF_EP / MF_PROMO)
 *
 * Quantisation constants (must match serialize.py exactly):
 *   FT activations clamped [0, 127] = FT_SCALE
 *   L1 / L2 activations clamped [0, 64] = L1_SCALE / L2_SCALE
 *   Output divided by OUT_SCALE → centipawns
 *
 * Changes vs original:
 *   • Added Value alias (int) for clarity in history / SEE signatures.
 *   • Added MAX_HISTORY — clamp for Stockfish-style gravity formula.
 *   • Added N_PIECE_TYPES_CONT for ContinuationHistory indexing (6 types).
 *   • Added SEE piece values array SEE_VALUE[].
 */

#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>

// ─────────────────────────── compile-time limits ───────────────────────────

inline constexpr int HALFKP_SIZE   = 40960;  // 64 king-sqs * 64 piece-sqs * 10 piece-types
inline constexpr int MAX_HIDDEN    = 512;    // ft output width (runtime H ≤ this)
inline constexpr int L1_SIZE       = 32;
inline constexpr int L2_SIZE       = 32;
inline constexpr int MAX_BUCKETS   = 32;
inline constexpr int MAX_PLY       = 128;
inline constexpr int MAX_MOVES     = 256;
inline constexpr int N_SQUARES     = 64;
inline constexpr int N_PIECE_TYPES = 10;     // 5 non-king types * 2 colors

// ─────────────────────────── NNUE fixed-point ──────────────────────────────

inline constexpr int FT_SCALE  = 127;
inline constexpr int L1_SCALE  = 64;
inline constexpr int L2_SCALE  = 64;
inline constexpr int OUT_SCALE = 600;

// ─────────────────────────── search constants ──────────────────────────────

inline constexpr int INF        = 1'000'000;
inline constexpr int MATE_SCORE = 900'000;
inline constexpr int DRAW_SCORE = 0;

// Piece values (centipawns) — PAWN=0 .. KING=5
inline constexpr int PIECE_VALUE[6] = {100, 320, 330, 500, 900, 20000};

// SEE piece values — slightly simplified for exchange evaluation.
// King value is set large so it is never traded away in SEE.
inline constexpr int SEE_VALUE[7] = {100, 320, 330, 500, 900, 20000, 0}; // NO_PIECE→0

// ─────────────────────────── history constants ─────────────────────────────

// Maximum absolute value stored in any history table.
// The Stockfish-style gravity formula clamps to this range:
//   h += bonus - h * |bonus| / MAX_HISTORY
inline constexpr int MAX_HISTORY = 16384;

// ─────────────────────────── primitive types ───────────────────────────────

using Bitboard = uint64_t;
using Square   = int;      // 0-63
using Move     = uint32_t; // encoded as described above
using Value    = int;      // generic search value / history score

// ─────────────────────────── enumerations ──────────────────────────────────

enum Color     : int { WHITE = 0, BLACK = 1 };
enum PieceType : int { PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5, NO_PIECE=6 };

// Move flags stored in bits 19:16
enum MoveFlag : uint32_t {
    MF_NORMAL = 0,
    MF_CASTLE = 1,
    MF_EP     = 2,
    MF_PROMO  = 3,
};

// Transposition-table bound types
enum TTFlag : uint8_t { TT_EXACT = 0, TT_LOWER = 1, TT_UPPER = 2 };

// ─────────────────────────── move encoding helpers ─────────────────────────

inline Square    move_from(Move m)     { return static_cast<Square>(m & 63); }
inline Square    move_to(Move m)       { return static_cast<Square>((m >> 6) & 63); }
inline PieceType move_promo(Move m)    { return static_cast<PieceType>((m >> 12) & 15); }
inline MoveFlag  move_flags(Move m)    { return static_cast<MoveFlag>((m >> 16) & 15); }

inline Move make_move(Square f, Square t,
                      PieceType promo = NO_PIECE,
                      MoveFlag  flags = MF_NORMAL)
{
    return static_cast<uint32_t>(f)
         | (static_cast<uint32_t>(t)     << 6)
         | (static_cast<uint32_t>(promo) << 12)
         | (static_cast<uint32_t>(flags) << 16);
}

inline constexpr Move NULL_MOVE = 0xFFFFFFFF;
inline constexpr Move NO_MOVE   = 0;

// ─────────────────────────── board geometry ────────────────────────────────

inline constexpr Bitboard FILE_A = 0x0101010101010101ULL;
inline constexpr Bitboard FILE_H = 0x8080808080808080ULL;
inline constexpr Bitboard RANK_1 = 0x00000000000000FFULL;
inline constexpr Bitboard RANK_2 = 0x000000000000FF00ULL;
inline constexpr Bitboard RANK_3 = 0x0000000000FF0000ULL;
inline constexpr Bitboard RANK_6 = 0x0000FF0000000000ULL;
inline constexpr Bitboard RANK_7 = 0x00FF000000000000ULL;
inline constexpr Bitboard RANK_8 = 0xFF00000000000000ULL;

inline Bitboard sq_bb(Square s)    { return Bitboard(1) << s; }
inline int      sq_file(Square s)  { return s & 7; }
inline int      sq_rank(Square s)  { return s >> 3; }
inline Square   sq_mirror(Square s){ return s ^ 56; }        // flip rank for black POV
inline Square   make_sq(int f, int r){ return r * 8 + f; }

// ─────────────────────────── bit utilities ─────────────────────────────────

inline Square lsb(Bitboard b) {
    assert(b);
    return static_cast<Square>(__builtin_ctzll(b));
}
inline Square msb(Bitboard b) {
    assert(b);
    return static_cast<Square>(63 ^ __builtin_clzll(b));
}
inline int popcount(Bitboard b) { return __builtin_popcountll(b); }

// Returns the LSB square and clears it in-place
inline Square pop_lsb(Bitboard &b) {
    Square s = lsb(b);
    b &= b - 1;
    return s;
}

// ─────────────────────────── HalfKP index helpers ──────────────────────────
// Must match serialize.py PIECE_TYPE_IDX and halfkp_w / halfkp_b exactly.
//
// Feature index layout: king_sq * (64 * 10) + piece_sq * 10 + piece_color_type
// PIECE_IDX[pt][color]: maps (piece-type, color) → the 0-9 type index

inline constexpr int PIECE_IDX[6][2] = {
    //  WHITE   BLACK
    {  0,  1 },  // PAWN
    {  2,  3 },  // KNIGHT
    {  4,  5 },  // BISHOP
    {  6,  7 },  // ROOK
    {  8,  9 },  // QUEEN
    { -1, -1 },  // KING — not a HalfKP feature
};

// White king perspective: features from white's board orientation
inline int halfkp_w(int king_sq, int piece_sq, int pidx) {
    return king_sq * (N_SQUARES * N_PIECE_TYPES) + piece_sq * N_PIECE_TYPES + pidx;
}

// Black king perspective: mirror rank of piece_sq, flip color bit of pidx
inline int halfkp_b(int king_sq_mirror, int piece_sq, int pidx) {
    return king_sq_mirror * (N_SQUARES * N_PIECE_TYPES)
         + sq_mirror(piece_sq) * N_PIECE_TYPES
         + (pidx ^ 1);  // flip color: 0↔1, 2↔3, 4↔5, 6↔7, 8↔9
}

// ─────────────────────────── UCI move strings ──────────────────────────────

inline std::string sq_to_str(Square s) {
    std::string r;
    r += static_cast<char>('a' + sq_file(s));
    r += static_cast<char>('1' + sq_rank(s));
    return r;
}
