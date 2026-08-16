/*
 * movepick.h — Staged move generation, multi-tier history, and SEE.
 *
 * Major changes vs original:
 *
 *   1. Staged move generation (state machine)
 *      MovePicker drives a stage enum: MAIN_TT → CAPTURE_INIT → GOOD_CAPTURE
 *      → REFUTATION → QUIET_INIT → GOOD_QUIET → BAD_CAPTURE → BAD_QUIET.
 *      Quiet moves are never generated if a TT move or good capture causes a
 *      beta cut-off — identical to Stockfish's approach.
 *
 *   2. Static Exchange Evaluation (SEE)
 *      see_ge(pos, move, threshold) determines whether a capture wins or loses
 *      material.  Captures that fail SEE are deferred to BAD_CAPTURE.
 *
 *   3. Multi-tier history tables
 *      • ButterflyHistory   g_butterfly[color][from][to]
 *      • ContinuationHistory g_cont_hist[piece][to][prev_piece][prev_to]
 *        — populated with the 1-ply and 2-ply previous moves.
 *      • CapturePieceToHistory g_capture_hist[piece][to][captured_type]
 *
 *   4. Stockfish-style gravity update formula
 *        h += bonus - h * |bonus| / MAX_HISTORY
 *      instead of simple clamped addition.
 *
 *   5. Partial insertion sort (instead of full selection sort) applied only to
 *      the subset of moves that will actually be visited.
 *
 * History tables and killers are reset by search_init() each new search.
 *
 * Dependencies: types.h, position.h
 */

#pragma once
#include "types.h"
#include "position.h"

#include <algorithm>
#include <cstring>
#include <cmath>

// ─────────────────────────── search constants (shared with search.h) ────────

inline constexpr int NMP_MIN_DEPTH  = 3;
inline constexpr int NMP_R_BASE     = 3;
inline constexpr int NMP_R_DIV      = 4;
inline constexpr int LMR_MIN_DEPTH  = 3;
inline constexpr int LMR_FULL_MOVES = 3;
inline constexpr int ASP_WINDOW     = 50;
inline constexpr int ASP_MAX_TRIES  = 4;
inline constexpr int DELTA_MARGIN   = 200;
inline constexpr int RFP_MARGIN     = 120;

// ProbCut constants
inline constexpr int PROBCUT_MIN_DEPTH = 5;
inline constexpr int PROBCUT_MARGIN    = 200;

// Singular extension constants
inline constexpr int SE_MIN_DEPTH  = 8;    // minimum depth to check singularity
inline constexpr int SE_DEPTH_DIV  = 2;    // singular search depth = (depth-1)/2

inline constexpr int FUTILITY_MARGIN[5] = {0, 100, 200, 300, 400};

// ─────────────────────────── scored move ────────────────────────────────────

struct ScoredMove {
    Move move;
    int  score;
};

// ─────────────────────────── history tables ─────────────────────────────────

// ButterflyHistory: indexed [color][from][to].
// Range ±MAX_HISTORY (16384).
inline int32_t g_butterfly[2][64][64];

// ContinuationHistory: [moving_piece_type][to_sq][prev_piece_type][prev_to_sq].
// Tracks pairs of consecutive moves; used at 1-ply and 2-ply lookback.
// KING (5) is included so the array is contiguous; its entries are simply
// never updated for king moves (they're noise in practice).
inline int16_t g_cont_hist[6][64][6][64];

// CapturePieceToHistory: [attacker_piece_type][to_sq][captured_piece_type].
// Rewards/penalises captures based on outcome.
// captured type index: PAWN=0..QUEEN=4 (KING never captured).
inline int16_t g_capture_hist[6][64][5];

// Two killers per ply (quiet moves that caused beta cut-offs).
inline Move g_killers[MAX_PLY][2];

// ── Gravity update helper ────────────────────────────────────────────────────
// Stockfish formula: h += bonus - h * |bonus| / MAX_HISTORY
// Keeps values in [-MAX_HISTORY, +MAX_HISTORY] without clamping.
template <typename T>
inline void gravity_update(T &h, int bonus) {
    h += static_cast<T>(bonus - static_cast<int>(h) * std::abs(bonus) / MAX_HISTORY);
}

// ── Reset all tables at search start ────────────────────────────────────────
inline void search_init() {
    memset(g_killers,      0, sizeof g_killers);
    memset(g_butterfly,    0, sizeof g_butterfly);
    memset(g_cont_hist,    0, sizeof g_cont_hist);
    memset(g_capture_hist, 0, sizeof g_capture_hist);
}

// ── Killer store ─────────────────────────────────────────────────────────────
inline void killer_store(int ply, Move m) {
    if (ply >= MAX_PLY) return;
    if (g_killers[ply][0] != m) {
        g_killers[ply][1] = g_killers[ply][0];
        g_killers[ply][0] = m;
    }
}

// ── History update (butterfly + continuation) for quiet moves ────────────────
// prev1 / prev2: the move played 1 and 2 plies ago (NO_MOVE if not applicable).
inline void history_update_quiet(Color c, Square from, Square to,
                                 PieceType pt,
                                 Move prev1, const Position &pos_before_prev1,
                                 Move prev2, const Position &pos_before_prev2,
                                 int depth, bool good)
{
    int bonus = good ? depth * depth : -(depth * depth);

    // Butterfly
    gravity_update(g_butterfly[c][from][to], bonus);

    // ContinuationHistory at 1-ply lookback
    if (prev1 != NO_MOVE && prev1 != NULL_MOVE) {
        PieceType ppt1 = pos_before_prev1.piece_on[move_from(prev1)];
        Square    pto1 = move_to(prev1);
        if (ppt1 < KING)
            gravity_update(g_cont_hist[pt][to][ppt1][pto1], bonus);
    }
    // ContinuationHistory at 2-ply lookback
    if (prev2 != NO_MOVE && prev2 != NULL_MOVE) {
        PieceType ppt2 = pos_before_prev2.piece_on[move_from(prev2)];
        Square    pto2 = move_to(prev2);
        if (ppt2 < KING)
            gravity_update(g_cont_hist[pt][to][ppt2][pto2], bonus);
    }
}

// Simplified butterfly-only update used when continuation context is unavailable.
inline void history_update(Color c, Square from, Square to, int depth, bool good) {
    int bonus = good ? depth * depth : -(depth * depth);
    gravity_update(g_butterfly[c][from][to], bonus);
}

// ── Capture history update ────────────────────────────────────────────────────
inline void capture_hist_update(PieceType attacker, Square to,
                                PieceType captured, int depth, bool good) {
    if (captured >= KING) return;   // safety guard
    int bonus = good ? depth * depth : -(depth * depth);
    gravity_update(g_capture_hist[attacker][to][captured], bonus);
}

// ─────────────────────────── Static Exchange Evaluation ─────────────────────
//
// Returns true if the sequence of captures on `to` starting with `m` yields
// a gain >= `threshold` for the moving side.
//
// Uses an approximation: iterates through all attackers in value order,
// alternating sides, and stops when either side's gain is determined.
// This mirrors Stockfish's see_ge() logic.

inline bool see_ge(const Position &pos, Move m, int threshold = 0) {
    // Only meaningful for captures and en-passant
    Square from  = move_from(m);
    Square to    = move_to(m);
    MoveFlag fl  = move_flags(m);

    int value = 0;
    if (fl == MF_EP) {
        value = SEE_VALUE[PAWN] - threshold;
    } else {
        PieceType cap = pos.piece_on[to];
        if (cap == NO_PIECE) return (0 >= threshold);  // quiet move
        value = SEE_VALUE[cap] - threshold;
    }

    // If taking the piece already does not cover the threshold, no need to
    // simulate further — bail early.
    if (value < 0) return false;

    PieceType next_pt = pos.piece_on[from];
    value -= SEE_VALUE[next_pt];

    // If even after giving our piece away we are above threshold, it is safe.
    if (value >= 0) return true;

    // Simulate the exchange sequence.
    Bitboard occ        = pos.all ^ sq_bb(from) ^ sq_bb(to);
    Color    stm        = static_cast<Color>(!pos.stm); // opponent moves next
    Bitboard diag       = pos.bb[WHITE][BISHOP] | pos.bb[WHITE][QUEEN]
                        | pos.bb[BLACK][BISHOP] | pos.bb[BLACK][QUEEN];
    Bitboard orth       = pos.bb[WHITE][ROOK]   | pos.bb[WHITE][QUEEN]
                        | pos.bb[BLACK][ROOK]   | pos.bb[BLACK][QUEEN];

    // Attackers to the target square (excluding already moved piece)
    auto attackers_to = [&](Bitboard occ2) -> Bitboard {
        return (g_pawn_attacks[WHITE][to]  & pos.bb[BLACK][PAWN])
             | (g_pawn_attacks[BLACK][to]  & pos.bb[WHITE][PAWN])
             | (g_knight_attacks[to]       & (pos.bb[WHITE][KNIGHT] | pos.bb[BLACK][KNIGHT]))
             | (bishop_attacks(to, occ2)   & diag)
             | (rook_attacks(to, occ2)     & orth)
             | (g_king_attacks[to]         & (pos.bb[WHITE][KING]   | pos.bb[BLACK][KING]));
    };

    Bitboard att = attackers_to(occ);

    while (true) {
        // Find least-valuable attacker for the side to move
        Bitboard stm_att = att & pos.occupied[stm];
        if (!stm_att) break;   // no more attackers — current side loses

        // Find least valuable piece type among stm_att
        PieceType lva_pt = NO_PIECE;
        Bitboard  lva_bb = 0;
        for (int pt = PAWN; pt <= KING; ++pt) {
            Bitboard cand = stm_att & pos.bb[stm][pt];
            if (cand) { lva_pt = static_cast<PieceType>(pt); lva_bb = cand; break; }
        }
        if (lva_pt == NO_PIECE) break;

        // Remove that piece from the occupancy
        occ ^= sq_bb(lsb(lva_bb));
        // Update diagonal / orthogonal sliders after removal
        att = attackers_to(occ);

        // Negamax the value
        value = -value - 1 - SEE_VALUE[lva_pt];
        stm   = static_cast<Color>(!stm);

        if (value >= 0) {
            // Current side wins if it still has the king available but opponent
            // king is needed to recapture — guard against illegal king captures.
            if (lva_pt == KING && (att & pos.occupied[stm]))
                stm = static_cast<Color>(!stm);
            break;
        }
    }

    // The side that was originally moving wins if `stm` is now the opponent.
    return stm != pos.stm;
}

// ─────────────────────────── continuation history lookup ────────────────────

// Returns the combined continuation history score for a move (pt, to) given
// the 1-ply and 2-ply previous moves.  Safe to call with NO_MOVE.
inline int cont_hist_score(PieceType pt, Square to,
                           Move prev1, const Position *pos1,
                           Move prev2, const Position *pos2) {
    int score = 0;
    if (prev1 != NO_MOVE && prev1 != NULL_MOVE && pos1) {
        PieceType p1 = pos1->piece_on[move_from(prev1)];
        Square    t1 = move_to(prev1);
        if (p1 < KING) score += g_cont_hist[pt][to][p1][t1];
    }
    if (prev2 != NO_MOVE && prev2 != NULL_MOVE && pos2) {
        PieceType p2 = pos2->piece_on[move_from(prev2)];
        Square    t2 = move_to(prev2);
        if (p2 < KING) score += g_cont_hist[pt][to][p2][t2];
    }
    return score;
}

// ─────────────────────────── move scoring ───────────────────────────────────
//
// Score bands (descending priority):
//   TT move          : 10,000,000
//   Good captures    : 1,000,000 + MVV-LVA  (SEE >= 0)
//   Promotions       : 900,000 (queen) … 700,000 (bishop)
//   Killers          : 800,000 / 700,000
//   Counter-move     : 600,000
//   Quiet history    : butterfly ± cont_hist  (can be negative)
//   Bad captures     : –1,000,000 + MVV-LVA  (SEE < 0, scored separately)

inline void score_moves(const Position &pos, ScoredMove *ms, int n,
                        int ply, Move tt_move,
                        Move prev1 = NO_MOVE, const Position *pos1 = nullptr,
                        Move prev2 = NO_MOVE, const Position *pos2 = nullptr)
{
    for (int i = 0; i < n; i++) {
        Move m = ms[i].move;

        if (m == tt_move) { ms[i].score = 10'000'000; continue; }

        Square    from     = move_from(m);
        Square    to       = move_to(m);
        PieceType attacker = pos.piece_on[from];
        PieceType victim   = pos.piece_on[to];
        MoveFlag  flags    = move_flags(m);

        // ── Captures ──────────────────────────────────────────────────────────
        if (victim != NO_PIECE || flags == MF_EP) {
            int vv  = (victim != NO_PIECE) ? SEE_VALUE[victim] : SEE_VALUE[PAWN];
            int av  = SEE_VALUE[attacker];
            int mvvlva = vv * 10 - av;
            // Separate good and bad captures using SEE.
            if (see_ge(pos, m, 0)) {
                ms[i].score = 1'000'000 + mvvlva
                            + (victim != NO_PIECE
                               ? g_capture_hist[attacker][to][victim < KING ? victim : QUEEN]
                               : 0);
            } else {
                // Bad capture — placed behind quiets; scored so they sort among themselves.
                ms[i].score = -1'000'000 + mvvlva;
            }
            continue;
        }

        // ── Promotions ────────────────────────────────────────────────────────
        if (flags == MF_PROMO) {
            static constexpr int PROMO_SCORE[7] = {
                0, 0,
                700'000,   // BISHOP
                700'100,   // KNIGHT
                800'000,   // ROOK
                900'000,   // QUEEN
                0
            };
            ms[i].score = PROMO_SCORE[move_promo(m)];
            continue;
        }

        // ── Killers ───────────────────────────────────────────────────────────
        if (ply < MAX_PLY) {
            if (m == g_killers[ply][0]) { ms[i].score = 800'000; continue; }
            if (m == g_killers[ply][1]) { ms[i].score = 700'000; continue; }
        }

        // ── Quiet: butterfly + continuation history ───────────────────────────
        int hist = g_butterfly[pos.stm][from][to]
                 + cont_hist_score(attacker, to, prev1, pos1, prev2, pos2);
        ms[i].score = hist;
    }
}

// ─────────────────────────── partial insertion sort ─────────────────────────
//
// Brings the best move in [idx, n) into position idx.
// (single selection-sort step — no full sort of the array)

inline void pick_next(ScoredMove *ms, int idx, int n) {
    int best = idx;
    for (int i = idx + 1; i < n; i++)
        if (ms[i].score > ms[best].score) best = i;
    if (best != idx) std::swap(ms[idx], ms[best]);
}

// ─────────────────────────── MovePicker — staged state machine ───────────────
//
// Usage:
//
//   MovePicker mp(pos, tt_move, ply, captures_only,
//                 prev_move1, pos_at_prev1, prev_move2, pos_at_prev2);
//   while (Move m = mp.next()) { ... }
//
// Stages (full search):
//   STAGE_TT          — yield tt_move if legal
//   STAGE_CAP_INIT    — generate + score captures
//   STAGE_GOOD_CAP    — yield captures that pass SEE >= 0
//   STAGE_REFUTATION  — killers
//   STAGE_QUIET_INIT  — generate + score quiets (deferred!)
//   STAGE_GOOD_QUIET  — yield quiets in history order
//   STAGE_BAD_CAP     — yield deferred losing captures
//
// For captures_only (qsearch):
//   STAGE_TT → STAGE_CAP_INIT → STAGE_GOOD_CAP only.

class MovePicker {
public:
    enum Stage {
        STAGE_TT = 0,
        STAGE_CAP_INIT,
        STAGE_GOOD_CAP,
        STAGE_REFUTATION,
        STAGE_QUIET_INIT,
        STAGE_GOOD_QUIET,
        STAGE_BAD_CAP,
        STAGE_DONE
    };

    MovePicker(Position &pos, Move tt_move, int ply,
               bool captures_only = false,
               Move prev1 = NO_MOVE, const Position *pos1 = nullptr,
               Move prev2 = NO_MOVE, const Position *pos2 = nullptr)
        : pos_(pos), tt_move_(tt_move), ply_(ply),
          captures_only_(captures_only),
          prev1_(prev1), pos1_(pos1),
          prev2_(prev2), pos2_(pos2),
          stage_(STAGE_TT),
          n_cap_(0), idx_cap_(0),
          n_bad_(0), idx_bad_(0),
          n_quiet_(0), idx_quiet_(0)
    {}

    // Returns the next move in priority order, or NO_MOVE when exhausted.
    Move next() {
        while (true) {
            switch (stage_) {

            // ── TT move ────────────────────────────────────────────────────────
            case STAGE_TT:
                stage_ = STAGE_CAP_INIT;
                if (tt_move_ != NO_MOVE && is_pseudo_legal(tt_move_))
                    return tt_move_;
                break;

            // ── Generate & score captures ──────────────────────────────────────
            case STAGE_CAP_INIT: {
                Move buf[MAX_MOVES];
                n_cap_ = pos_.gen_moves(buf, /*captures_only=*/true);
                int j = 0;
                for (int i = 0; i < n_cap_; i++) {
                    if (buf[i] == tt_move_) continue;  // already tried
                    caps_[j++] = {buf[i], 0};
                }
                n_cap_ = j;
                score_moves(pos_, caps_, n_cap_, ply_, NO_MOVE,
                            prev1_, pos1_, prev2_, pos2_);
                idx_cap_ = 0;
                n_bad_   = 0;
                stage_   = STAGE_GOOD_CAP;
                break;
            }

            // ── Good captures (SEE >= 0) ───────────────────────────────────────
            case STAGE_GOOD_CAP:
                while (idx_cap_ < n_cap_) {
                    pick_next(caps_, idx_cap_, n_cap_);
                    ScoredMove &sm = caps_[idx_cap_];
                    if (sm.score >= 1'000'000) {
                        // Already classified as good (score >= 1M threshold set in score_moves)
                        return caps_[idx_cap_++].move;
                    } else {
                        // Bad capture — stash for later
                        bad_caps_[n_bad_++] = sm;
                        idx_cap_++;
                    }
                }
                if (captures_only_) { stage_ = STAGE_DONE; break; }
                stage_ = STAGE_REFUTATION;
                break;

            // ── Killers ────────────────────────────────────────────────────────
            // Fill killer buffers once, then yield them one at a time via
            // killer_idx_.  stage_ stays at STAGE_REFUTATION until both are
            // exhausted, so STAGE_QUIET_INIT is only ever entered for quiet
            // move generation — never for killer yielding.
            case STAGE_REFUTATION:
                if (killer_idx_ == 0 && ply_ < MAX_PLY) {
                    for (int k = 0; k < 2; k++) {
                        Move km = g_killers[ply_][k];
                        if (km != NO_MOVE && km != tt_move_ && is_pseudo_legal(km)
                            && pos_.piece_on[move_to(km)] == NO_PIECE
                            && move_flags(km) != MF_EP)
                            killer_buf_[k] = km;
                        else
                            killer_buf_[k] = NO_MOVE;
                    }
                }
                while (killer_idx_ < 2) {
                    Move km = killer_buf_[killer_idx_++];
                    if (km != NO_MOVE) return km;  // stage_ stays STAGE_REFUTATION
                }
                stage_ = STAGE_QUIET_INIT;
                break;

            // ── Generate & score quiets ────────────────────────────────────────
            // This stage is entered exactly once per node — killers are fully
            // handled by STAGE_REFUTATION above.
            case STAGE_QUIET_INIT: {
                Move all_moves[MAX_MOVES];
                int  all_n = pos_.gen_moves(all_moves, /*captures_only=*/false);
                int  j     = 0;
                for (int i = 0; i < all_n; i++) {
                    Move mv = all_moves[i];
                    if (mv == tt_move_) continue;
                    if (pos_.piece_on[move_to(mv)] != NO_PIECE) continue;  // skip captures
                    if (move_flags(mv) == MF_EP)                continue;
                    if (ply_ < MAX_PLY
                        && (mv == g_killers[ply_][0] || mv == g_killers[ply_][1]))
                        continue;  // already yielded as killers
                    quiet_[j++] = {mv, 0};
                }
                n_quiet_ = j;
                score_moves(pos_, quiet_, n_quiet_, ply_, NO_MOVE,
                            prev1_, pos1_, prev2_, pos2_);
                idx_quiet_ = 0;
                stage_     = STAGE_GOOD_QUIET;
                break;
            }

            // ── Good quiets ────────────────────────────────────────────────────
            case STAGE_GOOD_QUIET:
                if (idx_quiet_ < n_quiet_) {
                    pick_next(quiet_, idx_quiet_, n_quiet_);
                    return quiet_[idx_quiet_++].move;
                }
                stage_ = STAGE_BAD_CAP;
                idx_bad_ = 0;
                break;

            // ── Bad captures ───────────────────────────────────────────────────
            case STAGE_BAD_CAP:
                if (idx_bad_ < n_bad_)
                    return bad_caps_[idx_bad_++].move;
                stage_ = STAGE_DONE;
                break;

            case STAGE_DONE:
            default:
                return NO_MOVE;
            }
        }
    }

private:
    // Quick pseudo-legality check to validate TT / killer moves without full gen.
    bool is_pseudo_legal(Move m) const {
        if (m == NO_MOVE || m == NULL_MOVE) return false;
        Square from = move_from(m);
        Square to   = move_to(m);
        if (from < 0 || from > 63 || to < 0 || to > 63) return false;
        if (pos_.piece_on[from] == NO_PIECE)    return false;
        if (pos_.color_on[from] != pos_.stm)    return false;
        if (pos_.piece_on[to] != NO_PIECE
            && pos_.color_on[to] == pos_.stm)   return false;
        return true;
    }

    Position       &pos_;
    Move            tt_move_;
    int             ply_;
    bool            captures_only_;
    Move            prev1_;
    const Position *pos1_;
    Move            prev2_;
    const Position *pos2_;

    Stage      stage_;

    ScoredMove caps_[MAX_MOVES];
    int        n_cap_, idx_cap_;

    ScoredMove bad_caps_[MAX_MOVES];
    int        n_bad_, idx_bad_;

    ScoredMove quiet_[MAX_MOVES];
    int        n_quiet_, idx_quiet_;

    Move killer_buf_[2] = {NO_MOVE, NO_MOVE};
    int  killer_idx_    = 0;
};
