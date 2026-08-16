/*
 * search.h — Negamax PVS search with iterative deepening.
 *
 * Algorithms implemented (new additions marked with ★):
 *   • Iterative deepening with aspiration windows
 *   • Principal variation search (PVS / null-window re-search)
 *   • Null move pruning
 *   • ★ ProbCut — shallow re-search with raised beta to prune subtrees early
 *   • ★ Singular Extensions — extend the TT move when it is uniquely best
 *   • Late-move reductions (LMR) with ★ dynamic adjustment via:
 *       – continuation history score of the move
 *       – whether the node is improving (static_eval > ss[-2].static_eval)
 *       – PV vs non-PV node
 *       – whether the move gives check
 *   • ★ Improving flag — tracks whether static eval is rising over two plies
 *   • ★ Correction history — adjusts raw NNUE eval to reduce systematic bias
 *       (pawn-structure correction table, indexed by pawn Zobrist key)
 *   • Reverse futility pruning
 *   • Futility pruning
 *   • Check extension
 *   • Multi-tier history updates (butterfly + continuation + capture)
 *   • Quiescence search with stand-pat and delta pruning
 *   • Transposition table (depth-preferred replacement)
 *   • Ply-aware repetition detection
 *
 * Dependencies: types.h, position.h, movepick.h, evaluate.h
 */

#pragma once
#include "types.h"
#include "position.h"
#include "movepick.h"
#include "evaluate.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// ─────────────────────────── transposition table ───────────────────────────

struct TTEntry {
    uint64_t key;
    int32_t  score;
    int16_t  depth;
    uint8_t  flag;    // TT_EXACT / TT_LOWER / TT_UPPER
    uint8_t  pad;
    Move     move;
};

inline std::vector<TTEntry> g_tt;
inline uint64_t             g_tt_mask = 0;

inline void tt_resize(int mb) {
    std::size_t entries = static_cast<std::size_t>(mb) * 1024 * 1024 / sizeof(TTEntry);
    std::size_t pot = 1;
    while (pot * 2 <= entries) pot *= 2;
    g_tt.assign(pot, TTEntry{0, 0, 0, 0, 0, NO_MOVE});
    g_tt_mask = static_cast<uint64_t>(pot - 1);
}

inline void tt_clear() {
    std::fill(g_tt.begin(), g_tt.end(), TTEntry{0, 0, 0, 0, 0, NO_MOVE});
}

inline TTEntry *tt_probe(uint64_t key) {
    TTEntry *e = &g_tt[key & g_tt_mask];
    return (e->key == key) ? e : nullptr;
}

inline void tt_store(uint64_t key, int depth, int score, uint8_t flag, Move mv) {
    TTEntry *e = &g_tt[key & g_tt_mask];
    if (e->key != key || depth >= e->depth || flag == TT_EXACT)
        *e = {key, static_cast<int32_t>(score), static_cast<int16_t>(depth), flag, 0, mv};
}

// ─────────────────────────── correction history ─────────────────────────────
//
// Tracks the error between static evaluation and the search score at shallow
// depths, keyed by the pawn Zobrist hash (mod table size).  On each completed
// node we update the table; before returning static_eval we apply the
// accumulated correction.  This reduces systematic NNUE bias in quiet positions.
//
// Table size must be a power of two.

inline constexpr int CORR_HIST_SIZE = 16384;   // 2^14 entries
inline constexpr int CORR_HIST_MAX  = 1024;    // clamp range (centipawns)

// Indexed [color][pawn_hash & (CORR_HIST_SIZE-1)].
inline int16_t g_corr_hist[2][CORR_HIST_SIZE];

// Simple pawn Zobrist hash extracted from the full position hash.
// We re-use the position's hash xored with the pawn-specific contribution.
// Since we do not separately track the pawn hash, we use the lower bits of the
// full Zobrist hash with a different stride as an approximation — cheap and
// still highly correlated with pawn structure.
inline uint64_t pawn_hash_approx(const Position &pos) {
    // XOR all pawn squares into a single key component.
    uint64_t h = 0;
    Bitboard wp = pos.bb[WHITE][PAWN];
    Bitboard bp = pos.bb[BLACK][PAWN];
    while (wp) { Square s = pop_lsb(wp); h ^= uint64_t(0x9E3779B97F4A7C15ULL) << (s & 31); }
    while (bp) { Square s = pop_lsb(bp); h ^= uint64_t(0x6C62272E07BB0142ULL) << (s & 31); }
    return h;
}

inline void corr_hist_update(const Position &pos, int depth, int static_eval, int search_score) {
    int diff   = search_score - static_eval;
    // Scale update weight by depth; cap to CORR_HIST_MAX.
    int bonus  = std::clamp(diff * depth / 8, -CORR_HIST_MAX, CORR_HIST_MAX);
    uint64_t ph = pawn_hash_approx(pos);
    int      idx = static_cast<int>(ph & (CORR_HIST_SIZE - 1));
    gravity_update(g_corr_hist[pos.stm][idx], bonus);
}

inline int corr_hist_apply(const Position &pos, int static_eval) {
    uint64_t ph  = pawn_hash_approx(pos);
    int      idx = static_cast<int>(ph & (CORR_HIST_SIZE - 1));
    int      adj = g_corr_hist[pos.stm][idx];
    return static_eval + adj / 4;   // apply a fraction to avoid over-correction
}

// ─────────────────────────── per-ply stack entry ────────────────────────────
//
// Carries context from parent to child node (move played, static eval, etc.).

struct StackEntry {
    Move move        = NO_MOVE;   // move played to reach this node
    int  static_eval = -INF;      // corrected static eval at this node
    bool in_check    = false;
};

// ─────────────────────────── timing / global search state ──────────────────

using Clock = std::chrono::steady_clock;
inline Clock::time_point g_start_time;
inline int64_t           g_time_limit_ms = -1;
inline int64_t           g_node_limit    = -1;
inline int64_t           g_nodes         = 0;
inline int               g_root_history_size = 0;
inline std::atomic<bool> g_stop{false};

inline int64_t elapsed_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               Clock::now() - g_start_time).count();
}

// force_check=true bypasses the periodic node-count gate and checks the clock
// unconditionally.  Use at the root of iterative deepening so a depth boundary
// never silently ignores an expired time limit.
inline bool out_of_time(bool force_check = false) {
    if (g_stop.load(std::memory_order_relaxed)) return true;
    if (g_node_limit >= 0 && g_nodes >= g_node_limit) return true;
    if (g_time_limit_ms < 0) return false;
    if (!force_check && (g_nodes & 4095) != 0) return false;
    return elapsed_ms() >= g_time_limit_ms;
}

// ─────────────────────────── quiescence search ─────────────────────────────

inline int quiescence(Position &pos, Accumulator &acc, int alpha, int beta) {
    ++g_nodes;

    const bool stm_white = (pos.stm == WHITE);
    const int  bucket    = get_bucket(pos);
    const int  raw_eval  = nnue_eval(acc.stack[acc.top], stm_white, bucket);
    const int  stand_pat = corr_hist_apply(pos, raw_eval);

    if (stand_pat >= beta) return beta;
    if (stand_pat + PIECE_VALUE[QUEEN] + DELTA_MARGIN < alpha) return alpha;
    if (stand_pat > alpha) alpha = stand_pat;

    Move moves[MAX_MOVES];
    int  n = pos.gen_moves(moves, /*captures_only=*/true);

    ScoredMove ms[MAX_MOVES];
    for (int i = 0; i < n; ++i) ms[i] = {moves[i], 0};
    score_moves(pos, ms, n, /*ply=*/0, NO_MOVE);

    for (int i = 0; i < n; ++i) {
        pick_next(ms, i, n);
        Move      m        = ms[i].move;
        Square    from     = move_from(m);
        PieceType attacker = pos.piece_on[from];

        // Skip losing captures that don't pass a basic SEE check
        if (!see_ge(pos, m, -DELTA_MARGIN)) continue;

        bool king_moved = (attacker == KING);
        bool is_castle  = (move_flags(m) == MF_CASTLE);
        if (is_castle || king_moved) {
            pos.do_move(m);
            acc.push_full(pos);
        } else {
            acc.push_incremental(pos, m);
            pos.do_move(m);
        }

        int score = -quiescence(pos, acc, -beta, -alpha);
        pos.undo_move(m);
        acc.pop();

        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

// ─────────────────────────── forward declaration ────────────────────────────

static int alpha_beta(Position &pos, Accumulator &acc,
                      int depth, int alpha, int beta,
                      int ply, bool allow_null,
                      Move *pv, int &pv_len,
                      StackEntry *ss);   // ss points to ss[ply]

// ─────────────────────────── alpha_beta ────────────────────────────────────

static int alpha_beta(Position &pos, Accumulator &acc,
                      int depth, int alpha, int beta,
                      int ply, bool allow_null,
                      Move *pv, int &pv_len,
                      StackEntry *ss)
{
    ++g_nodes;
    pv_len = 0;

    // ── Hard ply cap — prevents infinite recursion from check extensions ───────
    if (ply >= MAX_PLY) return quiescence(pos, acc, alpha, beta);

    if (__builtin_expect(out_of_time(), 0)) return alpha;

    // ── Cheap draw checks ─────────────────────────────────────────────────────
    if (pos.cur.halfmove_clock >= 100 || pos.is_insufficient_material())
        return DRAW_SCORE;
    if (ply > 0 && pos.is_repetition(g_root_history_size))
        return DRAW_SCORE;

    const bool is_pv    = (beta - alpha > 1);
    const bool in_check = pos.in_check();
    ss[ply].in_check    = in_check;

    // Check extension at the top of the recursion (before qsearch drop)
    if (in_check && depth <= 0) depth = 1;

    // ── TT probe ──────────────────────────────────────────────────────────────
    uint64_t key     = pos.cur.hash;
    Move     tt_move = NO_MOVE;
    TTEntry *entry   = tt_probe(key);
    int      tt_score = -INF;

    if (entry) {
        if (entry->move != NO_MOVE) {
            Move   em = entry->move;
            Square ef = move_from(em), et = move_to(em);
            if (ef < 64 && et < 64
                && pos.piece_on[ef] != NO_PIECE
                && pos.color_on[ef] == pos.stm
                && (pos.piece_on[et] == NO_PIECE || pos.color_on[et] != pos.stm))
                tt_move = em;
        }
        tt_score = entry->score;
        if (!is_pv && entry->depth >= depth && tt_move != NO_MOVE) {
            if (entry->flag == TT_EXACT) { pv[0] = tt_move; pv_len = 1; return tt_score; }
            if (entry->flag == TT_LOWER && tt_score >= beta)  return tt_score;
            if (entry->flag == TT_UPPER && tt_score <= alpha) return tt_score;
        }
    }
    
    // ── Drop into qsearch ─────────────────────────────────────────────────────
    if (depth <= 0) return quiescence(pos, acc, alpha, beta);

    // ── Static evaluation ─────────────────────────────────────────────────────
    const bool stm_white   = (pos.stm == WHITE);
    const int  bucket      = get_bucket(pos);
    const int  raw_eval    = nnue_eval(acc.stack[acc.top], stm_white, bucket);
    const int  static_eval = corr_hist_apply(pos, raw_eval);

    ss[ply].static_eval = static_eval;

    // ── Improving flag ────────────────────────────────────────────────────────
    // A node is "improving" if our static eval is higher than it was two plies
    // ago.  Used to tighten or relax pruning.
    const bool improving = !in_check && ply >= 2
                        && !ss[ply - 2].in_check
                        && static_eval > ss[ply - 2].static_eval;

    // ── Reverse futility pruning ──────────────────────────────────────────────
    if (!in_check && !is_pv && depth <= 4
        && static_eval - RFP_MARGIN * depth * (improving ? 1 : 2) / 2 >= beta)
        return static_eval;

    // ── Null move pruning ─────────────────────────────────────────────────────
    if (allow_null && !in_check && !is_pv
        && depth >= NMP_MIN_DEPTH && static_eval >= beta
        && pos.piece_count > 2)   // avoid zugzwang-prone K+P endings
    {
        int R = NMP_R_BASE + depth / NMP_R_DIV + std::min((static_eval - beta) / 200, 3);
        acc.null_push();
        pos.do_move(NULL_MOVE);
        ss[ply + 1].move = NULL_MOVE;
        Move null_pv[MAX_PLY]; int null_len = 0;
        int null_score = -alpha_beta(pos, acc, depth - 1 - R,
                                     -beta, -beta + 1, ply + 1, false,
                                     null_pv, null_len, ss);
        pos.undo_move(NULL_MOVE);
        acc.pop();
        if (null_score >= beta) return beta;
    }

    // ── ProbCut ───────────────────────────────────────────────────────────────
    // At sufficient depth, do a shallow search with a raised beta (beta + margin)
    // for promising captures.  If the score exceeds the threshold, cut early.
    if (!is_pv && !in_check && depth >= PROBCUT_MIN_DEPTH
        && std::abs(beta) < MATE_SCORE - MAX_PLY)
    {
        int pc_beta = beta + PROBCUT_MARGIN;

        Move pc_moves[MAX_MOVES];
        int  pc_n = pos.gen_moves(pc_moves, /*captures_only=*/true);
        ScoredMove pc_ms[MAX_MOVES];
        for (int i = 0; i < pc_n; i++) pc_ms[i] = {pc_moves[i], 0};
        score_moves(pos, pc_ms, pc_n, ply, tt_move);

        for (int i = 0; i < pc_n; i++) {
            pick_next(pc_ms, i, pc_n);
            Move m = pc_ms[i].move;
            if (!see_ge(pos, m, pc_beta - static_eval)) continue;

            Square    from    = move_from(m);
            PieceType pt      = pos.piece_on[from];
            bool      is_king = (pt == KING);
            bool      is_cast = (move_flags(m) == MF_CASTLE);

            if (is_cast || is_king) { pos.do_move(m); acc.push_full(pos); }
            else { acc.push_incremental(pos, m); pos.do_move(m); }

            ss[ply + 1].move = m;
            Move pc_pv[MAX_PLY]; int pc_len = 0;
            int pc_score = -alpha_beta(pos, acc, depth - PROBCUT_MIN_DEPTH + 1,
                                       -pc_beta, -pc_beta + 1,
                                       ply + 1, true, pc_pv, pc_len, ss);
            pos.undo_move(m);
            acc.pop();

            if (pc_score >= pc_beta) {
                tt_store(key, depth - 3, pc_score, TT_LOWER, m);
                return pc_score;
            }
        }
    }

    // ── Futility condition ────────────────────────────────────────────────────
    const bool futil = (!in_check && !is_pv && depth <= 4
                        && static_eval + FUTILITY_MARGIN[depth] <= alpha);

    // ── Previous moves for ContinuationHistory context ───────────────────────
    Move            prev1 = (ply >= 1) ? ss[ply - 1].move : NO_MOVE;
    const Position *pos1  = nullptr;
    Move            prev2 = (ply >= 2) ? ss[ply - 2].move : NO_MOVE;
    const Position *pos2  = nullptr;

    // ── Quick move-count check for stalemate/mate ─────────────────────────────
    // We need to know if there are any legal moves before committing to the
    // singular extension search.  Use a fast pseudo-legal count as a guard.
    {
        Move tmp[MAX_MOVES];
        if (pos.gen_moves(tmp) == 0)
            return in_check ? -(MATE_SCORE - ply) : DRAW_SCORE;
    }

    // ── Singular extension check ──────────────────────────────────────────────
    bool singular_ext  = false;
    Move singular_move = NO_MOVE;
    if (!is_pv && tt_move != NO_MOVE
        && depth >= SE_MIN_DEPTH
        && entry && entry->depth >= depth - 3
        && entry->flag != TT_UPPER
        && std::abs(tt_score) < MATE_SCORE - MAX_PLY)
    {
        int s_beta  = tt_score - 2 * depth;
        int s_depth = (depth - 1) / SE_DEPTH_DIV;

        int  excl_best = -INF;
        Move excl_pv[MAX_PLY]; int excl_len = 0;

        // Search all moves except the TT move at reduced depth
        MovePicker se_mp(pos, NO_MOVE /*exclude tt_move by passing NO_MOVE as tt*/,
                         ply, false, prev1, pos1, prev2, pos2);
        Move em;
        while ((em = se_mp.next()) != NO_MOVE) {
            if (em == tt_move) continue;  // skip the candidate singular move
            Square    ef  = move_from(em);
            PieceType ept = pos.piece_on[ef];
            bool      ek  = (ept == KING);
            bool      ec  = (move_flags(em) == MF_CASTLE);
            if (ec || ek) { pos.do_move(em); acc.push_full(pos); }
            else          { acc.push_incremental(pos, em); pos.do_move(em); }
            ss[ply + 1].move = em;
            excl_len = 0;
            int es = -alpha_beta(pos, acc, s_depth,
                                 -s_beta, -s_beta + 1,
                                 ply + 1, true, excl_pv, excl_len, ss);
            pos.undo_move(em); acc.pop();
            if (es > excl_best) excl_best = es;
            if (excl_best >= s_beta) break;  // not singular
        }

        if (excl_best < s_beta) {
            singular_ext  = true;
            singular_move = tt_move;
        }
    }

    int   best_score  = -INF;
    Move  best_move   = NO_MOVE;
    int   orig_alpha  = alpha;
    int   moves_done  = 0;
    Color moving_side = pos.stm;

    // Track quiet moves tried (for penalty on beta cut-off)
    Move quiets_tried[MAX_MOVES];
    int  n_quiets_tried = 0;

    MovePicker mp(pos, tt_move, ply, false, prev1, pos1, prev2, pos2);
    Move m;
    while ((m = mp.next()) != NO_MOVE) {
        Square    from       = move_from(m);
        Square    to         = move_to(m);
        PieceType pt         = pos.piece_on[from];
        bool      is_capture = (pos.piece_on[to] != NO_PIECE || move_flags(m) == MF_EP);
        bool      is_prom    = (move_flags(m) == MF_PROMO);
        bool      is_castle  = (move_flags(m) == MF_CASTLE);
        bool      king_moved = (pt == KING);

        // ── Futility pruning ──────────────────────────────────────────────────
        if (futil && moves_done > 0 && !is_capture && !is_prom) continue;

        // ── SEE pruning for quiets at low depth ───────────────────────────────
        if (!is_pv && !in_check && !is_capture && !is_prom && moves_done > 0
            && depth <= 5 && !see_ge(pos, m, -50 * depth))
            continue;

        // ── Guard accumulator stack overflow ─────────────────────────────────
        if (acc.top + 1 >= Accumulator::MAX_STACK) break;

        // ── Accumulator push + do_move ────────────────────────────────────────
        if (is_castle || king_moved) {
            pos.do_move(m);
            acc.push_full(pos);
        } else {
            acc.push_incremental(pos, m);
            pos.do_move(m);
        }

        ss[ply + 1].move = m;

        const bool gives_check = pos.in_check();

        // ── Extension logic ───────────────────────────────────────────────────
        int ext = 0;
        if (gives_check) ext = 1;
        else if (singular_ext && m == singular_move) ext = 1;

        Move child_pv[MAX_PLY]; int child_len = 0;
        int score;

        if (moves_done == 0) {
            score = -alpha_beta(pos, acc, depth - 1 + ext,
                                -beta, -alpha, ply + 1, true, child_pv, child_len, ss);
        } else {
            // ── Late-move reduction (dynamic) ─────────────────────────────────
            int R = 0;
            if (depth >= LMR_MIN_DEPTH && moves_done >= LMR_FULL_MOVES
                && !is_capture && !gives_check && !is_prom && !in_check)
            {
                R = static_cast<int>(
                    0.75 + std::log(static_cast<double>(depth))
                         * std::log(static_cast<double>(moves_done)) / 2.25);

                if (!is_pv)    R++;
                if (improving) R = std::max(R - 1, 0);

                int ch = cont_hist_score(pt, to, prev1, pos1, prev2, pos2);
                if      (ch >  4000) R = std::max(R - 1, 0);
                else if (ch < -4000) R++;

                R = std::clamp(R, 0, depth - 2);
            }

            score = -alpha_beta(pos, acc, depth - 1 - R + ext,
                                -alpha - 1, -alpha, ply + 1, true, child_pv, child_len, ss);

            if (score > alpha && R > 0) {
                child_len = 0;
                score = -alpha_beta(pos, acc, depth - 1 + ext,
                                    -alpha - 1, -alpha, ply + 1, true, child_pv, child_len, ss);
            }

            if (score > alpha && score < beta) {
                child_len = 0;
                score = -alpha_beta(pos, acc, depth - 1 + ext,
                                    -beta, -alpha, ply + 1, true, child_pv, child_len, ss);
            }
        }

        pos.undo_move(m);
        acc.pop();

        if (!is_capture && !is_prom && n_quiets_tried < MAX_MOVES)
            quiets_tried[n_quiets_tried++] = m;

        ++moves_done;

        if (__builtin_expect(out_of_time(), 0)) break;

        if (score > best_score) {
            best_score = score;
            best_move  = m;
            if (score > alpha) {
                alpha  = score;
                pv[0]  = m;
                if (child_len > 0 && child_len < MAX_PLY - 1)
                    memcpy(pv + 1, child_pv, sizeof(Move) * child_len);
                pv_len = child_len + 1;
            }
        }

        if (__builtin_expect(score >= beta, 0)) {
            if (!is_capture) {
                killer_store(ply, m);
                history_update(moving_side, from, to, depth, true);
                // Penalise all quiet moves tried before this cut-off
                for (int j = 0; j < n_quiets_tried - 1; ++j) {
                    Move mj = quiets_tried[j];
                    history_update(moving_side, move_from(mj), move_to(mj), depth, false);
                }
            } else {
                PieceType vic = pos.piece_on[to];
                if (vic != NO_PIECE && vic < KING)
                    capture_hist_update(pt, to, vic, depth, true);
            }
            break;
        }
    }

    // ── Correction history update ─────────────────────────────────────────────
    // Update only at interior nodes (not qsearch) and not in check.
    if (!in_check && best_move != NO_MOVE && !out_of_time()) {
        corr_hist_update(pos, depth, static_eval, best_score);
    }

    // ── TT store ──────────────────────────────────────────────────────────────
    uint8_t flag = (best_score > orig_alpha && best_score < beta) ? TT_EXACT
                 : (best_score >= beta)                            ? TT_LOWER
                                                                   : TT_UPPER;
    tt_store(key, depth, best_score, flag, best_move);
    return best_score;
}

// ─────────────────────────── iterative deepening ───────────────────────────

struct SearchResult {
    Move best_move;
    int  best_score;
};

inline SearchResult iterative_deepen(Position &pos, Accumulator &acc,
                                     int max_depth, int64_t movetime_ms)
{
    g_stop.store(false, std::memory_order_relaxed);
    g_start_time    = Clock::now();
    g_time_limit_ms = movetime_ms;
    g_nodes         = 0;
    g_root_history_size = pos.history_top;

    acc.reset(pos);

    Move best_move  = NO_MOVE;
    int  best_score = 0;
    int  prev_score = 0;

    if (g_weights.loaded) {
        int bkt = get_bucket(pos);
        prev_score = nnue_eval(acc.stack[0], pos.stm == WHITE, bkt);
        prev_score = corr_hist_apply(pos, prev_score);
    }

    // Per-ply stack — initialised to safe defaults.
    // ss[0..MAX_PLY-1]: we index ss + ply inside alpha_beta.
    // Two extra entries at the bottom (indices -2, -1) let ply=0 look back.
    static StackEntry ss_storage[MAX_PLY + 4];
    StackEntry *ss = ss_storage + 2;   // ss[-2] and ss[-1] are valid
    for (int i = -2; i < MAX_PLY + 2; i++) {
        ss[i].move        = NO_MOVE;
        ss[i].static_eval = 0;
        ss[i].in_check    = false;
    }

    Move pv[MAX_PLY];
    int  pv_len = 0;

    for (int d = 1; d <= max_depth; ++d) {
        if (out_of_time(/*force_check=*/true)) break;

        Move iter_pv[MAX_PLY]; int iter_len = 0;
        int  score;

        if (d >= 4 && g_weights.loaded) {
            int window = ASP_WINDOW;
            int a = prev_score - window;
            int b = prev_score + window;

            for (int t = 0; t < ASP_MAX_TRIES; ++t) {
                iter_len = 0;
                score = alpha_beta(pos, acc, d, a, b, 0, true, iter_pv, iter_len, ss);
                if (out_of_time()) break;
                if      (score <= a) { a -= window; window *= 2; }
                else if (score >= b) { b += window; window *= 2; }
                else break;
            }

            // Emergency full-width fallback
            if (!out_of_time(/*force_check=*/true) &&
                (score <= prev_score - ASP_WINDOW * ASP_MAX_TRIES ||
                 score >= prev_score + ASP_WINDOW * ASP_MAX_TRIES))
            {
                iter_len = 0;
                score = alpha_beta(pos, acc, d, -INF, INF, 0, true, iter_pv, iter_len, ss);
            }
        } else {
            score = alpha_beta(pos, acc, d, -INF, INF, 0, true, iter_pv, iter_len, ss);
        }

        if (out_of_time(/*force_check=*/true) && d > 1) break;

        if (iter_len > 0) {
            best_move  = iter_pv[0];
            best_score = score;
            prev_score = score;
            memcpy(pv, iter_pv, sizeof(Move) * iter_len);
            pv_len = iter_len;
        }

        int64_t t_ms = elapsed_ms();
        int64_t nps  = (t_ms > 0) ? g_nodes * 1000 / t_ms : g_nodes;
        std::string pv_str;
        const int show = std::min(pv_len, 6);
        for (int i = 0; i < show; ++i) {
            if (i) pv_str += ' ';
            pv_str += pos.move_uci(pv[i]);
        }
        std::cout << "info depth " << d
                  << " score cp "  << best_score
                  << " nodes "     << g_nodes
                  << " nps "       << nps
                  << " time "      << t_ms
                  << " pv "        << pv_str
                  << "\n" << std::flush;
    }

    if (best_move == NO_MOVE) {
        Move moves[MAX_MOVES];
        int  n = pos.gen_moves(moves);
        if (n > 0) best_move = moves[0];
    }

    return {best_move, best_score};
}
