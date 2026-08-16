/*
 * main.cpp — UCI protocol loop for the HalfKP NNUE engine.
 *
 * Handles:
 *   uci         → identify the engine and list options
 *   isready     → readyok
 *   ucinewgame  → reset board to startpos, clear TT
 *   setoption   → NNFile (load .nnue), Hash (resize TT), Threads (ignored)
 *   position    → set board from startpos / FEN, then apply move list
 *   go          → run iterative_deepen(), output info lines, "bestmove"
 *   stop        → set g_stop flag to interrupt the current search
 *   d           → debug: print current board to stderr
 *   quit        → exit
 *
 * Build (C++17):
 *   # Apple Silicon / ARM64
 *   clang++ -O3 -march=native -flto -std=c++17 \
 *       -o engine main.cpp && ./engine
 *
 *   # Linux / Windows x86-64
 *   g++ -O3 -march=native -mavx2 -mfma -flto -std=c++17 \
 *       -o engine main.cpp
 *
 * All engine logic lives in the headers; main.cpp is a thin UCI wrapper.
 */

#include "types.h"
#include "position.h"
#include "movepick.h"
#include "evaluate.h"
#include "search.h"

#include <iostream>
#include <sstream>
#include <string>

// ─────────────────────────── debug board print ─────────────────────────────

static void print_board(const Position &pos) {
    static const char PC[] = ".PNBRQKpnbrqk";
    for (int r = 7; r >= 0; --r) {
        for (int f = 0; f < 8; ++f) {
            Square    sq = make_sq(f, r);
            PieceType pt = pos.piece_on[sq];
            if (pt == NO_PIECE) {
                std::cerr << ". ";
            } else {
                Color c   = pos.color_on[sq];
                int   idx = (c == WHITE) ? pt + 1 : pt + 7;
                std::cerr << PC[idx] << ' ';
            }
        }
        std::cerr << '\n';
    }
    std::cerr << (pos.stm == WHITE ? "White" : "Black") << " to move\n" << std::flush;
}

// ─────────────────────────── main ──────────────────────────────────────────

int main() {
    // One-time global initialisation (magic bitboards, Zobrist tables)
    Position::init();
    tt_resize(128);   // default 128 MB; overridden by "setoption Hash"

    Position    pos;
    Accumulator acc;
    std::string nn_file;

    pos.set_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // Announce engine identity and options
    std::cout << "id name NNUEEngine\n"
              << "id author nnue-trainer\n"
              << "option name NNFile  type string default <empty>\n"
              << "option name Hash    type spin default 128 min 1 max 2048\n"
              << "option name Threads type spin default 1   min 1 max 1\n"
              << "uciok\n" << std::flush;

    std::string line;
    int         search_depth = 64;
    int64_t     movetime_ms  = -1;

    while (std::getline(std::cin, line)) {
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;

        // ── uci ──────────────────────────────────────────────────────────────
        if (cmd == "uci") {
            std::cout << "id name NNUEEngine\n"
                      << "id author nnue-trainer\n"
                      << "option name NNFile  type string default <empty>\n"
                      << "option name Hash    type spin default 128 min 1 max 2048\n"
                      << "option name Threads type spin default 1   min 1 max 1\n"
                      << "uciok\n" << std::flush;

        // ── isready ───────────────────────────────────────────────────────────
        } else if (cmd == "isready") {
            std::cout << "readyok\n" << std::flush;

        // ── ucinewgame ────────────────────────────────────────────────────────
        } else if (cmd == "ucinewgame") {
            pos.set_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            tt_clear();
            search_init();
            memset(g_corr_hist, 0, sizeof g_corr_hist);

        // ── setoption ─────────────────────────────────────────────────────────
        } else if (cmd == "setoption") {
            std::string tok, name;
            ss >> tok;  // "name"
            ss >> name;
            std::string extra;
            while (ss >> extra && extra != "value") {
                name += " " + extra;
            }
            // Read the entire rest of the stream as value to support spaces in file paths
            std::string value;
            std::getline(ss >> std::ws, value);

            if (name == "NNFile" && !value.empty() && value != "<empty>") {
                nn_file = value;
                load_nnue(value);
            }
            if (name == "Hash") {
                try { tt_resize(std::stoi(value)); } catch (...) {}
            }
            // Threads option is accepted but ignored (single-threaded engine)

        // ── position ──────────────────────────────────────────────────────────
        } else if (cmd == "position") {
            std::string token;
            ss >> token;

            if (token == "startpos") {
                pos.set_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                ss >> token;   // optionally "moves"
            } else if (token == "fen") {
                std::string fen;
                for (int i = 0; i < 6; ++i) {
                    std::string part;
                    ss >> part;
                    if (part == "moves") { token = part; break; }
                    if (i > 0) fen += ' ';
                    fen  += part;
                    token = "";
                }
                pos.set_from_fen(fen);
                if (token.empty()) ss >> token;   // consume "moves" if present
            }

            // Apply move list
            if (token == "moves") {
                std::string mv;
                while (ss >> mv) {
                    Move m = pos.parse_uci(mv);
                    if (m != NO_MOVE) pos.do_move(m);
                }
            }

            // CRITICAL FIX: Sync accumulator with current board state after playing moves
            acc.reset(pos);

        // ── go ────────────────────────────────────────────────────────────────
        } else if (cmd == "go") {
            g_stop.store(false, std::memory_order_relaxed); // Fix 1: Reset stop flag

            movetime_ms  = -1;
            search_depth = 64;
            int64_t go_nodes = -1;
            bool    infinite = false;
            int wtime = -1, btime = -1, winc = 0, binc = 0, movestogo = 30;

            std::string tok;
            while (ss >> tok) {
                if      (tok == "movetime")  ss >> movetime_ms;
                else if (tok == "depth")     ss >> search_depth;
                else if (tok == "nodes")     ss >> go_nodes;
                else if (tok == "infinite")  infinite = true;
                else if (tok == "wtime")     ss >> wtime;
                else if (tok == "btime")     ss >> btime;
                else if (tok == "winc")      ss >> winc;
                else if (tok == "binc")      ss >> binc;
                else if (tok == "movestogo") ss >> movestogo;
            }

            if (infinite) { movetime_ms = -1; go_nodes = -1; }

            if (!infinite && movetime_ms < 0 && go_nodes < 0
                && (wtime >= 0 || btime >= 0))
            {
                int myTime = (pos.stm == WHITE) ? wtime : btime;
                int myInc  = (pos.stm == WHITE) ? winc  : binc;
                if (myTime >= 0) {
                    movetime_ms = std::max(int64_t(50),
                        int64_t(myTime / movestogo + static_cast<int>(myInc * 0.8)));
                }
            }

            g_node_limit = go_nodes;

            // FIX 2: Re-sync NNUE feature accumulator with the current position!
            acc.reset(pos);

            if (!g_weights.loaded) {
                Move moves[MAX_MOVES];
                int  n = pos.gen_moves(moves);
                Move best = (n > 0) ? moves[0] : NO_MOVE;
                std::string best_str = (best != NO_MOVE) ? pos.move_uci(best) : "0000";
                std::cout << "bestmove " << best_str << "\n" << std::flush;
            } else {
                auto [mv, score] = iterative_deepen(pos, acc, search_depth, movetime_ms);
                std::string best_str = (mv != NO_MOVE) ? pos.move_uci(mv) : "0000";
                std::cout << "bestmove " << best_str << "\n" << std::flush;
            }
            
            
        // ── stop ──────────────────────────────────────────────────────────────
        } else if (cmd == "stop") {
            g_stop.store(true, std::memory_order_relaxed);

        // ── quit ──────────────────────────────────────────────────────────────
        } else if (cmd == "quit") {
            break;

        // ── d (debug: print board) ────────────────────────────────────────────
        } else if (cmd == "d") {
            print_board(pos);
        }
    }

    return 0;
}
