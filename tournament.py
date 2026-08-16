#!/usr/bin/env python3
"""
tournament.py — NNUE engine vs Stockfish, Elo measurement + PGN output.

Usage:
    python3 tournament.py                        # 100 games, SF@2000, depth 7
    python3 tournament.py --games 50 --elo 1500
    python3 tournament.py --games 200 --elo 2500 --depth 8
"""

import argparse
import math
import os
import platform
import subprocess
import sys
import time
import chess
import chess.engine
import chess.pgn

# ── Build helpers ─────────────────────────────────────────────────────────────

def build_command(src: str, out: str) -> list[str]:
    """Return the right compile command for the current platform."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        # Apple Silicon / ARM Linux — NEON is always-on, -mavx2/-mfma are x86-only
        return ["clang++", "-O3", "-march=native", "-flto",
                "-std=c++17", "-DNDEBUG", "-o", out, src]
    else:
        # x86-64 Linux / Windows
        return ["g++", "-O3", "-march=native", "-mavx2", "-mfma", "-flto",
                "-std=c++17", "-DNDEBUG", "-o", out, src]

def rebuild_engine(src: str, out: str) -> None:
    cmd = build_command(src, out)
    print(f"  Building: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  BUILD FAILED:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"  Build OK → {out}")

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE        = os.path.expanduser("~/Documents/ChessModelv2/nnue")
ENGINE_SRC  = os.path.join(BASE, "nnue_engine.cpp")   # source for --rebuild
ENGINE_PATH = os.path.join(BASE, "nnue_engine")
NNUE_PATH   = os.path.join(BASE, "checkpoints/model.nnue")
SF_PATH     = os.path.join(BASE, "stockfish/stockfish-macos-m1-apple-silicon")
PGN_DIR     = os.path.join(BASE, "games")

# ── Elo helpers ───────────────────────────────────────────────────────────────

def elo_diff(score: float) -> float:
    score = max(0.0001, min(0.9999, score))
    return 400 * math.log10(score / (1 - score))

def elo_margin(score: float, n: int) -> float:
    if n == 0 or score <= 0 or score >= 1:
        return 999.0
    return 400 * 1.96 * math.sqrt(score * (1 - score) / n) / math.log(10)

# ── One game ──────────────────────────────────────────────────────────────────

def play_game(nnue, sf, nnue_limit, sf_limit, nnue_is_white: bool, game_num: int, max_moves: int):
    board = chess.Board()
    game  = chess.pgn.Game()
    game.headers["Event"] = "NNUE Tournament"
    game.headers["Round"] = str(game_num)
    game.headers["White"] = "NNUE" if nnue_is_white else "Stockfish"
    game.headers["Black"] = "Stockfish" if nnue_is_white else "NNUE"
    node = game

    for _ in range(max_moves):
        if board.is_game_over(claim_draw=False):
            break
        nnue_turn = (board.turn == chess.WHITE) == nnue_is_white
        engine = nnue if nnue_turn else sf
        limit  = nnue_limit if nnue_turn else sf_limit
        try:
            result = engine.play(board, limit, game=game_num)
        except chess.engine.EngineTerminatedError:
            break
        if result.move is None:
            break
        node = node.add_variation(result.move)
        board.push(result.move)

    outcome = board.outcome(claim_draw=False)
    game.headers["Result"] = board.result(claim_draw=False)

    if outcome is None or outcome.winner is None:
        return "draw", game
    won = (outcome.winner == chess.WHITE) == nnue_is_white
    return ("win" if won else "loss"), game

# ── Tournament ────────────────────────────────────────────────────────────────

def run(engine_path, nnue_path, sf_path, sf_elo, num_games, nnue_depth, sf_movetime_ms, pgn_dir, max_moves):
    for path, label in [(engine_path, "engine"), (nnue_path, "model.nnue"), (sf_path, "stockfish")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    if pgn_dir:
        os.makedirs(pgn_dir, exist_ok=True)
        pgn_path = os.path.join(pgn_dir, f"vs_sf{sf_elo}.pgn")

    print(f"\n{'='*60}")
    print(f"  NNUE (depth {nnue_depth}) vs Stockfish@{sf_elo} ({sf_movetime_ms}ms/move)")
    print(f"  {num_games} games  |  PGN: {'yes' if pgn_dir else 'no'}")
    print(f"{'='*60}")

    # NNUE uses depth limit (avoids the stop/g_time_limit_ms=0 bug)
    # Stockfish uses movetime
    nnue_limit = chess.engine.Limit(depth=nnue_depth)
    sf_limit   = chess.engine.Limit(time=sf_movetime_ms / 1000)

    wins = losses = draws = 0
    t0 = time.time()

    with chess.engine.SimpleEngine.popen_uci(engine_path) as nnue, \
         chess.engine.SimpleEngine.popen_uci(sf_path) as sf:

        nnue.configure({"NNFile": nnue_path})
        sf.configure({
            "Hash": 64,
            "Threads": 1,
            "UCI_LimitStrength": True,
            "UCI_Elo": sf_elo,
        })

        for i in range(num_games):
            nnue_is_white = (i % 2 == 0)
            result, game = play_game(nnue, sf, nnue_limit, sf_limit, nnue_is_white, i + 1, max_moves)

            if result == "win":    wins   += 1
            elif result == "loss": losses += 1
            else:                  draws  += 1

            n      = wins + losses + draws
            score  = (wins + 0.5 * draws) / n
            diff   = elo_diff(score)
            margin = elo_margin(score, n)
            color  = "W" if nnue_is_white else "B"

            print(f"  {n:3d}/{num_games}  [{color}] {result:<4}  "
                  f"W{wins}/L{losses}/D{draws}  "
                  f"score={score:.0%}  "
                  f"Elo≈{sf_elo + diff:.0f} ±{margin:.0f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

            if pgn_dir:
                with open(pgn_path, "a") as f:
                    print(game, file=f)
                    print(file=f)

    n      = wins + losses + draws
    score  = (wins + 0.5 * draws) / max(n, 1)
    diff   = elo_diff(score)
    margin = elo_margin(score, n)

    print(f"\n{'='*60}")
    print(f"  Result: W{wins} / L{losses} / D{draws}  ({n} games)")
    print(f"  Score:  {score:.1%}")
    print(f"  Est. Elo: {sf_elo + diff:.0f} ± {margin:.0f}  (95% CI)")
    if pgn_dir:
        print(f"  PGN: {pgn_path}")
    print(f"{'='*60}\n")

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--engine",      default=ENGINE_PATH)
    p.add_argument("--src",         default=ENGINE_SRC,
                   help="C++ source file (used with --rebuild)")
    p.add_argument("--rebuild",     action="store_true",
                   help="Recompile nnue_engine.cpp before running "
                        "(auto-selects clang++/g++ and correct flags for your CPU)")
    p.add_argument("--nnue",        default=NNUE_PATH)
    p.add_argument("--stockfish",   default=SF_PATH)
    p.add_argument("--elo",         type=int,   default=2000)
    p.add_argument("--games",       type=int,   default=100)
    p.add_argument("--depth",       type=int,   default=7,
                   help="Search depth for your NNUE engine (default: 7)")
    p.add_argument("--sf-movetime", type=int,   default=50,
                   help="Stockfish ms per move (default: 50)")
    p.add_argument("--max-moves",   type=int,   default=300)
    p.add_argument("--no-pgn",      action="store_true")
    a = p.parse_args()

    if a.rebuild:
        rebuild_engine(a.src, a.engine)

    run(
        engine_path    = a.engine,
        nnue_path      = a.nnue,
        sf_path        = a.stockfish,
        sf_elo         = a.elo,
        num_games      = a.games,
        nnue_depth     = a.depth,
        sf_movetime_ms = a.sf_movetime,
        pgn_dir        = None if a.no_pgn else PGN_DIR,
        max_moves      = a.max_moves,
    )
