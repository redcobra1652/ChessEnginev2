#!/usr/bin/env python3
"""Quick correctness/sanity check for the polyglot opening book wired into
lichess-bot (lichess-bot/engines/book1.bin, lichess-bot/config.yml).

The book only exists at the lichess-bot Python layer -- get_book_move() is
called *before* any UCI `position`/`go` is sent, so a fastchess/UCI-only
SPRT can never exercise it (fastchess talks raw UCI straight to the
binary and has no concept of a book). This script instead reuses
lichess-bot's own get_book_move() directly, self-playing nnue_engine vs.
itself, alternating which side is allowed to consult the book each game.

Not an Elo experiment -- both sides run the identical binary/strength; the
only difference is whether one side is allowed to answer known theory
instantly from book instead of searching. This checks for (a) crashes /
illegal moves / hangs across the full get_book_move-then-engine flow, and
(b) how much clock time the book side actually saves in the opening.

Note: the engine itself now also has real UCI-level book support
(OwnBook/BookFile options, nnue_engine.cpp) which *can* be exercised
through fastchess -- see CLAUDE.md's opening-book section. This script
remains useful specifically for validating the lichess-bot Python-layer
book path, which is a separate code path from the engine's own.

Usage: python3 book_sanity_check.py [n_games]
"""
import sys
import time

import chess
import chess.engine
import chess.pgn

sys.path.insert(0, "lichess-bot/lib")
sys.path.insert(0, "lichess-bot")
from lib import engine_wrapper as ew  # noqa: E402
from lib.config import Configuration  # noqa: E402

ENGINE_PATH = "./nnue_engine"
NNFILE = "checkpoints/model.nnue"
BOOK_PATH = "lichess-bot/engines/book1.bin"

BASE_MS = 8000
INC_MS = 80

POLYGLOT_CFG = Configuration({
    "enabled": True,
    "book": {"standard": [BOOK_PATH]},
    "min_weight": 1,
    "selection": "weighted_random",
    "max_depth": 20,
    "normalization": "none",
})


class FakeGame:
    id = "sanity"


def play_game(book_is_white: bool, game_num: int):
    white_eng = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    black_eng = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    for eng in (white_eng, black_eng):
        eng.configure({"NNFile": NNFILE, "Threads": 1, "Hash": 64})

    board = chess.Board()
    wtime, btime = BASE_MS, BASE_MS
    book_hits = 0
    white_spend_ms = 0.0
    black_spend_ms = 0.0
    move_count = 0

    try:
        while not board.is_game_over(claim_draw=True) and move_count < 200:
            side_is_book = (board.turn == chess.WHITE) == book_is_white
            eng = white_eng if board.turn == chess.WHITE else black_eng
            clock = wtime if board.turn == chess.WHITE else btime
            t0 = time.monotonic()

            result = None
            if side_is_book:
                result = ew.get_book_move(board, FakeGame(), POLYGLOT_CFG)
                if result.move is not None:
                    book_hits += 1

            if result is None or result.move is None:
                limit = chess.engine.Limit(
                    white_clock=wtime / 1000, black_clock=btime / 1000,
                    white_inc=INC_MS / 1000, black_inc=INC_MS / 1000,
                )
                result = eng.play(board, limit, game=game_num)

            elapsed_ms = (time.monotonic() - t0) * 1000

            if result.move is None or result.move not in board.legal_moves:
                return "illegal_or_none", board.fen(), move_count, book_hits, white_spend_ms, black_spend_ms

            if board.turn == chess.WHITE:
                white_spend_ms += elapsed_ms
                wtime = max(0, wtime - elapsed_ms + INC_MS)
            else:
                black_spend_ms += elapsed_ms
                btime = max(0, btime - elapsed_ms + INC_MS)

            if wtime <= 0 or btime <= 0:
                return ("time_loss_white" if wtime <= 0 else "time_loss_black"), board.fen(), move_count, book_hits, white_spend_ms, black_spend_ms

            board.push(result.move)
            move_count += 1
    except chess.engine.EngineError as e:
        return f"engine_error:{e}", board.fen(), move_count, book_hits, white_spend_ms, black_spend_ms
    finally:
        white_eng.quit()
        black_eng.quit()

    outcome = board.outcome(claim_draw=True)
    res = outcome.result() if outcome else "*"
    return res, board.fen(), move_count, book_hits, white_spend_ms, black_spend_ms


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    results = []
    for i in range(n_games):
        book_is_white = (i % 2 == 0)
        res, fen, plies, book_hits, w_ms, b_ms = play_game(book_is_white, i)
        book_spend = w_ms if book_is_white else b_ms
        other_spend = b_ms if book_is_white else w_ms
        results.append((res, plies, book_hits, book_is_white, book_spend, other_spend))
        print(f"game {i+1}/{n_games}: result={res} plies={plies} book_hits={book_hits} "
              f"book_side={'white' if book_is_white else 'black'} "
              f"book_spend={book_spend/1000:.1f}s other_spend={other_spend/1000:.1f}s")

    bad = [r for r in results if r[0] not in ("1-0", "0-1", "1/2-1/2")]
    print()
    print(f"Completed {len(results)} games. Anomalies (crash/illegal/timeloss): {len(bad)}")
    for r in bad:
        print("  ", r)

    book_wins = sum(1 for res, _, _, is_white, _, _ in results
                     if (res == "1-0" and is_white) or (res == "0-1" and not is_white))
    book_losses = sum(1 for res, _, _, is_white, _, _ in results
                       if (res == "0-1" and is_white) or (res == "1-0" and not is_white))
    draws = sum(1 for res, _, _, _, _, _ in results if res == "1/2-1/2")
    total_book_hits = sum(r[2] for r in results)
    total_book_spend = sum(r[4] for r in results)
    total_other_spend = sum(r[5] for r in results)
    print(f"Book side: {book_wins}W-{book_losses}L-{draws}D across {len(results)} games")
    print(f"Total book moves played (across all games, both instant): {total_book_hits}")
    print(f"Total clock spend -- book side: {total_book_spend/1000:.1f}s, "
          f"no-book side: {total_other_spend/1000:.1f}s")


if __name__ == "__main__":
    main()
