#!/usr/bin/env python3
"""Full-coverage check of the new engine-internal OwnBook/BookFile UCI
support: for every prefix of every line in gen_opening_book.py's LINES
(the same theory build_polyglot_book.py encoded into book1.bin), feed the
engine that position via UCI and confirm its instant bestmove is one of
the moves python-chess's own polyglot reader says is actually in the book
at that position (not just legal -- actually book-sourced). Also times
each response to confirm it's instant (<50ms), i.e. it didn't fall through
to search.

Usage: python3 verify_book_coverage.py [engine_path]
"""
import subprocess
import sys
import time

import chess
import chess.polyglot

from gen_opening_book import LINES

ENGINE = sys.argv[1] if len(sys.argv) > 1 else "./nnue_engine_bookuci_candidate"
BOOK = "lichess-bot/engines/book1.bin"
NNFILE = "checkpoints/model.nnue"


def main():
    proc = subprocess.Popen([ENGINE], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    proc.stdin.write("uci\n")
    proc.stdin.flush()
    while "uciok" not in proc.stdout.readline():
        pass
    proc.stdin.write(f"setoption name NNFile value {NNFILE}\n")
    proc.stdin.write("setoption name OwnBook value true\n")
    proc.stdin.write(f"setoption name BookFile value {BOOK}\n")
    proc.stdin.write("isready\n")
    proc.stdin.flush()
    while "readyok" not in proc.stdout.readline():
        pass

    reader = chess.polyglot.open_reader(BOOK)

    total = 0
    fails = []
    slow = []
    for name, sans in LINES.items():
        board = chess.Board()
        moves_uci = []
        for san in sans:
            book_moves = {e.move for e in reader.find_all(board)}
            if not book_moves:
                break  # past book coverage for this line

            uci_line = "position startpos" + (" moves " + " ".join(moves_uci) if moves_uci else "")
            proc.stdin.write(uci_line + "\n")
            proc.stdin.write("go wtime 180000 btime 180000 winc 1000 binc 1000\n")
            proc.stdin.flush()
            t0 = time.monotonic()
            bestmove = None
            while True:
                line = proc.stdout.readline()
                if line.startswith("bestmove"):
                    bestmove = line.split()[1]
                    break
            elapsed_ms = (time.monotonic() - t0) * 1000

            total += 1
            mv = chess.Move.from_uci(bestmove)
            if mv not in book_moves:
                fails.append((name, board.fen(), bestmove, [m.uci() for m in book_moves]))
            if elapsed_ms > 50:
                slow.append((name, board.fen(), bestmove, elapsed_ms))

            move = board.push_san(san)
            moves_uci.append(move.uci())

    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.wait(timeout=5)

    print(f"Checked {total} book-covered positions across {len(LINES)} lines.")
    print(f"Wrong/non-book move: {len(fails)}")
    for f in fails[:10]:
        print("  ", f)
    print(f"Slow (>50ms, likely fell through to search): {len(slow)}")
    for s in slow[:10]:
        print("  ", s)

    ok = not fails and not slow
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
