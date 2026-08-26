#!/usr/bin/env python3
"""Verify nnue_engine's internal Polyglot Zobrist hash (used by the new
OwnBook/BookFile UCI options) matches python-chess's chess.polyglot.zobrist_hash()
byte-for-byte on a range of positions, including castling-rights and
en-passant-eligible cases -- the two trickiest parts of the algorithm to
get right. Drives the engine directly via a temporary UCI debug command
("polyhash") that just prints the hash for the current position.

Usage: python3 verify_polyglot_hash.py [engine_path]
"""
import subprocess
import sys

import chess
import chess.polyglot

ENGINE = sys.argv[1] if len(sys.argv) > 1 else "./nnue_engine_bookuci_candidate"

def make(moves):
    b = chess.Board()
    for m in moves:
        b.push_san(m)
    return b

TEST_CASES = [
    ("startpos", make([])),
    ("1.e4", make(["e4"])),
    ("1.e4 e5 2.Nf3 Nc6 3.Bb5", make(["e4", "e5", "Nf3", "Nc6", "Bb5"])),
    ("after castling both sides", make(["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O", "Nf6", "d3", "O-O"])),
    ("en-passant eligible (white to move)", make(["e4", "Nf6", "e5", "d5"])),
    ("en-passant square set but NOT eligible", make(["h4", "a5", "h5", "g5"])),
    ("Sicilian Najdorf", make(["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"])),
    ("black castled queenside", make(["d4", "d5", "Nc3", "Nc6", "Bf4", "Bf5", "Qd2", "Qd7", "O-O-O", "O-O-O"])),
]


def engine_hash(proc, board: chess.Board) -> int:
    fen = board.fen()
    proc.stdin.write(f"position fen {fen}\n")
    proc.stdin.write("polyhash\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if line.startswith("polyhash "):
            return int(line.split()[1], 16)


def main():
    proc = subprocess.Popen([ENGINE], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    proc.stdin.write("uci\n")
    proc.stdin.flush()
    while "uciok" not in proc.stdout.readline():
        pass

    all_ok = True
    for name, board in TEST_CASES:
        py_hash = chess.polyglot.zobrist_hash(board)
        cpp_hash = engine_hash(proc, board)
        ok = py_hash == cpp_hash
        all_ok &= ok
        print(f"{'OK  ' if ok else 'FAIL'} {name:35s} py=0x{py_hash:016x} cpp=0x{cpp_hash:016x}")

    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.wait(timeout=5)

    print()
    print("ALL MATCH" if all_ok else "MISMATCH FOUND")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
