#!/usr/bin/env python3
"""Build a Polyglot opening book (.bin) for lichess-bot and for the engine's
own UCI OwnBook/BookFile options, from the same standard-theory lines used
by gen_opening_book.py's LINES dict, so the engine plays known-sound
openings instantly instead of burning search time on well-established
theory.

Every line contributes one weighted entry per ply: (position before the
move) -> (move played). Lines sharing an opening trunk (e.g. most 1.e4 e5
lines) naturally accumulate weight on the shared prefix and branch into
multiple candidate moves (each its own entry under the same position key)
at the point they diverge -- lichess-bot's "weighted_random" selection
(and the engine's own OwnBook selection) picks among those. All entries
here get equal weight (no engine-strength bias yet -- see CLAUDE.md's
opening-book section for why: the only self-play data available,
games/vs_sf2750.pgn, is 100 games from earlier, weaker versions of this
engine and too sparse per-opening to trust as a "plays well in this line"
signal for the current engine).

EXTRA_CONTINUATIONS extends each LINES entry with a further 4-6 half-moves
of standard theory, *past* the point gen_opening_book.py stops (that stop
point is exactly where openings.epd's FENs sit, since openings.epd is
generated from these same LINES). This exists so a fastchess SPRT that
starts games from openings.epd (for game-to-game diversity across 49 real
starting positions -- see the "actual measurement" SPRT in CLAUDE.md's
opening-book section for why this matters) still lands inside book
coverage instead of immediately falling through to search: without this,
every game would start exactly at the book's dead end. gen_opening_book.py
and openings.epd are NOT regenerated from this -- their existing 49 FENs
stay exactly as they are; this is purely additive, book-only continuation.

Usage: python3 build_polyglot_book.py [output_path]
"""
import struct
import sys

import chess

from gen_opening_book import LINES

ENTRY_STRUCT = struct.Struct(">QHHI")

EXTRA_CONTINUATIONS = {
    "Ruy Lopez Morphy":        "Be7 Re1 b5 Bb3".split(),
    "Ruy Lopez Berlin":        "d4 Nd6 Bxc6 dxc6".split(),
    "Ruy Lopez Exchange":      "O-O f6 d4 exd4".split(),
    "Italian Giuoco Piano":    "d3 O-O O-O d6".split(),
    "Italian Two Knights":     "exd5 Na5 Bb5+ c6".split(),
    "Scotch Game":             "Nxc6 bxc6 e5 Qe7".split(),
    "Petrov Defense":          "d4 d5 Bd3 Nc6".split(),
    "Four Knights":            "O-O O-O d3 d6".split(),
    "Vienna Game":             "Bg2 Nb6 Nge2 Nc6".split(),
    "King's Gambit":           "Ne5 Nf6 Bc4 d5".split(),

    "Sicilian Najdorf":        "Be2 e5 Nb3 Be7".split(),
    "Sicilian Dragon":         "Be3 Bg7 f3 O-O".split(),
    "Sicilian Sveshnikov":     "Ndb5 d6 Bg5 a6".split(),
    "Sicilian Taimanov":       "g3 a6 Bg2 Nf6".split(),
    "Sicilian Kan":            "Nc3 Qc7 Be2 Nf6".split(),
    "Sicilian Alapin":         "cxd4 d6 Nf3 Nc6".split(),
    "Sicilian Closed":         "d3 d6 f4 e6".split(),

    "French Winawer":          "a3 Bxc3+ bxc3 Ne7".split(),
    "French Classical":        "e5 Nfd7 Bxe7 Qxe7".split(),
    "French Tarrasch":         "Bd3 c5 c3 Nc6".split(),
    "French Advance":          "Nf3 Qb6 Be2 Nh6".split(),

    "Caro-Kann Classical":     "Ng3 Bg6 h4 h6".split(),
    "Caro-Kann Advance":       "Be2 c5 Be3 Qb6".split(),
    "Caro-Kann Exchange":      "c3 Nf6 Bf4 Bg4".split(),

    "Pirc Defense":            "Nf3 O-O Bd3 c5".split(),
    "Modern Defense":          "Nf3 O-O Bd3 c5".split(),
    "Alekhine Defense":        "Bc4 Nb6 Bb3 Bg7".split(),
    "Scandinavian Defense":    "Nf3 c6 Bc4 Bf5".split(),

    "Queen's Gambit Declined": "e3 O-O Nf3 h6".split(),
    "Queen's Gambit Accepted": "Bxc4 c5 O-O a6".split(),
    "Slav Defense":            "a4 Bf5 e3 e6".split(),
    "Semi-Slav":               "e3 Nbd7 Bd3 dxc4".split(),
    "Nimzo-Indian":            "Bd3 d5 Nf3 c5".split(),
    "Queen's Indian":          "Bg2 Be7 O-O O-O".split(),
    "King's Indian Classical": "Nf3 O-O Be2 e5".split(),
    "King's Indian Samisch":   "Be3 e5 d5 c6".split(),
    "Grunfeld Defense":        "e4 Nxc3 bxc3 Bg7".split(),
    "Catalan":                 "Nf3 O-O O-O dxc4".split(),
    "Benoni Defense":          "cxd5 d6 Nf3 g6".split(),
    "Dutch Defense":           "O-O O-O c4 d6".split(),
    "London System":           "Bg3 O-O Bd3 c5".split(),

    "English Symmetrical":     "Bg2 Bg7 O-O O-O".split(),
    "English Reversed Sicilian": "cxd5 Nxd5 Bg2 Nb6".split(),
    "Reti Opening":            "O-O O-O b3 c5".split(),
    "Bird's Opening":          "O-O O-O d3 c5".split(),

    "Queen's Pawn Colle":      "O-O Nc6 Nbd2 Bd6".split(),
    "Trompowsky Attack":       "c3 Nf6 d5 Qb6".split(),
    "Torre Attack":            "e3 b6 Bd3 Bb7".split(),
    "Budapest Gambit":         "Nf3 Bb4+ Nbd2 Qe7".split(),
}


def encode_move(move: chess.Move) -> int:
    promotion_part = (move.promotion - 1) if move.promotion else 0
    return (move.from_square << 6) | move.to_square | (promotion_part << 12)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "lichess-bot/engines/book1.bin"

    # key -> raw_move -> weight
    entries: dict[int, dict[int, int]] = {}
    n_lines = 0
    n_extended = 0
    for name, sans in LINES.items():
        full_sans = list(sans) + EXTRA_CONTINUATIONS.get(name, [])
        board = chess.Board()
        try:
            for i, san in enumerate(full_sans):
                key = chess.polyglot.zobrist_hash(board)
                move = board.push_san(san)
                raw = encode_move(move)
                entries.setdefault(key, {})
                entries[key][raw] = entries[key].get(raw, 0) + 1
                if i == len(sans) - 1:
                    n_lines += 1
        except Exception as e:
            print(f"[skip rest of] {name}: {e}")
            continue
        if name in EXTRA_CONTINUATIONS:
            n_extended += 1

    rows = []
    for key in sorted(entries):
        for raw, weight in entries[key].items():
            rows.append((key, raw, min(weight, 65535)))
    rows.sort(key=lambda r: r[0])

    with open(out_path, "wb") as f:
        for key, raw, weight in rows:
            f.write(ENTRY_STRUCT.pack(key, raw, weight, 0))

    print(f"Wrote {len(rows)} entries from {n_lines} lines ({n_extended} with extra continuations) to {out_path}")


if __name__ == "__main__":
    import chess.polyglot  # noqa: E402
    main()
