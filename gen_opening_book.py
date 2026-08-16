#!/usr/bin/env python3
"""Generate a small, diverse opening book (EPD, one FEN per line) for
fastchess (-openings file=... format=epd). Lines are standard, well-known
opening theory (public domain), typed from memory -- not scraped from any
source. Diversity here matters for SPRT test validity: games need
independent starting positions, not 100 replays of the same startpos line.
"""
import chess

# Each entry: (name, list of SAN moves). ~5-7 moves deep -- past the auto-
# book "book" depth but still standard theory, roughly balanced positions.
LINES = {
    "Ruy Lopez Morphy":        "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O".split(),
    "Ruy Lopez Berlin":        "e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4".split(),
    "Ruy Lopez Exchange":      "e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6".split(),
    "Italian Giuoco Piano":    "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6".split(),
    "Italian Two Knights":     "e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5".split(),
    "Scotch Game":             "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6".split(),
    "Petrov Defense":          "e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4".split(),
    "Four Knights":            "e4 e5 Nf3 Nc6 Nc3 Nf6 Bb5 Bb4".split(),
    "Vienna Game":             "e4 e5 Nc3 Nf6 g3 d5 exd5 Nxd5".split(),
    "King's Gambit":           "e4 e5 f4 exf4 Nf3 g5 h4 g4".split(),

    "Sicilian Najdorf":        "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6".split(),
    "Sicilian Dragon":         "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6".split(),
    "Sicilian Sveshnikov":     "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5".split(),
    "Sicilian Taimanov":       "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7".split(),
    "Sicilian Kan":            "e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6".split(),
    "Sicilian Alapin":         "e4 c5 c3 Nf6 e5 Nd5 d4 cxd4".split(),
    "Sicilian Closed":         "e4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7".split(),

    "French Winawer":          "e4 e6 d4 d5 Nc3 Bb4 e5 c5".split(),
    "French Classical":        "e4 e6 d4 d5 Nc3 Nf6 Bg5 Be7".split(),
    "French Tarrasch":         "e4 e6 d4 d5 Nd2 Nf6 e5 Nfd7".split(),
    "French Advance":          "e4 e6 d4 d5 e5 c5 c3 Nc6".split(),

    "Caro-Kann Classical":     "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5".split(),
    "Caro-Kann Advance":       "e4 c6 d4 d5 e5 Bf5 Nf3 e6".split(),
    "Caro-Kann Exchange":      "e4 c6 d4 d5 exd5 cxd5 Bd3 Nc6".split(),

    "Pirc Defense":            "e4 d6 d4 Nf6 Nc3 g6 f4 Bg7".split(),
    "Modern Defense":          "e4 g6 d4 Bg7 Nc3 d6 f4 Nf6".split(),
    "Alekhine Defense":        "e4 Nf6 e5 Nd5 d4 d6 Nf3 g6".split(),
    "Scandinavian Defense":    "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6".split(),

    "Queen's Gambit Declined": "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7".split(),
    "Queen's Gambit Accepted": "d4 d5 c4 dxc4 Nf3 Nf6 e3 e6".split(),
    "Slav Defense":            "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4".split(),
    "Semi-Slav":               "d4 d5 c4 c6 Nf3 Nf6 Nc3 e6".split(),
    "Nimzo-Indian":            "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O".split(),
    "Queen's Indian":          "d4 Nf6 c4 e6 Nf3 b6 g3 Bb7".split(),
    "King's Indian Classical": "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6".split(),
    "King's Indian Samisch":   "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O".split(),
    "Grunfeld Defense":        "d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5".split(),
    "Catalan":                 "d4 Nf6 c4 e6 g3 d5 Bg2 Be7".split(),
    "Benoni Defense":          "d4 Nf6 c4 c5 d5 e6 Nc3 exd5".split(),
    "Dutch Defense":           "d4 f5 g3 Nf6 Bg2 e6 Nf3 Be7".split(),
    "London System":           "d4 d5 Nf3 Nf6 Bf4 e6 e3 Bd6".split(),

    "English Symmetrical":     "c4 c5 Nf3 Nf6 Nc3 Nc6 g3 g6".split(),
    "English Reversed Sicilian": "c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5".split(),
    "Reti Opening":            "Nf3 d5 c4 e6 g3 Nf6 Bg2 Be7".split(),
    "Bird's Opening":          "f4 d5 Nf3 Nf6 e3 g6 Be2 Bg7".split(),

    "Queen's Pawn Colle":      "d4 d5 Nf3 Nf6 e3 e6 Bd3 c5".split(),
    "Trompowsky Attack":       "d4 Nf6 Bg5 Ne4 Bf4 c5 f3 Qa5+".split(),
    "Torre Attack":            "d4 Nf6 Nf3 e6 Bg5 Be7 Nbd2 O-O".split(),
    "Budapest Gambit":         "d4 Nf6 c4 e5 dxe5 Ng4 Bf4 Nc6".split(),
}


def main():
    out_path = "openings.epd"
    n = 0
    with open(out_path, "w") as f:
        for name, sans in LINES.items():
            board = chess.Board()
            try:
                for san in sans:
                    board.push_san(san)
            except Exception as e:
                print(f"[skip] {name}: {e}")
                continue
            f.write(board.fen() + "\n")
            n += 1
    print(f"Wrote {n} opening positions to {out_path}")


if __name__ == "__main__":
    main()
