#!/bin/bash
# diagnose.sh — NNUE engine move generation + eval diagnostics
# Usage: bash diagnose.sh [path/to/nnue_engine] [path/to/model.nnue]

ENGINE=${1:-./nnue_engine}
NNUE=${2:-checkpoints/model.nnue}

run() {
    local label="$1"
    local cmds="$2"
    echo "══════════════════════════════════════════"
    echo "  $label"
    echo "══════════════════════════════════════════"
    printf "%s" "$cmds" | "$ENGINE" 2>&1
    echo
}

# 1. Startpos — must not contain h1h2, e1f1, or any rook/king moving through pieces
run "Startpos depth 4" "uci
setoption name NNFile value $NNUE
isready
position startpos
go depth 4
quit
"

# 2. After 1.d4 d5 — the position that crashed with e1f1
run "After 1.d4 d5 (was crashing with e1f1)" "uci
setoption name NNFile value $NNUE
isready
position startpos moves d2d4 d7d5
go depth 4
quit
"

# 3. Castling position — king on e1, both rooks in place, path clear
run "Castling available (both sides)" "uci
setoption name NNFile value $NNUE
isready
position fen r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1
go depth 4
quit
"

# 4. Promotion position
run "Promotion (white pawn on e7)" "uci
setoption name NNFile value $NNUE
isready
position fen 3k4/4P3/8/8/8/8/8/4K3 w - - 0 1
go depth 4
quit
"

# 5. En passant
run "En passant available (e5 d5, ep on d6)" "uci
setoption name NNFile value $NNUE
isready
position startpos moves e2e4 a7a6 e4e5 d7d5
go depth 4
quit
"

# 6. Check evasion — king must move
run "King in check, must evade" "uci
setoption name NNFile value $NNUE
isready
position fen 4k3/8/8/8/8/8/8/r3K3 w - - 0 1
go depth 4
quit
"

echo "══════════════════════════════════════════"
echo "  Done. Look for:"
echo "  - Any illegal moves (piece moving through own pieces)"
echo "  - h1h2 / e1f1 style rook-through-pawn moves"
echo "  - bestmove 0000 (no legal moves found when there should be)"
echo "  - Castling: should see e1g1 or e1c1 in position 3"
echo "  - Promotion: should see e7e8q as best"
echo "  - EP: should consider e5d6"
echo "══════════════════════════════════════════"
