"""
data.py -- PGN streaming for NNUE training.

Output per position:
  white_indices : int32 array [MAX_ACTIVE] -- active HalfKP feature indices, white POV
  black_indices : int32 array [MAX_ACTIVE] -- active HalfKP feature indices, black POV
  score_cp      : float32                  -- eval in centipawns, side-to-move relative
  wdl           : float32 in [0,1]         -- game result from side-to-move perspective
  bucket_idx    : int64                    -- material-based bucket index
  stm_white     : bool                     -- True if White is the side to move

  stm_white is needed by model.forward(stm_white=...) to order the accumulator
  STM-first, matching search_core.c nnue_eval() and IncrementalAccumulator.evaluate().

HalfKP index layout (Stockfish-compatible):
  index = king_sq * (N_PIECE_TYPES * N_SQUARES) + piece_sq * N_PIECE_TYPES + piece_type_idx
  Dimension order: [king_sq=64][piece_sq=64][piece_type=10]  →  40960 features

PGN eval annotations:
  %eval stores values in PAWN units (e.g. +1.23 = 123 cp).
  parse_eval_cp() always returns centipawns regardless of input format.
  pawn_units=True  → input is in pawns  → multiply by 100  (default, covers Stockfish/LC0 PGNs)
  pawn_units=False → input is already in centipawns         (use if your engine writes raw cp)
"""

import glob
import gzip
import os
import pickle
import random
import re
import time

import chess
import chess.pgn
import numpy as np
import torch

# ── Constants ────────────────────────────────────────────────────────────────

PIECE_TYPES   = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
N_PIECE_TYPES = len(PIECE_TYPES) * 2   # 10: 5 types × 2 colors
N_SQUARES     = 64
# Stockfish HalfKP layout: [king_sq][piece_sq][piece_type]
HALFKP_SIZE   = N_SQUARES * N_SQUARES * N_PIECE_TYPES   # 64 × 64 × 10 = 40960
MAX_ACTIVE    = 30   # max non-zero features per perspective (30 non-king pieces max)

RESULT_TO_WDL_WHITE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}

# Matches both decimal pawn scores (+1.23, -0.50) and mate scores (#3, #-2)
EVAL_RE = re.compile(r"\[%eval\s+(#?-?\d+(?:\.\d+)?)\]")
MATE_CP = 30000   # sentinel centipawn value used for mate scores

RESERVOIR_VERSION = 4   # v4: matches bin format v4 (uint16 indices, compact eval)


# ── Bucket calculation ───────────────────────────────────────────────────────

def get_bucket(board: chess.Board, num_buckets: int = 8) -> int:
    """Classify position into a bucket based on non-king piece count (0..num_buckets-1)."""
    num_pieces = len(board.piece_map()) - 2   # exclude the two kings
    return min(num_buckets - 1, max(0, num_pieces * num_buckets // 32))


# ── HalfKP feature encoding ──────────────────────────────────────────────────

def _piece_type_index(piece: chess.Piece) -> int:
    """
    Map a piece to a 0..9 index.
    Layout: [P_w, P_b, N_w, N_b, B_w, B_b, R_w, R_b, Q_w, Q_b]
    i.e. piece_type_idx = PIECE_TYPES.index(type) * 2 + (0 if white else 1)
    """
    type_idx  = PIECE_TYPES.index(piece.piece_type)
    color_idx = 0 if piece.color == chess.WHITE else 1
    return type_idx * 2 + color_idx


def board_to_halfkp(board: chess.Board):
    """
    Encode board as two sets of HalfKP sparse indices (white POV and black POV).

    Stockfish-compatible index formula:
        white POV: king_sq * 640 + piece_sq * 10 + piece_type_idx
        black POV: mirror(king_sq) * 640 + mirror(piece_sq) * 10 + mirror(piece_type_idx)

    'mirror' = chess.square_mirror (vertical flip, a1↔a8).
    piece_type_idx is mirrored by flipping the color bit (XOR 1).

    Returns:
        white_indices: np.int16 [MAX_ACTIVE], -1 for unused slots
        black_indices: np.int16 [MAX_ACTIVE], -1 for unused slots
    """
    white_indices = np.full(MAX_ACTIVE, -1, dtype=np.int32)  # -1 = padding; convert.py maps to 65535 on pack
    black_indices = np.full(MAX_ACTIVE, -1, dtype=np.int32)

    white_king        = board.king(chess.WHITE)
    black_king_mirror = chess.square_mirror(board.king(chess.BLACK))

    wi = bi = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.piece_type == chess.KING:
            continue

        pidx = _piece_type_index(piece)

        # ── White POV ────────────────────────────────────────────────────────
        # index = king_sq * (N_SQUARES * N_PIECE_TYPES) + piece_sq * N_PIECE_TYPES + pidx
        if wi < MAX_ACTIVE:
            white_indices[wi] = (
                white_king * (N_SQUARES * N_PIECE_TYPES)
                + sq       * N_PIECE_TYPES
                + pidx
            )
            wi += 1

        # ── Black POV (board flipped vertically, colors swapped) ─────────────
        # Mirror the piece square and flip the color bit of the piece type index.
        mirrored_sq   = chess.square_mirror(sq)
        mirrored_pidx = pidx ^ 1   # swap color bit: white↔black piece type

        if bi < MAX_ACTIVE:
            black_indices[bi] = (
                black_king_mirror * (N_SQUARES * N_PIECE_TYPES)
                + mirrored_sq     * N_PIECE_TYPES
                + mirrored_pidx
            )
            bi += 1

    return white_indices, black_indices


# ── Eval parsing ─────────────────────────────────────────────────────────────

def parse_eval_cp(comment: str, pawn_units: bool = True) -> float | None:
    """
    Extract evaluation from a PGN comment and return it in CENTIPAWNS.

    PGN standard (and most engines including Stockfish, LC0) store %eval in
    pawn units: '+1.23' means +123 cp.  Set pawn_units=True (default).

    If your engine writes raw centipawns ('[%eval 123]'), pass pawn_units=False.

    Mate scores are returned as ±MATE_CP (30 000 cp).  Positions where
    abs(score) > max_eval_cp are filtered out in iter_positions.
    """
    if not comment:
        return None
    m = EVAL_RE.search(comment)
    if not m:
        return None
    raw = m.group(1)
    if raw.startswith("#"):
        mate_str = raw[1:]
        sign = -1 if mate_str.startswith("-") else 1
        return float(sign * MATE_CP)
    val = float(raw)
    # Convert to centipawns if the annotation is in pawn units (the common case)
    return val * 100.0 if pawn_units else val


# ── PGN path resolution ───────────────────────────────────────────────────────

def resolve_pgn_paths(pgn_arg) -> list[str]:
    if isinstance(pgn_arg, str):
        pgn_arg = [pgn_arg]
    paths = set()
    for arg in pgn_arg:
        if os.path.isdir(arg):
            paths.update(glob.glob(os.path.join(arg, "*.pgn")))
        elif any(ch in arg for ch in "*?["):
            paths.update(glob.glob(arg))
        else:
            paths.add(arg)
    resolved = sorted(paths)
    if not resolved:
        raise FileNotFoundError(f"No .pgn files found for {pgn_arg!r}")
    return resolved


# ── Position streaming ────────────────────────────────────────────────────────

def iter_positions(pgn_paths: list[str],
                   pawn_units: bool = True,
                   skip_check: bool = True,
                   min_ply: int = 16,
                   max_eval_cp: float = 5000.0,
                   num_buckets: int = 8):
    """
    Stream positions from PGN files.

    Yields tuples:
        (white_indices, black_indices, score_cp_stm, wdl_stm, bucket_idx, stm_white)

    score_cp_stm is in centipawns from the perspective of the side to move
    after the annotated move was played (i.e. the side whose turn it is when
    the engine evaluates the resulting position).

    stm_white is True when White is the side to move.  Pass it to
    model.forward(stm_white=...) so the accumulator is ordered STM-first.

    pawn_units:
        True  (default) — %eval is in pawn units (+1.00 = 100 cp). This is
                           the format used by Stockfish, LC0, and most GUIs.
        False           — %eval is already in centipawns.
    """
    for pgn_path in pgn_paths:
        with open(pgn_path, encoding="utf-8", errors="replace") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                result = game.headers.get("Result", "*")
                if result not in RESULT_TO_WDL_WHITE:
                    continue

                wdl_white = RESULT_TO_WDL_WHITE[result]
                board = game.board()
                node  = game
                ply   = 0

                for move in game.mainline_moves():
                    node = node.next()
                    board.push(move)
                    ply += 1

                    if ply < min_ply:
                        continue
                    if skip_check and board.is_check():
                        continue

                    # parse_eval_cp always returns centipawns (or None)
                    eval_cp_white = parse_eval_cp(node.comment, pawn_units=pawn_units)
                    if eval_cp_white is None or abs(eval_cp_white) > max_eval_cp:
                        continue

                    # After board.push(), board.turn is the side to move next.
                    # The %eval annotation is always from White's perspective.
                    # Negate for Black so score_cp_stm is always STM-relative.
                    stm_is_white = (board.turn == chess.WHITE)
                    score_cp_stm = eval_cp_white if stm_is_white else -eval_cp_white
                    wdl_stm      = wdl_white     if stm_is_white else (1.0 - wdl_white)

                    bucket_idx = get_bucket(board, num_buckets=num_buckets)
                    white_idx, black_idx = board_to_halfkp(board)

                    yield (white_idx, black_idx,
                           np.float32(score_cp_stm),
                           np.float32(wdl_stm),
                           np.int64(bucket_idx),
                           stm_is_white)


# ── Reservoir ─────────────────────────────────────────────────────────────────

class Reservoir:
    """
    Reservoir sampler — keeps a uniform random sample of all positions seen.

    The internal buffer stores raw numpy tuples; conversion to torch tensors
    happens only at sample_batch() time to keep memory usage minimal.
    """

    def __init__(self, capacity: int):
        self.capacity    = capacity
        self.buf: list   = []
        self.total_seen  = 0

    # ── Insertion ─────────────────────────────────────────────────────────────

    def add(self, item):
        self.total_seen += 1
        if len(self.buf) < self.capacity:
            self.buf.append(item)
        else:
            j = random.randrange(self.total_seen)
            if j < self.capacity:
                self.buf[j] = item

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample_batch(self, batch_size: int):
        """
        Returns a tuple of tensors:
            (white_indices, black_indices, score_cp, wdl, bucket, stm_white)

        stm_white is a bool tensor of shape [B]; pass it to model.forward() so
        that the accumulator is ordered STM-first, matching the C engine.

        Older reservoir files (version < 2 with 5-element tuples) will raise
        an IndexError on buf[i][5].  Delete the cache and re-stream in that case.
        """
        idxs = [random.randrange(len(self.buf)) for _ in range(batch_size)]
        wi   = np.stack([self.buf[i][0] for i in idxs])
        bi   = np.stack([self.buf[i][1] for i in idxs])
        cp   = np.array([self.buf[i][2] for i in idxs], dtype=np.float32)
        wdl  = np.array([self.buf[i][3] for i in idxs], dtype=np.float32)
        b    = np.array([self.buf[i][4] for i in idxs], dtype=np.int64)
        stm  = np.array([self.buf[i][5] for i in idxs], dtype=bool)
        return (torch.from_numpy(wi).long(),
                torch.from_numpy(bi).long(),
                torch.from_numpy(cp),
                torch.from_numpy(wdl),
                torch.from_numpy(b),
                torch.from_numpy(stm))

    def __len__(self):
        return len(self.buf)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Serialise the reservoir to a gzip-compressed pickle file.

        A version tag is stored so that future format changes can be detected
        cleanly rather than producing silent garbage.

        Typical size: ~80–120 MB for 2 M positions.
        """
        tmp = path + ".tmp"
        payload = {
            "version":     RESERVOIR_VERSION,
            "capacity":    self.capacity,
            "total_seen":  self.total_seen,
            "buf":         self.buf,
        }
        with gzip.open(tmp, "wb", compresslevel=1) as f:   # level 1 = fast
            pickle.dump(payload, f, protocol=4)
        os.replace(tmp, path)   # atomic rename — no half-written files
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  Saved reservoir → {path}  "
              f"({len(self.buf):,} positions, {size_mb:.0f} MB)", flush=True)

    @classmethod
    def load(cls, path: str) -> "Reservoir":
        """
        Load a reservoir previously saved with save().

        Raises ValueError if the file was written by an incompatible version.
        """
        t0 = time.time()
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)

        ver = payload.get("version", 1)
        if ver != RESERVOIR_VERSION:
            raise ValueError(
                f"Reservoir file version mismatch: file={ver}, "
                f"code={RESERVOIR_VERSION}. Delete the cache and re-stream."
            )

        r = cls(payload["capacity"])
        r.total_seen = payload["total_seen"]
        r.buf        = payload["buf"]
        print(f"  Loaded reservoir ← {path}  "
              f"({len(r.buf):,} positions, {time.time()-t0:.1f}s)", flush=True)
        return r
