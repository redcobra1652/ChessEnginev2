import os
import time
import chess
import chess.engine

# ── Paths (matching tournament.py) ─────────────────────────────────────────────
BASE        = os.path.expanduser("~/Documents/ChessModelv2/nnue")
ENGINE_PATH = os.path.join(BASE, "engine/engine")
NNUE_PATH   = os.path.join(BASE, "checkpoints/model.nnue")

DEPTH = 1
MOVE_TIMEOUT = 5.0  # Seconds per move limit before timing out


def debug_game():
    board = chess.Board()

    # Validate paths exist before starting engine process
    for path, label in [(ENGINE_PATH, "Engine binary"), (NNUE_PATH, "NNUE weights")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found at: {path}")

    print(f"[*] Launching engine: {ENGINE_PATH}")
    print(f"[*] Sending NNUE path: {NNUE_PATH}")

    # Launch engine process
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

    try:
        # Pass NNFile via UCI option (identical to tournament.py)
        engine.configure({"NNFile": NNUE_PATH})
        print("[+] Engine configured with NNFile successfully.")

        move_count = 1
        while not board.is_game_over():
            stm = "White" if board.turn == chess.WHITE else "Black"
            print(f"\n--- Move {move_count} ({stm}) | FEN: {board.fen()} ---")

            start_time = time.time()

            try:
                result = engine.play(
                    board,
                    chess.engine.Limit(depth=DEPTH, time=MOVE_TIMEOUT)
                )
                elapsed = time.time() - start_time

                print(f"[+] Engine returned move: {result.move} in {elapsed:.4f}s")
                board.push(result.move)
                move_count += 1

            except chess.engine.TimeoutError:
                elapsed = time.time() - start_time
                print(f"\n[!] ENGINE FROZE/TIMED OUT after {elapsed:.2f}s on Move {move_count}!")
                print(f"[!] FEN where freeze occurred:\n{board.fen()}")
                print(f"[!] Board Visual:\n{board}")
                break

    except Exception as e:
        print(f"\n[!] Error: {e}")

    finally:
        engine.quit()


if __name__ == "__main__":
    debug_game()