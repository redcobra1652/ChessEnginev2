#!/usr/bin/env python3
"""
chess_server.py — local HTTP server that drives the nnue_engine UCI binary
and exposes /analyze (blocking) and /analyze_stream (SSE) endpoints for the
Chrome extension.

This replaces the old MCTS/policy-value server: the current engine
(nnue_engine.cpp) is a Stockfish-style alpha-beta UCI engine with a single
principal variation (no MultiPV support), not the old sims-based MCTS
searcher. The server's job is now just: speak UCI to one long-lived engine
subprocess, and translate `info depth ... score cp ... pv ...` lines into
JSON the extension can render.

Run:
    python3 chess_server.py
(auto-detects ../nnue_engine and ../checkpoints/model.nnue relative to this
file; override with --engine / --nnue if your layout differs.)
"""

import argparse
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# =============================================================================
# UCI engine driver
# =============================================================================

# nnue_engine.cpp's own MATE_SCORE (see nnue_engine.cpp) — the engine always
# emits "score cp N", never "score mate N", so a near-mate score just shows
# up as a huge cp value close to this constant. Anything with |cp| beyond
# MATE_THRESHOLD is a forced mate, not a real material/positional evaluation
# (no real position ever scores anywhere near this many centipawns).
MATE_SCORE = 900_000
MATE_THRESHOLD = 90_000


class EngineError(RuntimeError):
    pass


class UciEngine:
    """Drives one long-lived nnue_engine subprocess over UCI.

    The engine's main loop is single-threaded (search() runs synchronously
    inside the same loop that reads stdin), so a "stop" sent while a search
    is in flight won't be picked up until that search's own movetime budget
    naturally elapses. Rather than fight that, every search here is bounded
    by an explicit `go movetime N` (never `go infinite`), and calls are
    serialized with a lock — a piled-up caller just waits its turn, and the
    wait is always short and bounded by movetime_ms.
    """

    def __init__(self, engine_path, nnue_path, hash_mb=128, threads=4):
        self.engine_path = engine_path
        self.nnue_path = nnue_path
        self.hash_mb = hash_mb
        self.threads = threads
        self.lock = threading.Lock()
        self.proc = None
        self._start()

    def _start(self):
        self.proc = subprocess.Popen(
            [self.engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        if self.nnue_path:
            self._send(f"setoption name NNFile value {self.nnue_path}")
        self._send(f"setoption name Hash value {self.hash_mb}")
        self._send(f"setoption name Threads value {self.threads}")
        # Analysis wants a real search + eval every time, never a silent
        # book move with no info lines — keep the book off regardless of
        # what the live-play binary is configured with elsewhere.
        self._send("setoption name OwnBook value false")
        self._send("isready")
        self._wait_for("readyok")
        print(f"[server] nnue_engine ready (pid {self.proc.pid}) "
              f"nnue={self.nnue_path} hash={self.hash_mb}MB threads={self.threads}")

    def _send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise EngineError("engine closed stdout during startup/handshake")
            if token in line:
                return

    def new_game(self):
        with self.lock:
            self._send("ucinewgame")
            self._send("isready")
            self._wait_for("readyok")

    def analyze(self, fen, movetime_ms, info_cb=None):
        """
        Run one `go movetime N` search from `fen`. Calls info_cb(dict) for
        every parsed `info depth ...` line as it arrives (streaming). Returns
        the final result dict (the last info line's fields plus bestMove)
        once `bestmove` is seen.
        """
        with self.lock:
            self._send(f"position fen {fen}")
            self._send(f"go movetime {movetime_ms}")
            last_info = None
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    raise EngineError("engine closed stdout mid-search")
                line = line.strip()
                if line.startswith("info depth"):
                    parsed = parse_info_line(line)
                    if parsed:
                        last_info = parsed
                        if info_cb:
                            info_cb(parsed)
                elif line.startswith("bestmove"):
                    parts = line.split()
                    best = parts[1] if len(parts) > 1 else "0000"
                    result = dict(last_info or {})
                    result["bestMove"] = best
                    return result

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()


def parse_info_line(line):
    """Parse `info depth D score cp S nodes N nps R time T pv m1 m2 ...`."""
    toks = line.split()
    out = {}
    i = 1  # skip "info"
    n = len(toks)
    while i < n:
        key = toks[i]
        if key == "depth" and i + 1 < n:
            out["depth"] = int(toks[i + 1]); i += 2
        elif key == "score" and i + 2 < n:
            # This engine only ever emits "score cp N" (see nnue_engine.cpp);
            # handle "score mate N" defensively anyway in case that changes.
            if toks[i + 1] == "cp":
                out["scoreCp"] = int(toks[i + 2])
            elif toks[i + 1] == "mate":
                mate_in = int(toks[i + 2])
                sign = 1 if mate_in >= 0 else -1
                out["scoreCp"] = sign * (MATE_SCORE - abs(mate_in) * 2)
            i += 3
        elif key == "nodes" and i + 1 < n:
            out["nodes"] = int(toks[i + 1]); i += 2
        elif key == "nps" and i + 1 < n:
            out["nps"] = int(toks[i + 1]); i += 2
        elif key == "time" and i + 1 < n:
            out["timeMs"] = int(toks[i + 1]); i += 2
        elif key == "pv":
            out["pv"] = toks[i + 1:]
            break
        else:
            i += 1
    return out if "scoreCp" in out else None


# =============================================================================
# HTTP server
# =============================================================================

ENGINE: UciEngine = None
DEFAULT_MOVETIME = 800
MIN_MOVETIME = 50
MAX_MOVETIME = 5000


def clamp_movetime(ms):
    try:
        ms = int(ms)
    except (TypeError, ValueError):
        ms = DEFAULT_MOVETIME
    return max(MIN_MOVETIME, min(ms, MAX_MOVETIME))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        try:
            if int(args[1]) >= 400:
                super().log_message(fmt, *args)
        except (IndexError, ValueError):
            super().log_message(fmt, *args)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path == "/ping":
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok","engine":"NNUEEngine"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/analyze_stream":
            self._handle_analyze_stream()
        else:
            self.send_response(404)
            self.end_headers()

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        data = json.loads(body)
        fen = data.get("fen", "").strip()
        if not fen:
            raise ValueError("No FEN provided")
        movetime_ms = clamp_movetime(data.get("movetimeMs", DEFAULT_MOVETIME))
        new_game = bool(data.get("newGame", False))
        return fen, movetime_ms, new_game

    def _send_json_error(self, e):
        err = json.dumps({"error": str(e)}).encode()
        self.send_response(400)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(err)))
        self.end_headers()
        self.wfile.write(err)

    # ------------------------------------------------------------------
    # /analyze_stream — SSE, one event per completed iterative-deepening
    # depth, final event carries bestMove.
    # ------------------------------------------------------------------
    def _handle_analyze_stream(self):
        try:
            fen, movetime_ms, new_game = self._read_body()
        except Exception as e:
            self._send_json_error(e)
            return

        turn = "white" if fen.split(" ")[1] == "w" else "black"

        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        client_gone = False

        def sse(data_dict):
            nonlocal client_gone
            if client_gone:
                return
            line = f"data: {json.dumps(data_dict)}\n\n"
            try:
                self.wfile.write(line.encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                client_gone = True

        def info_cb(info):
            sse({**info, "turn": turn, "final": False})

        try:
            if new_game:
                ENGINE.new_game()
            result = ENGINE.analyze(fen, movetime_ms, info_cb=info_cb)
            sse({**result, "turn": turn, "final": True})
        except Exception as e:
            sse({"error": str(e)})


# =============================================================================
# Entry point
# =============================================================================

def find_default(name, subpath):
    """Look for `subpath` next to this script's parent dir, then cwd."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", subpath),
        os.path.join(os.getcwd(), subpath),
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isfile(c):
            return c
    return None


def main():
    global ENGINE

    parser = argparse.ArgumentParser(description="NNUE engine HTTP/UCI bridge for the Chrome extension")
    parser.add_argument("--engine", default=None, help="Path to nnue_engine binary (auto-detected if omitted)")
    parser.add_argument("--nnue", default=None, help="Path to model.nnue (auto-detected if omitted)")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--hash", type=int, default=128, help="TT size in MB")
    parser.add_argument("--threads", type=int, default=4, help="Lazy SMP search threads")
    args = parser.parse_args()

    engine_path = args.engine or find_default("engine", "nnue_engine")
    nnue_path = args.nnue or find_default("nnue", "checkpoints/model.nnue")

    if not engine_path:
        print("[server] ERROR: nnue_engine binary not found. Build it with:\n"
              "  clang++ -O3 -march=native -flto -std=c++17 -DNDEBUG -o nnue_engine nnue_engine.cpp\n"
              "or pass --engine /path/to/nnue_engine", file=sys.stderr)
        sys.exit(1)
    if not os.access(engine_path, os.X_OK):
        print(f"[server] ERROR: {engine_path} is not executable", file=sys.stderr)
        sys.exit(1)
    if not nnue_path:
        print("[server] ERROR: checkpoints/model.nnue not found. Pass --nnue /path/to/model.nnue", file=sys.stderr)
        sys.exit(1)

    print(f"[server] engine binary: {engine_path}")
    print(f"[server] nnue weights:  {nnue_path}")
    ENGINE = UciEngine(engine_path, nnue_path, hash_mb=args.hash, threads=args.threads)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[server] Listening on http://127.0.0.1:{args.port}")
    print("[server] Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] Stopping…")
        ENGINE.quit()
        sys.exit(0)


if __name__ == "__main__":
    main()
