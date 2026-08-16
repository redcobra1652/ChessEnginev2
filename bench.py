#!/usr/bin/env python3
"""
bench.py — speed diagnostic for nnue_engine
Usage:
    python3 bench.py [--engine ./nnue_engine] [--depths 4,6,8,10,12] [--nnue checkpoints/model.nnue]

For each depth it runs a fixed set of positions, collects the engine's
info lines, and prints a summary table of time, nodes, and NPS.
The model is loaded once and verified before any search begins.
"""

import argparse
import subprocess
import sys
import time
import re
from pathlib import Path

# ── benchmark positions ────────────────────────────────────────────────────
POSITIONS = [
    ("startpos",       "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("sicilian",       "rnbqkb1r/pp1ppppp/5n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),
    ("french-adv",     "rnbqkb1r/ppp2ppp/4pn2/3pP3/3P4/2N5/PPP2PPP/R1BQKBNR w KQkq - 1 5"),
    ("mid-tactic",     "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 4 10"),
    ("rook-end",       "8/5pk1/6p1/7p/7P/6P1/5PK1/8 w - - 0 1"),
    ("q-vs-r",         "8/8/8/3k4/8/8/3K4/3Q4 w - - 0 1"),
    ("complex-mid",    "r2q1rk1/1b2bppp/p1n1pn2/1p6/3P4/1BN1PN2/PP3PPP/R1BQR1K1 b - - 0 12"),
    ("king-safety",    "r3k2r/1pp2ppp/p1nb1n2/4p1q1/4P3/P1NB1N2/1PP2PPP/R2QR1K1 w kq - 0 12"),
]

# ── UCI helpers ────────────────────────────────────────────────────────────

def send(proc, line: str):
    proc.stdin.write(line + "\n")
    proc.stdin.flush()

def read_until(proc, sentinel: str, timeout: float = 120.0) -> list[str]:
    lines = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.rstrip()
        lines.append(line)
        if line.startswith(sentinel):
            break
    return lines

def parse_info(lines: list[str]) -> dict:
    """Return the last 'info depth' line parsed into a dict."""
    result = {}
    for line in reversed(lines):
        if not line.startswith("info depth"):
            continue
        for key in ("depth", "seldepth", "score cp", "nodes", "nps", "time"):
            m = re.search(rf"{re.escape(key)}\s+(-?\d+)", line)
            if m:
                result[key] = int(m.group(1))
        m = re.search(r"score mate\s+(-?\d+)", line)
        if m:
            result["score mate"] = int(m.group(1))
        break
    return result

def fmt_num(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)

# ── model loading ──────────────────────────────────────────────────────────

def load_model(proc, nnue_path: str) -> bool:
    """
    Send setoption NNFile, then isready and look for the engine's
    'info string Loaded ...' confirmation. Returns True on success.
    """
    nnue = Path(nnue_path)
    if not nnue.exists():
        print(f"  [ERROR] Model file not found: {nnue_path}")
        print( "          Make sure the path is correct relative to where you run this script.")
        return False

    print(f"  Loading model: {nnue_path}  ({nnue.stat().st_size / 1024:.1f} KB) ... ", end="", flush=True)
    t0 = time.monotonic()

    send(proc, f"setoption name NNFile value {nnue_path}")
    send(proc, "isready")

    # Engine emits "info string Loaded <path> ..." on success,
    # or "info string Cannot open ..." on failure, then "readyok".
    lines = read_until(proc, "readyok", timeout=10)

    elapsed = time.monotonic() - t0

    loaded   = any("info string Loaded"      in l for l in lines)
    failed   = any("info string Cannot open" in l for l in lines)
    no_model = any("info string Read error"  in l for l in lines)

    if loaded:
        # Pull the confirmation line for display
        msg = next((l for l in lines if "info string Loaded" in l), "")
        print(f"OK  ({elapsed*1000:.0f} ms)")
        print(f"  Engine says: {msg.replace('info string ', '')}")
        return True
    elif failed or no_model:
        err = next((l for l in lines if "info string" in l), "unknown error")
        print("FAILED")
        print(f"  Engine says: {err.replace('info string ', '')}")
        print( "  Scores will be meaningless (handcrafted eval only). Aborting.")
        return False
    else:
        # readyok came but no load message — engine may have silently ignored it
        print(f"? ({elapsed*1000:.0f} ms) — no load confirmation from engine")
        print( "  Continuing, but scores may be wrong if model was not read.")
        return True   # let the user decide

# ── main bench ─────────────────────────────────────────────────────────────

def run_bench(engine_path: str, depths: list[int], nnue_path: str | None):
    engine = Path(engine_path)
    if not engine.exists():
        sys.exit(f"[ERROR] Engine binary not found: {engine_path}\n"
                 f"        Build it first:  g++ -O3 -std=c++17 -o nnue_engine nnue_engine.cpp")

    print(f"\n{'═'*68}")
    print(f"  nnue_engine bench")
    print(f"  engine : {engine.resolve()}")
    print(f"  model  : {nnue_path or '(none — handcrafted eval)'}")
    print(f"  depths : {depths}")
    print(f"  positions: {len(POSITIONS)}")
    print(f"{'═'*68}\n")

    # Start engine
    proc = subprocess.Popen(
        [str(engine.resolve())],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,   # capture stderr so engine errors surface
        text=True,
        bufsize=1,
    )

    # Initial UCI handshake
    send(proc, "uci")
    read_until(proc, "uciok", timeout=5)

    # ── model loading ──────────────────────────────────────────────────────
    if nnue_path:
        ok = load_model(proc, nnue_path)
        if not ok:
            proc.terminate()
            sys.exit(1)
    else:
        send(proc, "isready")
        read_until(proc, "readyok", timeout=5)
        print("  [WARN] No model specified. Running with handcrafted eval (weak).")
        print("         Pass --nnue checkpoints/model.nnue for real scores.\n")

    print()

    # ── per-depth loop ─────────────────────────────────────────────────────
    col_w = 10
    header = (
        f"  {'Position':<14}  {'Time(s)':>{col_w}}  "
        f"{'Nodes':>{col_w}}  {'NPS':>{col_w}}  {'Depth':>5}  {'Score':>8}"
    )
    sep = "  " + "─" * (len(header) - 2)

    depth_totals: dict[int, dict] = {}

    for depth in depths:
        totals = {"wall_ms": 0, "nodes": 0, "nps_sum": 0, "n": 0}

        print(f"{'─'*68}")
        print(f"  Depth {depth}")
        print(f"{'─'*68}")
        print(header)
        print(sep)

        for name, fen in POSITIONS:
            send(proc, "ucinewgame")
            send(proc, f"position fen {fen}")

            wall_start = time.monotonic()
            send(proc, f"go depth {depth}")
            lines = read_until(proc, "bestmove", timeout=300)
            wall_ms = (time.monotonic() - wall_start) * 1000

            info   = parse_info(lines)
            t_ms   = info.get("time",  int(wall_ms))
            nodes  = info.get("nodes", 0)
            nps    = info.get("nps",   0)
            reached= info.get("depth", depth)

            # Score display
            if "score mate" in info:
                score_str = f"M{info['score mate']:+d}"
            elif "score cp" in info:
                score_str = f"{info['score cp']:+d} cp"
            else:
                score_str = "—"

            # Warn if score looks like no-model (always 0 cp)
            no_eval_warn = ""
            if info.get("score cp", None) == 0 and nodes > 100:
                no_eval_warn = " (?)"

            totals["wall_ms"] += wall_ms
            totals["nodes"]   += nodes
            totals["nps_sum"] += nps
            totals["n"]       += 1

            t_s = t_ms / 1000
            print(
                f"  {name:<14}  {t_s:>{col_w}.3f}  "
                f"{fmt_num(nodes):>{col_w}}  {fmt_num(nps):>{col_w}}  "
                f"{reached:>5}  {score_str:>8}{no_eval_warn}"
            )

        n = totals["n"]
        avg_nps  = totals["nps_sum"] // max(n, 1)
        total_s  = totals["wall_ms"] / 1000

        print(sep)
        print(
            f"  {'TOTAL':<14}  {total_s:>{col_w}.3f}  "
            f"{fmt_num(totals['nodes']):>{col_w}}  {fmt_num(avg_nps):>{col_w}}"
        )
        print()

        depth_totals[depth] = {
            "total_s": total_s,
            "nodes":   totals["nodes"],
            "avg_nps": avg_nps,
        }

    # ── summary table ──────────────────────────────────────────────────────
    print(f"{'═'*68}")
    print("  SUMMARY")
    print(f"{'═'*68}")
    print(f"  {'Depth':>6}  {'Total time':>12}  {'Total nodes':>12}  {'Avg NPS':>10}")
    print("  " + "─" * 48)
    for d, v in depth_totals.items():
        print(
            f"  {d:>6}  {v['total_s']:>11.2f}s  "
            f"{fmt_num(v['nodes']):>12}  {fmt_num(v['avg_nps']):>10}"
        )

    # ── effective branching factor ─────────────────────────────────────────
    dl = sorted(depth_totals)
    if len(dl) >= 2:
        print(f"\n  Effective branching factor (EBF):")
        for i in range(1, len(dl)):
            d0, d1 = dl[i-1], dl[i]
            n0 = depth_totals[d0]["nodes"]
            n1 = depth_totals[d1]["nodes"]
            if n0 > 0 and n1 > 0:
                ebf = (n1 / n0) ** (1 / (d1 - d0))
                flag = "  ← high, check pruning" if ebf > 4 else ""
                print(f"    depth {d0:>2} → {d1:>2}:  EBF = {ebf:.2f}{flag}")
    print(f"{'═'*68}\n")

    proc.stdin.write("quit\n")
    proc.stdin.flush()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def main():
    parser = argparse.ArgumentParser(description="nnue_engine speed diagnostic")
    parser.add_argument("--engine", default="./nnue_engine",
                        help="Path to compiled nnue_engine binary (default: ./nnue_engine)")
    parser.add_argument("--depths", default="4,6,8,10,12",
                        help="Comma-separated depths to test (default: 4,6,8,10,12)")
    parser.add_argument("--nnue", default="checkpoints/model.nnue",
                        help="Path to .nnue weights file (default: checkpoints/model.nnue)")
    args = parser.parse_args()

    depths = [int(d.strip()) for d in args.depths.split(",") if d.strip()]
    if not depths:
        sys.exit("No valid depths specified.")

    run_bench(args.engine, depths, args.nnue)


if __name__ == "__main__":
    main()
