import subprocess
import time
import re

# Path to your compiled executable
ENGINE_PATH = "engine/./engine"

# Test positions (FEN format)
TEST_POSITIONS = {
    "Startpos": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "Middlegame": "r3k2r/p1pp1ppp/1b3nb1/1B6/1P0nP3/1Q4N1/P1PP1PPP/R1B2RK1 w kq - 0 1",
    "Endgame": "8/2p5/4k3/1p6/8/8/3K4/8 w - - 0 1"
}

def send_command(process, command):
    process.stdin.write(f"{command}\n")
    process.stdin.flush()

def read_until_bestmove(process):
    output_lines = []
    while True:
        line = process.stdout.readline()
        if not line:
            break
        output_lines.append(line.strip())
        if line.startswith("bestmove"):
            break
    return output_lines

def benchmark_engine(depths=[6, 10, 14]):
    print("=" * 65)
    print(f"{'Position':<12} | {'Depth':<6} | {'Time (ms)':<10} | {'Nodes':<10} | {'NPS':<10}")
    print("=" * 65)

    # Start engine process
    engine = subprocess.Popen(
        [ENGINE_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # Initialize UCI
    send_command(engine, "uci")
    send_command(engine, "isready")
    
    # Wait for readyok
    while True:
        line = engine.stdout.readline().strip()
        if line == "readyok":
            break

    for pos_name, fen in TEST_POSITIONS.items():
        for d in depths:
            send_command(engine, "ucinewgame")
            send_command(engine, f"position fen {fen}")
            
            start_time = time.perf_counter()
            send_command(engine, f"go depth {d}")
            
            lines = read_until_bestmove(engine)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Parse search output info for nodes if available
            nodes = "N/A"
            nps = "N/A"
            for line in reversed(lines):
                if line.startswith("info"):
                    node_match = re.search(r'\bnodes\s+(\d+)', line)
                    nps_match = re.search(r'\bnps\s+(\d+)', line)
                    if node_match:
                        nodes = node_match.group(1)
                    if nps_match:
                        nps = nps_match.group(1)
                    break

            print(f"{pos_name:<12} | {d:<6} | {elapsed_ms:<10.2f} | {nodes:<10} | {nps:<10}")

    send_command(engine, "quit")
    engine.terminate()

if __name__ == "__main__":
    benchmark_engine(depths=[4, 8, 12])
