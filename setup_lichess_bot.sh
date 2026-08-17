#!/bin/bash
# Sets up the lichess-bot framework (https://github.com/lichess-bot-devs/lichess-bot,
# AGPLv3, a separate tool — not part of this project, see README.md and LICENSE.md)
# to run nnue_engine as a Lichess bot.
#
# What this does NOT do: create a Lichess account, upgrade it to a BOT account, or
# generate an API token. Those require your own Lichess login and are documented in
# README.md's "Publish on Lichess" section — do those first, then run this script.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [ ! -x "./nnue_engine" ]; then
    echo "error: ./nnue_engine not found or not executable — build it first:" >&2
    echo "  clang++ -O3 -march=native -flto -std=c++17 -DNDEBUG -pthread -o nnue_engine nnue_engine.cpp" >&2
    exit 1
fi
if [ ! -f "./checkpoints/model.nnue" ]; then
    echo "error: ./checkpoints/model.nnue not found — see README.md for how to obtain/generate it." >&2
    exit 1
fi

if [ ! -d "lichess-bot" ]; then
    echo "Cloning lichess-bot..."
    git clone --depth 1 https://github.com/lichess-bot-devs/lichess-bot.git lichess-bot
fi

if [ ! -d "lichess-bot/venv" ]; then
    echo "Creating Python venv and installing lichess-bot's dependencies..."
    python3 -m venv lichess-bot/venv
    lichess-bot/venv/bin/pip install --quiet --upgrade pip
    lichess-bot/venv/bin/pip install --quiet -r lichess-bot/requirements.txt
fi

if [ -f "lichess-bot/config.yml" ]; then
    echo "lichess-bot/config.yml already exists — leaving it as-is."
    echo "(If you moved this checkout, double-check the NNFile path under engine.uci_options is still correct.)"
else
    echo "Creating lichess-bot/config.yml from the template, wired to nnue_engine..."
    cp lichess-bot/config.yml.default lichess-bot/config.yml

    # Single Python pass over the whole file: config.yml.default's exact wording
    # can drift if lichess-bot is updated, so match structurally (line prefixes)
    # rather than chaining several fragile sed substitutions against exact text.
    python3 - "$PROJECT_ROOT" << 'PYEOF'
import re, sys
root = sys.argv[1]
path = f"{root}/lichess-bot/config.yml"
with open(path) as f:
    lines = f.readlines()

# Lines to drop outright: config.yml.default's own uci_options entries.
# nnue_engine doesn't implement any of them (unrecognized "setoption" names
# are silently ignored, so leaving them wouldn't break anything, but they'd
# be dead weight) — we insert our own NNFile/Hash/Threads instead, right
# after the "uci_options:" key.
DROP_PREFIXES = (
    "Move Overhead:", "Threads:", "Hash:", "SyzygyPath:", "UCI_ShowWDL:",
)

out = []
for line in lines:
    if line.startswith("token:"):
        out.append('token: "%LICHESS_BOT_TOKEN%"    # Overridden by the LICHESS_BOT_TOKEN env var — see README.md. Never put a real token in this file.\n')
        continue
    if re.match(r'^\s*dir:\s*"\./engines/"', line):
        out.append('  dir: ".."                        # nnue_engine lives one directory up (the project root).\n')
        continue
    if re.match(r'^\s*name:\s*"engine_name"', line):
        out.append('  name: "nnue_engine"              # Binary name of the engine to use.\n')
        continue
    if re.match(r'^\s*ponder:\s*true', line):
        out.append('  ponder: false                    # nnue_engine does not implement pondering.\n')
        continue

    m = re.match(r'^(\s*)uci_options:', line)
    if m:
        indent = m.group(1)
        out.append(line)
        out.append(f'{indent}  NNFile: "{root}/checkpoints/model.nnue"  # Absolute path — must be updated if you move this checkout.\n')
        out.append(f'{indent}  Hash: 512\n')
        out.append(f'{indent}  Threads: 1  # Threads>1 (Lazy SMP) is implemented but not yet SPRT-confirmed — see CLAUDE.md.\n')
        continue

    if any(line.strip().startswith(p) for p in DROP_PREFIXES):
        continue

    out.append(line)

with open(path, "w") as f:
    f.writelines(out)
print(f"Wrote engine wiring into {path}")
PYEOF
fi

echo ""
echo "Sanity-checking the engine wiring (no Lichess connection needed for this part)..."
"$PROJECT_ROOT/lichess-bot/venv/bin/python3" - << PYEOF
import chess, chess.engine, yaml
with open("$PROJECT_ROOT/lichess-bot/config.yml") as f:
    cfg = yaml.safe_load(f)
eng = cfg["engine"]
import os
path = os.path.join(eng["dir"], eng["name"])
engine = chess.engine.SimpleEngine.popen_uci(path, cwd="$PROJECT_ROOT/lichess-bot")
engine.configure(eng["uci_options"])
move = engine.play(chess.Board(), chess.engine.Limit(time=0.5)).move
engine.quit()
print(f"OK — engine responded with {move} using the config as written.")
PYEOF

echo ""
echo "Setup complete. Remaining manual steps (see README.md 'Publish on Lichess'):"
echo "  1. export LICHESS_BOT_TOKEN=\"your-bot-scoped-token\""
echo "  2. cd lichess-bot && ./venv/bin/python3 lichess-bot.py"
