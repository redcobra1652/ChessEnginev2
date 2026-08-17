# nnue_engine

A self-contained C++ UCI chess engine using a custom-trained HalfKP NNUE for
evaluation (H=256, L1=32, L2=32, 8 output buckets), paired with a
Stockfish-style alpha-beta search (magic bitboards, PVS, aspiration windows,
null-move pruning, LMR, reverse futility pruning, ProbCut, internal iterative
reduction, singular extensions, late move pruning, SEE pruning,
capture/continuation/countermove/correction history, and Lazy SMP
multithreading). The net is trained from scratch via `train.py` /
`model.py` / `serialize.py` on Lichess cloud-eval positions and benchmarked
against Stockfish via `fastchess`/SPRT.

For the full development history, every measured result, and the reasoning
behind every design decision in this codebase, see **CLAUDE.md** — it's the
project's running lab notebook and is kept far more current and detailed
than this file. This README covers just what you need to build, run, and
publish the engine.

## Building

```
clang++ -O3 -march=native -flto -std=c++17 -DNDEBUG -pthread -o nnue_engine nnue_engine.cpp
```

Single file, no external dependencies beyond a C++17 standard library and
pthreads. Compiles cleanly with zero warnings under `-Wall -Wextra`.

## Running

`nnue_engine` speaks standard UCI. It needs a trained network file, pointed
at via the `NNFile` UCI option:

```
echo -e "uci\nsetoption name NNFile value $(pwd)/checkpoints/model.nnue\nisready\nposition startpos\ngo movetime 1000\n" | ./nnue_engine
```

Point any UCI-compatible GUI (Arena, CuteChess, Nibbler, etc.) or CLI
tournament tool at the binary the same way any other UCI engine is used;
just make sure `NNFile` gets set before the first `go`.

### UCI options

| Option | Default | Notes |
|---|---|---|
| `NNFile` | *(none)* | Path to the trained `.nnue` weights file. Required. |
| `Hash` | 128 | Transposition table size in MB. |
| `Threads` | 1 | Lazy SMP worker count. See CLAUDE.md's multithreading section for the current SPRT status before relying on `Threads > 1` for anything you care about the result of. |

### A few debug-only UCI commands (purely additive, don't affect normal play)

- `eval` — prints the raw static NNUE eval of the current position, no search.
- `perft <depth>` / `perft divide <depth>` — movegen regression test.
- `prunestats` / `prunestats reset` — pruning firing-rate instrumentation, gated behind the `NNUE_PRUNE_DEBUG` env var.

## Training a network

1. `positions.bin` — a binary-packed dataset of (position, Stockfish cloud-eval, game result) tuples. `convert.py`/`convert_lichess.py` build this from raw Lichess data; `data.py` defines the HalfKP feature encoding used both at train time and (independently, for train/serve parity) in `nnue_engine.cpp`.
2. `python3 train.py` — trains the float model (`model.py`), checkpoints to `checkpoints/nnue_best.pt`.
3. `python3 serialize.py checkpoints/nnue_best.pt checkpoints/model.nnue` — quantizes the float checkpoint into the int8/int16 format the C++ engine loads. Run `python3 verify_quantization.py` afterward — it checks byte-exactness against `serialize.py`'s own quantization math, per-layer clipping rates, and quantized-vs-float eval agreement, and it has caught a real, severe quantization bug before (see CLAUDE.md).

## Testing

- `python3 -m pip install -r <none needed — bench.py/tournament.py use only python-chess>` — `pip install chess` if you don't already have it.
- `./fastchess` (built from `fastchess_src/`, a vendored copy of [Disservin/fastchess](https://github.com/Disservin/fastchess)) drives SPRT-style engine-vs-engine testing; `openings.epd` is the opening book used. See CLAUDE.md's "SPRT harness" section for example commands and the operational lessons learned running them (always rebuild `nnue_engine_baseline` after a change lands, always run a 40-game sanity check before a real SPRT, etc.).
- `bench.py`/`tournament.py` — Python-based benchmarking against a local Stockfish build. Stockfish (GPLv3) is not vendored in this repository — build or download it yourself into `stockfish/` (same treatment as `fastchess`/`lichess-bot`: a separate tool this project tests against, not part of it). CLAUDE.md documents why fastchess-based SPRT results are more trustworthy than this path's numbers.

## Publishing on Lichess

This repo includes a setup script that wires up
[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) (the
standard, actively-maintained framework the Lichess team recommends for
running any UCI/XBoard engine as a bot) to drive `nnue_engine`.
**lichess-bot is a separate project under its own license (AGPLv3) — it is
not part of this repository, isn't committed here, and isn't covered by
this project's `LICENSE.md`.** The setup script clones it fresh into
`lichess-bot/` (gitignored).

### 1. Create a Lichess bot account

Bot accounts on Lichess are separate from normal accounts and **can never
play rated games as a human would** — use a fresh account you don't mind
dedicating to this, not your main account. Sign up at lichess.org, then
generate a personal API token at
`https://lichess.org/account/oauth/token/create` with at least the
`bot:play` scope.

Upgrade the account to BOT status (this is **irreversible** and only works
on an account that has never played a game):

```
curl -d '' https://lichess.org/api/bot/account/upgrade -H "Authorization: Bearer YOUR_TOKEN"
```

### 2. Build the engine and run the setup script

```
clang++ -O3 -march=native -flto -std=c++17 -DNDEBUG -pthread -o nnue_engine nnue_engine.cpp
./setup_lichess_bot.sh
```

The script clones `lichess-bot`, creates a Python venv, installs its
dependencies, and writes `lichess-bot/config.yml` with `engine.dir`,
`engine.name`, and `uci_options.NNFile` already pointed at this checkout's
`nnue_engine` binary and `checkpoints/model.nnue`, `Threads` conservatively
left at 1, and `ponder` disabled (this engine doesn't implement pondering).
It also runs a quick standalone sanity check — spawning the engine exactly
the way lichess-bot will and requesting one move — so a broken build or bad
path is caught before you ever touch the Lichess API. Safe to re-run;
it won't overwrite an existing `config.yml`, so any edits you make there
afterward (challenge-acceptance rules, resign/draw policy, etc. — see
[lichess-bot's own config documentation](https://github.com/lichess-bot-devs/lichess-bot/wiki)
for the full option set) are preserved.

### 3. Run the bot

```
export LICHESS_BOT_TOKEN="your-bot-scoped-token"
cd lichess-bot
./venv/bin/python3 lichess-bot.py
```

It'll log in as your bot account and start accepting challenges per
whatever rules are configured in `lichess-bot/config.yml`'s `challenge:`
section (defaults are conservative — review them before leaving the bot
running unattended). Leave the process running for as long as you want the
bot online; `Ctrl+C` to stop (or set `quit_after_all_games_finish: true` in
the config to finish in-progress games first).

## License

See `LICENSE.md`. In short: you're welcome to play against this engine or
use the compiled binary, but the source code itself is not open for reuse,
copying, or redistribution, and this repository is only ever distributed
through the author's own GitHub page or the author directly.
