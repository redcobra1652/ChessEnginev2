# nnue_engine — project notes

A self-contained C++ UCI chess engine (`nnue_engine.cpp`) using a custom-trained
HalfKP NNUE (`checkpoints/model.nnue`, H=256, L1=32, L2=32, 8 output buckets)
for evaluation, paired with a Stockfish-style alpha-beta search. Trained via
`train.py`/`model.py`/`serialize.py` on `positions.bin` (~256M positions,
Lichess cloud-eval labels at depth ≥20). Benchmarked/tournament-tested against
Stockfish via `tournament.py` and `bench.py`.

**Current status: ~2683 Elo ± 17 (measured via `tournament.py` against
Stockfish@2750, 100 games, after fixing a `ucinewgame`-per-game bug in the
harness — see "Session follow-up" below), consistent with the earlier
~2600 estimate. Goal: ~3000 Elo, i.e. roughly 320 Elo remaining. Read
"Session follow-up: harness fix, code audit, and NNUE scale diagnostic"
below before doing anything else — it supersedes some of this document's
earlier conclusions (notably: cross-engine depth comparisons are invalid,
see that section) and lists concrete, verified next actions with their
evidence.**

## Baseline before this work

`nnue_engine_new.cpp` was already a near-complete Stockfish-style engine:
magic bitboards, PVS, aspiration windows, null-move pruning, LMR, reverse
futility pruning, ProbCut, internal iterative reduction, singular extensions,
late move pruning, SEE pruning, continuation/capture/countermove history, and
a NEON-vectorized NNUE forward pass. Reported playing strength ~2550–2650 Elo,
depth 12 in ~0.4–0.8s on an average middlegame position.

## What was fixed in `nnue_engine.cpp`

Correctness/stability bugs found in the baseline and fixed:

1. **Singular extension was unsound.** It re-searched the *same, unmade*
   position (no `do_move`), so its "verification" search always self-hit the
   TT entry it was trying to verify and returned instantly — extensions never
   actually fired, and a spurious, unverified cutoff fired instead whenever
   `tt_value ≥ beta + 2·depth`. Rewrote with a real `excludedMove` parameter:
   the verification search runs at the same ply/board state with the TT
   cutoff and TT store suppressed for that call, and the excluded move is
   skipped in the move loop. This is the standard design (mirrors Stockfish).
2. **Aborted searches were poisoning the TT.** A time-out mid-move-loop still
   unconditionally stored `best_score` at full `depth`; since the TT persists
   across moves until `ucinewgame`, this corruption could survive into later
   moves of the same game. Now gated on a `search_aborted` flag.
3. **Mate scores weren't ply-adjusted through the TT**, and quiescence's mate
   return ignored ply entirely (`-(MATE_SCORE)` flat). Added
   `score_to_tt`/`score_from_tt` conversions at every TT store/load site, and
   ply-adjusted the qsearch mate return.
4. **Unbounded quiescence recursion.** No ply cap existed, so a pathological
   run of checks in qsearch could in principle overrun the fixed-size
   `SearchStack` / `Accumulator` / `Board::history` arrays. Added a hard ply
   cap (`MAX_PLY - 2`) in both `alpha_beta` and `quiescence`, mirroring
   Stockfish's `ss->ply >= MAX_PLY` leaf cutoff.
5. **`improving` heuristic read stale/invalid data.** It used `staticEval !=
   0` as a "did we compute an eval here" test, but 0 is also a legitimate
   eval and in-check/early-returned nodes left it at that same default.
   Introduced an `EVAL_NONE` sentinel used consistently.

Node-count / search-quality additions:

- Quiescence now probes and stores the TT (it previously had none).
- Mate-distance pruning at the top of `alpha_beta`.
- Static eval refined by a compatible TT bound when not near mate scores
  (Stockfish-style "trust the TT over a fresh eval when it's tighter").
- NNUE forward pass skipped entirely when in check (nothing downstream used
  it there anyway — RFP/NMP/ProbCut/futility are all gated on `!in_check`).

Per-node speed:

- Vectorized the NNUE L2 and output layers with NEON (previously scalar).
  Verified bit-exact against the unvectorized path (integer addition is
  exactly associative, so reordering the summation cannot change the result).

All changes through that point preserved the original UCI interface
byte-for-byte (verified by diff). **Update:** a follow-up session added one
new, purely additive UCI debug command, `eval` (prints the raw static NNUE
eval of the current position, no search — see "Session follow-up" below) —
this is the one exception to byte-for-byte preservation, and it does not
alter the behavior of any existing command. Build: `clang++ -O3
-march=native -flto -std=c++17 -DNDEBUG -o nnue_engine nnue_engine.cpp`
(Apple Silicon) — compiles with zero warnings under `-Wall -Wextra`.

## Measured effect (nodes/depth, engine-internal)

- At depth 7–9 (below the depth-8 singular-extension threshold), the fixed
  engine visits 6–20% fewer nodes than baseline — the qsearch TT / mate-distance
  pruning / eval-refinement work is a clean win there.
- At depth 12+, results are mixed: the now-functioning singular extension
  spends real nodes on verification searches it previously skipped (via its
  bug), so some positions show more total nodes at fixed depth. This is the
  expected, standard trade-off — singular extensions cost nodes for better
  move selection in essentially every engine that implements them correctly.
- Depth reached in a fixed time budget is roughly a wash vs. baseline.

## Benchmark methodology — read this first

**This is the most important finding of the follow-up diagnostic pass, and
almost certainly explains why Elo barely moved despite real search fixes.**

`tournament.py`'s `run()` limits the two engines completely differently:

```python
nnue_limit = chess.engine.Limit(depth=nnue_depth)       # default depth = 7
sf_limit   = chess.engine.Limit(time=sf_movetime_ms/1000)  # default 50 ms/move
```

Your engine is capped by **fixed search depth** (default `--depth 7`);
Stockfish is capped by **fixed time per move** (default `--sf-movetime 50`).
Measured this session: your engine reaches depth 7 in roughly **20–50ms** on a
typical position — well *under* Stockfish's 50ms/move budget. That means:

- Every speed improvement made this session (NEON vectorization, node-count
  reductions from qsearch TT / mate-distance pruning / eval-refinement) is
  **invisible to this benchmark**, because the engine stops at depth 7
  regardless of how fast it got there. Only get there *faster*; the benchmark
  doesn't let it search *deeper* with the time saved.
- The ~2600 measurement almost certainly reflects only the *move-quality*
  fixes (singular extension no longer unsound, TT no longer corrupting) at a
  shallow, fixed depth — not any of the efficiency work.
- Stockfish, by contrast, is using its full 50ms budget every move via
  iterative deepening, reaching whatever depth it can in that time — a
  fundamentally different (and much more favorable) resource allocation.

**Recommended first action for the next session, before any further engine
changes:** re-run `tournament.py` with your engine also time-limited to
something comparable to Stockfish's per-move budget (`python-chess`'s
`chess.engine.Limit` supports `time=` directly — swap `nnue_limit` from
`depth=` to `time=` to match `sf_movetime_ms`, or at minimum raise `--depth`
substantially, e.g. to 12–14, since ~800ms–1.2s at depth 12 is a much more
realistic per-move budget). Measure Elo again before concluding whether
further code changes are even necessary — it's plausible a large chunk of the
gap to 3000 closes from this alone, since it was never actually tested.

## Eval/search speed diagnostics (this session)

Isolated NNUE forward-pass throughput (no movegen, no search — pure eval
calls in a tight loop, measured via a throwaway instrumented build, not part
of `nnue_engine.cpp`):

- **~213 ns/eval, ~4.6M evals/sec**, consistent across opening/middlegame/
  endgame test positions.

Real single-threaded search NPS, Hash=128MB, depth 12, same hardware:

| | Nodes/sec |
|---|---|
| Your engine | ~1.6–2.2M |
| Stockfish 18 (`bench 128 1 12`, Threads=1) | ~1.94M |

Both land in the same ballpark despite Stockfish's network being far larger
(dual big/small nets) and its search being over a decade more tuned — so raw
per-node overhead (movegen, legality filtering, eval) is **not** what's
holding your engine back relative to Stockfish. This reinforces the
benchmark-methodology finding above: the bottleneck to close is either how
search time is actually allocated in testing, or move/search quality — not
raw throughput.

## Elo ceiling estimate for `checkpoints/model.nnue`

This is an informed estimate from architecture/training analysis, not a
measured number — only actual games give a real figure, and per the
methodology note above, the current ~2600 measurement almost certainly
*understates* what this net can support at a fairer time budget.

Your net's shape — HalfKP, H=256, L1=32, L2=32 — is **the same architecture
as Stockfish's very first NNUE net** (Sept 2020, `nn-c157e0a5755b.nnue`). That
exact-shape net, paired with Stockfish 12's search, reached roughly
**3450–3550 Elo** (CCRL-style scale) — the realistic upper bound for this
architecture.

Your training loss (0.0073, MSE on `sigmoid(cp/410)` targets blended 70%
Stockfish-eval / 30% game result, no held-out validation split in `train.py`)
corresponds to an RMSE of roughly 8.5 percentage points of win-probability
error — a solid number. Two things likely keep the net below the historical
3450–3550 ceiling:

- Stockfish's original net was trained on a **self-play reinforcement
  flywheel** (successive generations of engine-generated data at huge scale);
  yours is a **single-generation supervised fit** against external Lichess
  cloud-eval labels (Stockfish depth ≥20 — still high quality, but not the
  same iterative process).
- No validation split means 0.0073 is a training-set loss. With ~256M
  positions and a fairly small net (256-wide) this is more likely
  underfitting than overfitting, which is reassuring, but there's currently
  no way to confirm generalization from the number alone.

**Estimated ceiling: roughly 2900–3300 Elo** for this net paired with a
correctly-functioning search at a realistic time budget. This is the basis
for saying **the net is very likely not the current bottleneck** — the gap
between that ceiling and the measured ~2600 is much better explained by the
benchmark-methodology issue above (and possibly remaining search/engineering
maturity gaps) than by the network itself.

## NNUE embedding — investigated, not worth doing for speed

Stockfish embeds its `.nnue` weights directly in the executable (`incbin` at
compile time) and runs with **zero external files** (verified: its 113.8MB
binary works standalone from an empty directory). Measured on this engine:
loading `checkpoints/model.nnue` (21MB) at startup costs **~20–30ms total
process startup time**, and `tournament.py` calls `popen_uci` **once per
tournament run**, outside the game loop — not once per game. That cost is
already paid once and is negligible against real game/tournament time.
Embedding would not speed up search or evaluation at all: by the time any
search runs, the weights are already fully resident in memory as static
arrays, identical whether they arrived via `ifstream`+`memcpy` or a
compiled-in data segment. Only worth it for single-file deployment
convenience, not performance. **Do not spend time on this to chase Elo.**

## Session follow-up: harness fix, code audit, and NNUE scale diagnostic

### Harness fix (done)

`tournament.py`'s `play_game()` called `engine.play(board, limit)` with no
`game=` argument. python-chess's `UciProtocol.play()` only sends
`ucinewgame` when `game != self.game` (or on the very first call ever) —
since `game` was `None` on every call, **`ucinewgame` was sent once for the
entire tournament, not once per game.** TT and every history table
(`g_main_history`, `g_cont_history`, `g_capture_history`, `g_countermoves`,
`g_killers`) carried stale state across all 100 games in a run. Fixed by
passing `game=game_num`. Re-ran after the fix: 100 games vs Stockfish@2750 →
W33/L52/D15, **score 40.5%, Elo≈2683±17** — consistent with the pre-fix
~2600 estimate, which is expected: this fix was about measurement
trustworthiness, not engine strength, so it shouldn't and didn't move the
number.

### Cross-engine nominal-depth comparison is invalid — do not use it again

A natural diagnostic ("why does Stockfish reach depth 16–18 while we don't,
at similar NPS?") turned out to have an invalid premise. Measured node
counts to reach fixed nominal depths (`go depth N`), same hardware, same
Hash/Threads:

```
             depth=8      depth=10     depth=12     depth=14
startpos      26x          135x          32x          152x
italian_mg    21x           52x          46x           31x
complex_mg   209x          373x         608x          325x
endgame      1.9x          4.0x         9.2x          7.3x
```
(ratio = this engine's nodes / Stockfish's nodes to complete the same
nominal depth). The ratio is non-monotonic with depth on the same position —
the signature of dividing two incommensurable quantities, not a real trend.
Stockfish completing nominal depth 12 on `complex_mg` in 4,073 nodes is not
a genuine depth-12 search by any classical definition; modern Stockfish's
reductions subtract directly from the depth counter, so its nominal depth
has decoupled from actual tree depth. **"Depth 18" in Stockfish and "depth
18" here are not the same unit — don't compare them again.** This engine's
own node counts (e.g. 1.3M nodes for depth 12 at startpos) are ordinary for
a classical alpha-beta search. The `endgame` row is the only one that
behaves sanely (grows smoothly with depth) — few pieces means few
reductions, so depth semantics converge there; it suggests a real but modest
~2–9x node-efficiency gap in simple positions, not the huge numbers seen at
other depths in the other rows.

If a cross-engine search-quality comparison is needed later, use a fixed
tactical test suite (e.g. WAC) at equal time-per-position and compare
solved-count — that measures search quality in units both engines share.
Fixed-depth node counts do not.

### Bugs found this pass, with impact estimates

- **Blanket, ungated check extension** (`nnue_engine.cpp:2365`,
  `2385`). Every move that gives check gets an unconditional `+1` ply
  extension at every node, with no SEE gate and no per-line extension
  budget, and is also exempted from LMR. Verified by disabling it and
  re-running the fixed-depth node comparison (same-engine, so this
  comparison is valid unlike the cross-engine one above):
  ```
  italian_mg   depth=12  baseline=1,288,219  no-ext=704,534   (-45.3% nodes)
  italian_mg   depth=14  baseline=3,389,971  no-ext=2,603,106  (-23.2% nodes)
  complex_mg   depth=12  baseline=2,475,403  no-ext=1,070,468  (-56.8% nodes)
  complex_mg   depth=14  baseline=4,142,278  no-ext=4,329,401  (+4.5% nodes)
  ```
  Real node savings (23–57%) in three of four cases, but `complex_mg` at
  depth 14 got *worse* with it removed, and node savings are not Elo —
  removing check extensions can lose tactical accuracy even while cutting
  nodes. **Needs SPRT-style game testing to decide, not node counts.**
  Worth gating with an SEE check and/or a per-line budget as the concrete
  next step, then measuring via games.

- **No correction history.** Confirmed absent from the file entirely (no
  pawn/material static-eval correction of any kind). Stockfish's own
  testing puts this around **+20–30 Elo**. The one genuinely valid point
  from an outside AI's review this session (see "AI-generated review
  fact-check" below) — worth implementing.

- **`quiescence()` never polls `out_of_time()`.** Confirmed via grep: the
  clock is checked in `alpha_beta` (`2055`, `2423`) and the iterative
  deepening loop, but never inside `quiescence()` (`1916–2030`), even
  though it recurses on itself (`2012`). `g_nodes` does increment inside
  qsearch and control returns to the parent `alpha_beta` check after each
  move, so this is not a proven hang — no qsearch subtree has been shown to
  fan out wide enough to matter between polls. Treat as **"worth a bounded
  check as cheap insurance,"** not as a confirmed live bug, until someone
  demonstrates an actual overrun. Matters only for real-clock games, not
  the current depth/movetime-limited benchmark.

### Verified clean — ruled out, so don't re-investigate these

- **Transposition table**: full 64-bit key match (no truncated-signature
  collisions), sane cluster-based replacement (age + depth weighted
  eviction, `nnue_engine.cpp:1655–1676`), correct per-search age increment.
- **Internal Iterative Reduction**: present and correct (`2121–2122`).
- **History gravity formula**: `hist_update` (`1711–1712`) uses the
  standard `h += bonus - h*|bonus|/HIST_MAX` self-bounding update — no
  int16 overflow risk from accumulating across a long game.
- **Bucket formula train/serve parity**: `data.py:60–62`
  (`num_pieces * num_buckets // 32`) and `nnue_engine.cpp:1910–1913`
  (`n * num_buckets / 32` on the cached non-king piece count) use the
  identical formula on the identical piece-count convention. This rules out
  bucket-selection skew, the scariest class of silent NNUE bug (produces no
  crash, just a permanently miscalibrated eval) — **confirmed not the
  issue.** Note this formula only needs to be self-consistent between
  training and inference, which it is; it does not need to match
  Stockfish's own bucket formula (`(pieceCount - 1) / 4` on an
  all-pieces-including-kings count) — different engines, different nets,
  independently self-consistent conventions. Both are the same general
  idea: a handful of material-range-selected output layers sharing one
  trunk, so the net doesn't need one giant layer to represent all game
  phases.

### AI-generated code review fact-check

A pasted third-party AI review claimed 6 gaps. Verified against the code:
5 of 6 are false, 1 is valid.

| Claim | Verdict | Evidence |
|---|---|---|
| "Lack Singular Extensions" | **False** | Full impl incl. double-extension, `2250–2275` |
| "cont/capture history not integrated into ordering" | **False** | Ordering use `1843`; cutoff updates `2458, 2465` |
| "LMR reduction is static/simple" | **False** | Stat-score/cut-node/PV adjusted, `2387–2399` |
| "NMP reductions primitive" | **False** | `R` is depth- and eval-margin-dependent, `2170` |
| "Need improving/non-improving flags" | **False** | Already wired into RFP/LMP/ProbCut, `2163, 2321, 2184` |
| "Need Pawn/Correction History" | **True** | See "Bugs found" above |

### NNUE eval-scale diagnostic (`net_eval_diagnostic.py`, new script)

New standalone script (does not touch `train.py`, per explicit instruction)
that compares the engine's static eval (new `eval` UCI debug command, no
search) against a Stockfish-search-derived, game-result-blended target —
same blend formula as `train.py`'s `nnue_loss` (`lam=0.7` default) — on 800
positions sampled from `games/vs_sf2750.pgn` (real played games, independent
of the `positions.bin` training set). Run via `python3
net_eval_diagnostic.py --samples 800 --sf-movetime 250`.

**Headline finding — a real, measured centipawn-scale mismatch, cause
identified:**

| Source | median &#124;cp&#124; | mean &#124;cp&#124; |
|---|---|---|
| `positions.bin` training labels (Lichess cloud-eval) | 42 | 145 |
| This engine's net, static eval, held-out positions | 66 | 87 |
| This repo's bundled Stockfish binary, `go movetime 250` | 286 | 506 |

The net's output tracks its own training labels' scale (66 vs 42 — same
order of magnitude); the mismatch is between the Lichess cloud-eval cp
convention the training labels were generated under and this specific
(modern) Stockfish binary's cp normalization, which reports considerably
larger magnitudes for comparable positions (Stockfish has changed its
internal cp-to-win-rate normalization across versions). **The net is
correctly calibrated to what it was trained on** — this is not a
generalization failure, and the earlier regression "slope=0.072" number
from this same measurement is not a reliable summary (it's dominated by a
handful of extreme-magnitude Stockfish outlier scores against a
range-bounded net output; the three median numbers above are the trustworthy
summary). Do not read the script's raw MAE (452cp) / MSE (0.068) /
per-bucket error gradient as net-quality or generalization figures — they
mostly measure this convention gap, amplified in low-piece-count buckets
because endgame scores are naturally more decisive/extreme in Stockfish's
larger-magnitude scale. That per-bucket gradient is *not* evidence of
undertrained endgame buckets.

**Tested and ruled out: this does not explain the search node-count
findings above.** The natural hypothesis — "the engine's static-eval pruning
margins (RFP 234, futility 100–400, NMP/ProbCut constants) were ported from
real Stockfish source assuming Stockfish's cp scale, so if the net's eval is
on a smaller scale those margins are effectively too large and rarely
trigger" — was tested directly with instrumented counters (temporary build,
not merged) on `complex_mg` at depth 12:

```
nonleaf_nodes=1,667,869   rfp=292,525 (17.5%)   nmp=24,632 (1.5%)
probcut=68   futility_skips=307,101
```

RFP and futility are firing on a large, healthy fraction of applicable
nodes; NMP and ProbCut fire at normal, expected low rates for their narrow
trigger conditions. **The pruning is active, not inert — this hypothesis is
refuted by direct measurement.** The scale-convention mismatch is real and
worth understanding, but it is not the explanation for the earlier
node-count gap. That gap remains only partially explained (by the check
extension, ~25–57% at depth 12 on two of four test positions).

**On the Elo-ceiling estimate:** the previous session's 2900–3300 estimate
was an architecture analogy (same shape as Stockfish's first NNUE net), not
a measurement, and this pass doesn't license replacing it with a new
point number either — an eval-scale/RMSE number doesn't convert cleanly to
Elo without a calibration curve this project doesn't have. What changed:
the net is confirmed to correctly reproduce its own training distribution
(rules out one obvious failure mode), and the weakest joint in the original
2900–3300 argument — single-generation supervised fit on Lichess labels vs.
Stockfish's self-play flywheel — is unaffected by anything measured this
session. **The only fully trustworthy number remains the measured
full-system result: 2683 Elo (Stockfish's `UCI_Elo` scale) via real games.**
Treat 2900–3300 as an unverified prior, not a target to plan around.

## SPRT harness (built and working) + one CRITICAL unresolved bug found while testing it

The SPRT harness from item 1 below is **already built and verified working** —
not a to-do, a fact about this checkout:

- **`fastchess`** (binary at repo root: `./fastchess`) — built from source
  (`fastchess_src/`, `Disservin/fastchess` upstream, built via plain `make`,
  no cmake needed). `./fastchess -version` confirms it runs.
- **`openings.epd`** (49 positions) — a diverse opening book covering all
  major openings (Ruy Lopez, Sicilian variations, French, Caro-Kann, QGD/QGA,
  Nimzo/King's/Queen's Indian, English, Reti, etc.), generated by
  `gen_opening_book.py` from hardcoded standard opening theory (not scraped —
  regenerate/extend by editing the `LINES` dict and re-running).
- Confirmed end-to-end: engine launches under `fastchess`, loads
  `NNFile` via `option.NNFile=...`, uses the opening book, plays real games,
  and reports Elo/nElo/LOS — this was tested directly, not assumed.

**Example SPRT command** (A/B test a candidate change against the current
binary — copy the binary to a second path first, e.g. `cp nnue_engine
nnue_engine_baseline` before making a code change, then build the changed
version as `nnue_engine`):

```
./fastchess \
  -engine cmd=./nnue_engine          name=candidate option.NNFile=$(pwd)/checkpoints/model.nnue \
  -engine cmd=./nnue_engine_baseline name=baseline  option.NNFile=$(pwd)/checkpoints/model.nnue \
  -each proto=uci option.Hash=64 tc=8+0.08 timemargin=200 \
  -openings file=openings.epd format=epd order=random \
  -games 2 -repeat -rounds 2000 \
  -concurrency <N, see note below> \
  -sprt elo0=0 elo1=10 alpha=0.05 beta=0.05 \
  -pgnout file=sprt_result.pgn \
  -maxmoves 200
```
`elo0=0 elo1=10` tests "is this change worth at least ~5 Elo" (standard
non-regression/small-gain SPRT bounds) — adjust per what's being tested.
Set `-concurrency` conservatively (well under physical core count) given the
timing issue below — concurrency contention makes the overrun bug worse, not
just slower.

### CRITICAL — a real, only partially-fixed time-management bug; fix before trusting any SPRT result

Building the harness immediately surfaced a bug that would otherwise have
silently contaminated every SPRT run with spurious time losses unrelated to
move quality:

**Found and fixed this session:** `quiescence()` never called
`out_of_time()` (see "Bugs found" above, which has since been upgraded from
"unconfirmed" to **confirmed** — this is no longer a hypothesis). Under
`tournament.py`'s fixed-depth testing this was invisible; under `fastchess`'s
strict per-move time enforcement it caused real, repeated "loses on time"
forfeits — including at least one game that did not produce a move within a
5-minute wall-clock budget at `st=0.2`. Fix applied at
`nnue_engine.cpp:1926` (mirrors `alpha_beta`'s existing
`out_of_time()`-then-`return alpha` pattern). Rebuilt and verified via a
standalone test: 49/49 direct `go movetime 2000` calls (one per opening-book
position, no `fastchess`, no concurrency) returned `bestmove` within
2.00–2.06s — clean, no overruns, no stalls.

**Still unresolved — confirmed real, not a testing artifact:** even after
the fix above, with a *generous* budget (`st=2` = 2000ms, `timemargin=100`,
`-concurrency 1`, so no CPU contention), a real game under `fastchess`
produced `White loses on time (143ms overrun)` on game 1. This rules out
"just a tight budget" or "just concurrency contention" as the explanation —
those were tested and ruled out separately (concurrency=1, large margin).
**The standalone 49-position test above did not reproduce this**, which is
itself informative: the overrun appears to require a position reached
*through actual play* (deeper/more complex middlegame, or a search that has
accumulated real TT/history content across several moves in the same game),
not a fresh book-depth position probed in isolation.

**This must be root-caused and fixed before any SPRT result is trustworthy**
— a 100–150ms overrun on a several-second budget is large enough to cause
real spurious losses, and SPRT is specifically vulnerable to a systematic
bias like this (it looks exactly like "the engine playing worse," not like
noise, because it's a consistent, repeatable time forfeit, not a random
swing). Suggested starting point for whoever picks this up: this is not
"more guessing about which heuristic to disable" — instrument directly.
Add wall-clock timestamps (a) immediately when the `go` command is parsed,
(b) immediately before `Engine::search()` returns, and (c) immediately
before `std::cout << "bestmove ..."` prints, all to `std::cerr`. That
isolates whether the overrun is happening *inside* the polling loop (meaning
the 4096-node polling granularity, or one specific unguarded code path
between polls, is insufficient — check the aspiration-window retry loop and
the LMR fail-high re-search path for a chain of calls that could run long
between two poll points) versus *after* `search()` already returned (meaning
something in output/UCI-loop overhead, not the search itself, is the
culprit). Reproduce with: `./fastchess -engine cmd=./nnue_engine
option.NNFile=$(pwd)/checkpoints/model.nnue -engine cmd=./nnue_engine
option.NNFile=$(pwd)/checkpoints/model.nnue -each proto=uci st=2
option.Hash=64 timemargin=100 -openings file=openings.epd format=epd
order=random -rounds 10 -games 2 -repeat -concurrency 1 -maxmoves 80` and
watch for `loses on time` in the output — reproduced within the first game
in testing, so it shouldn't take long to catch again.

## Prioritized next steps toward ~3000 Elo

Status of the original list: item 1 (benchmark methodology) is **done** —
see "Session follow-up" above; the `ucinewgame` harness bug turned out to
matter more than the depth-vs-time mismatch, and re-measurement landed at
2683±17, close to the original ~2600 estimate. Items are renumbered/updated
below to reflect everything found this pass.

1. **Fix the remaining time-management overrun bug — see "SPRT harness"
   section above.** This is now step 0 in practice: the harness itself is
   already built and working, but this bug makes every result from it
   untrustworthy until fixed (spurious time losses look exactly like a real
   regression to SPRT, not like noise). Nothing below this is decidable
   until it's resolved.
2. **Use the SPRT harness (already built — `fastchess` + `openings.epd`,
   see above) for every change below.** This is not optional: correction
   history and check-extension gating are each individually in the 10–30
   Elo band, and the old `tournament.py`-based ±17 nominal margin was
   optimistic besides (games share openings from `startpos`, so they're
   correlated).
3. **Gate the check extension** (`nnue_engine.cpp:2365`) with an SEE check
   and/or a per-line extension budget, instead of unconditional +1 for every
   checking move. Confirmed real via node-count testing (23–57% fewer nodes
   in 3 of 4 cases) but needs the SPRT harness (item 1) to know if it costs
   or gains Elo — `complex_mg` depth 14 got worse without it, so this is not
   a free win.
4. **Add correction history.** Confirmed absent; ~20–30 Elo in Stockfish's
   own testing. Needs the SPRT harness to verify on this codebase too, but
   this is the most likely single small win on the list.
5. **Multithreading (Lazy SMP).** The one item on this list big enough to
   measure with the *existing* 100-game harness (real multithreading gains
   are typically well above the noise floor, unlike the two items above).
   Requires redesigning every global (`g_tt`, `g_main_history`,
   `g_cont_history`, `g_capture_history`, `g_countermoves`, `g_killers`) for
   thread safety — a rewrite, not a patch. Do this deliberately, and
   re-verify TT/history correctness under contention.
6. **Improve time management** once testing moves to real per-move time
   budgets (SPRT harness territory). Current `go` handling
   (`myTime/movestogo + myInc*0.8`, `movestogo` defaulting to 30) has no
   soft/hard limit split and no "extend if the best move is unstable" logic.
   Not yet exercised by any current test (`tournament.py` uses fixed depth
   for this engine), so not measurable until item 1 is in place.
7. **Staged move generation (MovePicker-style).** Try the TT move first
   without generating anything; only generate captures, then quiets, if
   needed. Real node-count lever, but requires promoting the TT move's
   legality check to fully rigorous — add a perft regression test (item 8)
   first, given the illegal-move risk.
8. **Perft-based movegen regression test.** Still no automated perft check
   in the UCI loop. Add before item 7.
9. **Bigger/better NNUE net.** Still not the likely bottleneck — the
   2900–3300 architecture-based estimate is unverified either way (see
   "Session follow-up" above), and every item above this one is cheaper to
   try first and doesn't require retraining. Revisit only after items 1–5
   are done and re-measured, and only with a genuine held-out validation
   split (a separate diagnostic script, not a `train.py` change, per this
   session's precedent with `net_eval_diagnostic.py`) to confirm it's
   actually the limiting factor before assuming a bigger architecture helps.
