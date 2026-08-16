# nnue_engine — project notes

A self-contained C++ UCI chess engine (`nnue_engine.cpp`) using a custom-trained
HalfKP NNUE (`checkpoints/model.nnue`, H=256, L1=32, L2=32, 8 output buckets)
for evaluation, paired with a Stockfish-style alpha-beta search. Trained via
`train.py`/`model.py`/`serialize.py` on `positions.bin` (~256M positions,
Lichess cloud-eval labels at depth ≥20). Benchmarked/tournament-tested against
Stockfish via `tournament.py` and `bench.py`.

**Current status: the ~2683 Elo figure is retired — not "beaten" or
"regressed from", just measured on a different, no-longer-trusted harness
(`tournament.py`, fixed depth=7 for this engine vs. fixed 50ms for
Stockfish — both sides were actually thinking for a similar ~20–50ms/move
at that setting, which doesn't resemble a real time control and isn't
comparable to anything below). The new, trustworthy external reference:
**fastchess, `nnue_engine` vs. Stockfish 18 @ `UCI_LimitStrength=true
UCI_Elo=2750`, `tc=8+0.08` equal both sides** (this IS a real, fair,
equal-time-control comparison). Two data points on this anchor so far:

| State | Score vs. SF@2750 | Elo |
|---|---|---|
| timing-fix + corrhist + checkext | 34.50% (200 games) | -111.37 ± 45.79 |
| + soft/hard time management | 37.00% (200 games) | -92.46 ± 46.37 |

The ~+19 Elo anchor movement is directionally consistent with, but smaller
than, the +53.28 Elo the same time-management change measured on the
internal (engine-vs-engine) SPRT — expected: different opponent, different
noise floor, and Elo doesn't compound linearly across measurement scales.
**Both anchor points are below "2750"** on Stockfish's own `UCI_Elo` scale,
but per the note in "Next lever" below, `UCI_Elo` limits strength via
move-selection noise + depth caps calibrated for longer time controls, and
is known to under-limit (play stronger than its label) at fast time
controls like `tc=8+0.08` — so treat -92 to -111 as a pessimistic estimate
of true relative strength, not a precise CCRL-style number. **The ~3000 Elo
goal was set against the retired 2683/`tournament.py` scale and has no
defined meaning on this new anchor** — re-establishing what "3000" should
mean here (e.g. a longer, more standard time control, or a CCRL-style
external ladder) is worth doing at some point, but isn't blocking: keep
improving and re-running this same anchor command to track real progress
in the meantime. Read "Prioritized next steps toward ~3000 Elo" for what's
done vs. open, and "Operational lessons from this session" before running
any more SPRTs — both are near the bottom of this document.

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

### RESOLVED — the time-management overrun bug is root-caused and fixed

(Formerly "CRITICAL — a real, only partially-fixed time-management bug."
This section is history for context; skip to "Prioritized next steps" for
what's still open.)

Building the harness immediately surfaced a bug that would otherwise have
silently contaminated every SPRT run with spurious time losses unrelated to
move quality. Two sub-bugs, found and fixed across two sessions:

**Sub-bug 1 (earlier session): `quiescence()` never called `out_of_time()`.**
Under `tournament.py`'s fixed-depth testing this was invisible; under
`fastchess`'s strict per-move time enforcement it caused real, repeated
"loses on time" forfeits. Fixed at `nnue_engine.cpp:1926` (mirrors
`alpha_beta`'s existing `out_of_time()`-then-`return alpha` pattern).

**Sub-bug 2 (this session, the actual root cause of the residual 143–165ms
overruns): three re-search call sites chained a second, more expensive
search onto the result of a first one without ever checking the clock in
between.** The exact sites CLAUDE.md flagged as the likely culprit before
this session started ("check the aspiration-window retry loop and the LMR
fail-high re-search path") were exactly right:

- LMR fail-high re-search (full-depth, `nnue_engine.cpp` — was ungated)
- The PV re-search when a null-window search lands inside `(alpha, beta)`
  (was ungated)
- ProbCut's `alpha_beta` re-search fired after a qsearch probe crosses
  `prob_cut_beta` (was ungated)

Mechanism: `out_of_time()`'s node-count gate (`g_nodes & 4095 == 0`) is
checked at every `alpha_beta`/`quiescence` *function entry*, so in isolation
it's tight — empirically ~2–4ms between consecutive real polls, confirmed by
instrumentation (see below). But when the *first* (reduced-depth or
narrow-window) search in one of the three chains above is itself aborted by
a timeout deep inside its own recursion, it returns its local `alpha` bound.
After negation across several plies of differing alpha/beta windows, that
can look exactly like a genuine fail-high or a real PV-window result — with
no real search behind it. None of the three sites checked `out_of_time()`
before trusting that score and launching the second, more expensive search.
That second search is a *fresh* function call, so it starts at a node count
that just moved one past a 4096-boundary — meaning the next real clock check
doesn't happen until the count organically reaches the *following* boundary,
up to ~4096 nodes later. Nested plies can each hit this same gap
independently and compound, which is why observed overruns were variable
(5–165ms) rather than a fixed amount.

**Fix:** gate all three re-search call sites on `!out_of_time()` — if time is
already up, skip the more expensive re-search and use the
already-computed (if aborted) score as-is; the existing post-move
`out_of_time()` check (`nnue_engine.cpp:2451`-ish, after `undo_move`) still
catches it correctly from there.

**Diagnostic instrumentation added** (kept in the codebase, off by default):
set env var `NNUE_TIME_DEBUG=1` to make the engine print one `[TIMEDBG]`
line per move to stderr — `budget`, `search_wall`, `overrun`, plus internal
poll-gap fields (`last_poll_to_return_gap`, `final_poll_interval`,
`max_poll_gap`) that were used to localize the bug and can be reused if a
similar issue ever resurfaces. This is purely additive — zero behavior
change when the env var is unset (confirmed: it gates every stderr print in
the `go` handler and `main()`).

**Verified, in increasing order of realism:**

1. `st=2` (2000ms/move), `timemargin=100`, `-concurrency 1`: before the fix,
   overrun was 5–165ms on *nearly every move* (617–2770 samples measured
   across several runs); after the fix, 12 full games / 2770 samples showed
   overrun capped at 0–4ms (one 24ms outlier), **zero time losses** (was:
   reproduced within the first game, consistently).
2. `tc=8+0.08` (the actual SPRT time control) with `-concurrency 4`
   (realistic contention, not the artificially clean `concurrency=1` case):
   60/60 games completed cleanly, 8126 samples, overrun essentially 0–2ms
   with a handful up to 31ms, **zero time losses**.

Root cause confirmed, fix verified under both an artificially generous
setup and the actual conditions a real SPRT run will use. **Every prior
SPRT-blocking concern in this document is now resolved** — proceed with
game-based A/B testing for the remaining prioritized items below.

**A regression check (post-fix vs. pre-fix engine, `elo0=-5 elo1=5`) was
attempted and abandoned as methodologically invalid, not because the fix is
suspect.** At `tc=8+0.08` (~330ms/move) with `timemargin=200`, the pre-fix
engine's 5–165ms/move overrun is a 10–15%+ *effective time-control edge*
that a lenient margin never punishes — the test was comparing an honest
engine to one quietly playing at ~1.1x the nominal time control, and after
102 games was drifting negative (-63 ± 61 Elo) for exactly that reason, not
because the fix costs search quality. **Don't re-run this comparison** —
tightening the margin would only turn it into a forfeit-rate measurement,
which is already known (pre-fix: real, repeated forfeits; post-fix: zero
across 8126+ sampled moves). The fix is correct and necessary regardless of
any Elo comparison: an engine that systematically overspends its allocated
time is violating the UCI contract and will be disqualified or forfeited by
any real arbiter with reasonable enforcement.

**Important corollary: the ~2683 Elo figure at the top of this document is
now known to be inflated by an uncertain amount and should not be used as a
clean baseline for future comparisons.** It was measured with the pre-fix
binary, which was quietly overrunning its time budget on nearly every move.
Some fraction of that 2683 was time theft, not search quality. Every SPRT
from here on (candidate vs. `nnue_engine_baseline`, both post-fix, both
honest) is internally consistent and trustworthy — but a future re-measurement
against Stockfish via `tournament.py`/a fresh 100-game run should be expected
to land *below* 2683 even with zero code-quality regression, and that is not
a red flag when it happens.

## Operational lessons from this session — read before running more SPRTs

1. **Never rebuild `./nnue_engine` while any SPRT process references that
   path.** `fastchess` defaults to `restart=off`, meaning engine subprocesses
   are long-lived across the whole run (not re-spawned per game) — an
   already-running process keeps executing its originally-mapped binary
   image in memory even if the file on disk is overwritten (standard Unix
   exec semantics), so an in-flight run isn't corrupted by a rebuild. But
   there is no guarantee against it (a crash-restart or future concurrency
   change could spawn a process that picks up the new file), and it makes
   auditing "what was actually tested" needlessly hard. Practice: copy the
   candidate to a uniquely-named binary (e.g. `nnue_engine_<change>_candidate`)
   *before* launching its SPRT, and don't touch `./nnue_engine` again until
   that run is stopped or complete. `nnue_engine_baseline` should always be a
   copy of the last-known-good, already-merged state — recopy it fresh from
   `nnue_engine` immediately after each change lands, before starting the next.
2. **Always run a small (~40-game, few-minute) sanity check before committing
   to a full multi-hour SPRT.** This isn't a statistically meaningful Elo
   measurement — it's a fast, cheap check for "is this obviously broken"
   (a bug that makes the candidate lose ~90%+ of games doesn't need 2000
   rounds to see; a healthy, balanced-looking 40-game result is the bar to
   clear, not a p-value). This session's correction-history feature had two
   serious bugs (see its commit history) that a 40-game check caught in
   about 10 minutes each; the first real SPRT attempt had already burned 44
   games trending toward a near-100% loss before this practice was adopted.
   Command shape: same as the full SPRT but `-rounds 20 -games 2 -repeat`
   (40 games total) with no `-sprt` flag, then eyeball `Games/Wins/Losses/
   Draws` in the output — a wildly lopsided score (e.g. worse than ~15%) is
   a bug signal, not a "the change is bad" signal; go find the bug.

## Prioritized next steps toward ~3000 Elo

Status of the original list: item 1 (benchmark methodology) is **done** —
see "Session follow-up" above; the `ucinewgame` harness bug turned out to
matter more than the depth-vs-time mismatch, and re-measurement landed at
2683±17, close to the original ~2600 estimate. Items are renumbered/updated
below to reflect everything found this pass.

1. **DONE — the time-management overrun bug is fixed and verified.** See
   "RESOLVED — the time-management overrun bug is root-caused and fixed"
   above. Root cause: three re-search call sites (LMR fail-high, PV
   re-search, ProbCut re-search) could launch an expensive, ungated second
   search on a score that was itself an artifact of an already-timed-out
   subtree. Fixed by gating each on `!out_of_time()`. Verified clean (zero
   time losses) at the actual SPRT time control under real contention
   (`tc=8+0.08`, `concurrency=4`, 8126+ sampled moves). The planned
   post-fix-vs-pre-fix regression check was abandoned as methodologically
   invalid (see "Operational lessons" above) — don't re-attempt it.
2. **DONE — correction history added, fixed, and SPRT-verified.** First
   attempt was badly broken (two independent bugs — a missing grain divisor
   that let a saturated correction swing eval by over a pawn, and a
   `best_move`-vs-`alpha_raising_move` gating bug that made the table only
   ever ratchet upward) and lost ~90% of games in a 40-game sanity check
   before either bug reached a real SPRT. See the "Fix correction history"
   commit for both root causes. Single-component (pawn-structure-only), see
   the code comment at `g_pawn_corrhist`'s declaration for the design.
3. **DONE — check extension gated with an SEE check and a per-line budget
   (`CHECK_EXT_BUDGET=16`).** Previously unconditional +1 for every checking
   move, no SEE gate, no budget — confirmed via node-count testing to cost
   23–57% extra nodes at fixed depth in 3 of 4 positions (CLAUDE.md,
   earlier session).
4. **Items 2+3 combined, SPRT-verified together (not isolated — see below):
   +159.65 ± 41.24 Elo vs. the honest (post-timing-fix) baseline, LOS
   100%, `elo0=0 elo1=10` — H1 accepted at 256 games, LLR crossed the bound
   with more than 3x the required margin.** This is a very large result —
   far above the ~20–30 Elo originally estimated for correction history
   alone — most likely explained by the check-extension fix mattering far
   more than expected (the 23–57% node-count savings measured earlier
   translating directly into search depth/quality at a fixed time budget),
   with correction history adding on top. **Caveat: this SPRT tested the two
   changes combined, not in isolation** (they landed as sequential commits
   on the same tree before this was noticed) — if precise per-feature
   attribution matters later, that requires a follow-up SPRT isolating one
   change from the other (e.g., build a check-extension-only candidate from
   `git show <timing-fix-commit>:nnue_engine.cpp` plus a cherry-picked
   check-extension diff). Not done this session — the combined result was
   decisive enough that isolating attribution wasn't the priority; revisit
   only if a future change's SPRT result looks surprising and disentangling
   past changes would help explain why.
   **`nnue_engine_baseline` now holds this combined, winning state** —
   every SPRT below should compare against it, and should be re-copied
   fresh from `nnue_engine` immediately after the *next* change lands (see
   "Operational lessons" above for why: never reuse a stale baseline copy).
5. **Multithreading (Lazy SMP).** Still the next big lever. Requires
   redesigning every global (`g_tt`, `g_main_history`, `g_cont_history`,
   `g_capture_history`, `g_countermoves`, `g_killers`, and now
   `g_pawn_corrhist`) for thread safety — a rewrite, not a patch. Do this
   deliberately, and re-verify TT/history correctness under contention.
6. **DONE — soft/hard time-limit split with best-move-instability
   extension.** `soft_limit` keeps the old `myTime/movestogo + myInc*0.8`
   formula as the "normal" allocation; `hard_limit = min(myTime/2,
   soft_limit*4)` is a generous cap. Iterative deepening won't start a new
   depth once `soft_limit` is used up, unless the best move changed between
   consecutive completed iterations at depth ≥5, in which case the
   effective soft limit stretches 1.3x per instability event (capped at
   `hard_limit`). Only active for wtime/btime-derived budgets — fixed
   `go movetime`/`go depth`/`go infinite` unaffected. **SPRT-verified:
   +53.28 ± 22.74 Elo vs. the corrhist+checkext baseline, 598 games,
   `elo0=0 elo1=10`, H1 accepted, LOS 100%, zero time losses.**
   `nnue_engine_baseline` now holds this state. **As of the end of this
   session, `nnue_engine_baseline` is rebuilt fresh from HEAD
   (`6c0c9b1`)** — it also includes the (behaviorally inert but additive)
   perft command and every documented-dead tuning experiment was reverted
   before that rebuild, so `nnue_engine_baseline`'s source == `nnue_engine.cpp`
   at HEAD exactly. Keep this invariant: rebuild `nnue_engine_baseline`
   fresh after every commit that changes `nnue_engine.cpp`, don't reuse an
   older copy (see "Operational lessons" above for why this matters).
7. **DEAD — staged move generation (MovePicker-style). Ruled out by
   measurement, don't revisit without new evidence.** The idea (try the TT
   move first with no movegen, generate captures/quiets only as needed) was
   never actually about search quality, just cutting the cost of `gen_moves`
   at nodes that cut off on the TT move alone. `perft` (item 8, done — see
   below) makes that cost directly measurable: pure movegen + legality
   filtering + do/undo, no search overhead at all, runs at **~100–200M
   nodes/sec** (e.g. startpos depth 5: 4,865,609 nodes in 0.031s). Real
   search runs at ~1.6–2.2M nodes/sec (see the earlier NPS diagnostic
   section). That's a ~100x gap — `gen_moves` is single-digit nanoseconds
   against a static eval alone costing ~213ns (measured earlier this
   project), so even under a generous margin for what perft's number
   doesn't include (TT probe, `score_moves`, SEE, actual eval — all real
   search pays these and perft doesn't), movegen is nowhere near the
   per-node cost floor. Staged movegen also carries real correctness risk
   (promoting the TT move's legality assumption without regenerating the
   full list) for a win that isn't there. Not worth it.
8. **DONE — perft-based movegen regression test.** `perft <depth>` / `perft
   divide <depth>` UCI commands added, purely additive. Verified against
   the 5 standard test positions (startpos, Kiwipete, positions 3–5) at
   depths 1–5: every result matches the known-correct node count exactly.
   This is also what ruled out item 7 above — use it again before any
   future movegen change.
9. **Bigger/better NNUE net.** Still not the likely bottleneck — the
   2900–3300 architecture-based estimate is unverified either way (see
   "Session follow-up" above), and it's an expensive lever (requires
   retraining) relative to what's left below. Revisit only after
   multithreading is decided one way or the other, and only with a genuine
   held-out validation split (a separate diagnostic script, not a
   `train.py` change, per this session's precedent with
   `net_eval_diagnostic.py`) to confirm it's actually the limiting factor.

## Status: the original prioritized list is exhausted

Items 1–4 and 6–8 are done (verified via SPRT or perft, see each item
above); item 5 (Lazy SMP) is deliberately deferred (see below); item 9 is
deprioritized as expensive-and-unlikely-to-be-the-bottleneck. **There is no
undone item left on the original list — the next session should not assume
one exists and go looking for it.** What actually produced every real win
this session was finding something *wrong* in code that looked finished
(an ungated re-search, an unconditional check extension, a single-limit
time budget), not adding new features — that's the pattern to keep
following, not a checklist to keep exhausting. Two concrete starting points
for whoever picks this up next, both cheap (one-line change + 40-game
sanity gate + SPRT) and neither yet tried:

- **`CHECK_EXT_BUDGET` tuning: tried 6 and 24, both inconclusive-to-worse,
  keeping 16.**
  - `=6` vs. the `=16` baseline (`elo0=-5 elo1=5`): after 373 games the
    estimate had stabilized around **-23 to -27 Elo**, three consecutive
    checkpoints all in that band, tightening error bars — a clear,
    consistent negative trend. Stopped before full LLR convergence since
    the direction was no longer in doubt. A tighter budget costs more in
    missed tactics than it saves in nodes.
  - `=24` vs. `=16` (`elo0=0 elo1=10`): **cautionary tale, read before
    trusting any mid-run SPRT trend on a small effect.** At ~700–1100
    games this looked like a real win (+9 to +17 Elo, tightening error
    bars, briefly LLR 25–34% of the way to the H1 bound) — tempting to
    stop early and adopt. Kept running instead. By ~1660 games the
    estimate had decayed to **+3.82 ± 12.05 Elo, LLR ≈ 0** (bouncing
    between -0.07 and +0.26) — a small, noisy effect indistinguishable
    from zero, not the double-digit win it looked like 500 games earlier.
    Stopped and reverted to `=16`. **Lesson: for a small true effect size,
    a promising-looking trend at 700-1100 games is not evidence — SPRT
    error bars shrink slower than they look, and an estimate that hasn't
    yet crossed the bound can still drift a long way before it does (or
    doesn't). Don't adopt a change on a mid-run trend; wait for LLR to
    actually cross a bound, or accept that games in the low thousands
    weren't enough for this effect size and the change is genuinely
    borderline.** If revisited, either commit to running until LLR
    actually resolves (could be several thousand more games), or accept
    `16` as good-enough and spend the compute elsewhere.
  - Also untried: the `1.3x` instability stretch factor and
    `hard_limit = min(myTime/2, soft_limit*4)` formula in time management
    — both still just reasonable guesses, not yet SPRT-tuned.
- **DEAD — aspiration window fallback rate.** Instrumented directly
  (temporary counters, since removed): zero full-width fallbacks fired
  across 45 aspiration-window iterations spanning 7 diverse positions
  (openings through tactical middlegames) at a realistic ~330–1320ms
  budget. The `ASP_MAX_TRIES=4` retries with widening converge before ever
  needing it in practice — not a real cost center, don't pursue this.

**Lazy SMP (item 5) is deliberately still deferred, not forgotten.** Both
the internal SPRT and the Stockfish anchor (see "Current status" at the top)
run single-threaded on both sides — the harness as it exists cannot see a
multithreading win at all, so building it now means flying blind on
whether it worked. Decide first whether multi-threaded strength is actually
part of the goal; if so, the benchmark needs redesigning (a threaded
opponent, a threaded anchor) before writing any thread-safety code, not
after.
