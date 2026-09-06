# nnue_engine — project notes

A self-contained C++ UCI chess engine (`nnue_engine.cpp`) using a custom-trained
HalfKP NNUE (`checkpoints/model.nnue`, H=256, L1=32, L2=32, 8 output buckets)
for evaluation, paired with a Stockfish-style alpha-beta search. Trained via
`train.py`/`model.py`/`serialize.py` on `positions.bin` (~256M positions,
Lichess cloud-eval labels at depth ≥20). Benchmarked/tournament-tested against
Stockfish via `tournament.py` and `bench.py`.

**Current status: the ~2683 Elo figure is retired** — not "beaten" or
"regressed from", just measured on a different, no-longer-trusted harness
(`tournament.py`, fixed depth=7 for this engine vs. fixed 50ms for
Stockfish — both sides were actually thinking for a similar ~20–50ms/move
at that setting, which doesn't resemble a real time control and isn't
comparable to anything below). The trustworthy external reference:
**fastchess, `nnue_engine` vs. Stockfish 18 @ `UCI_LimitStrength=true
UCI_Elo=2750`, `tc=8+0.08` equal both sides** (a real, fair, equal-time-
control comparison). Four data points on this anchor so far, tracking the
session's work in order:

| State | Score vs. SF@2750 | Elo |
|---|---|---|
| timing-fix + corrhist + checkext | 34.50% (200 games) | -111.37 ± 45.79 |
| + soft/hard time management | 37.00% (200 games) | -92.46 ± 46.37 |
| + **NNUE output-layer quantization fix** | **58.75% (200 games)** | **+61.43 ± 42.49** |

**The single largest lever this session, by far, was not a search
improvement at all — it was a ~1000x-miscalibrated quantization scale in
the trained net's output layer, present since the net was first serialized
and silently corrupting every eval this engine has ever computed.** See
"NNUE output-layer quantization bug — found, fixed, and it was the
dominant bottleneck" below for the full diagnosis. The jump from -92.46 to
+61.43 (**+153.89 Elo on this anchor from that one fix**, SPRT-confirmed
separately at +211.62 ± 103.71 on the internal engine-vs-engine harness,
resolved in 46 games) means **this engine now plays above Stockfish's own
"2750" self-rating** at this time control — with the big caveat (see that
section's "Next lever" note) that Stockfish's `UCI_Elo` limiting is known
to under-limit at fast time controls, so treat +61 as directionally huge
but not a precise CCRL-style number.

**The ~3000 Elo goal was set against the retired 2683/`tournament.py`
scale and still has no defined meaning on this new anchor** —
re-establishing what "3000" should mean here (a longer, more standard time
control, or a CCRL-style external ladder) is worth doing at some point,
but isn't blocking: keep improving and re-running this same anchor command
to track real progress. Read "Prioritized next steps toward ~3000 Elo" for
what's done vs. open, and "Operational lessons from this session" before
running any more SPRTs — both are near the bottom of this document.

## NNUE output-layer quantization bug — found, fixed, and it was the dominant bottleneck

The user asked directly: given this net has the same architecture as
Stockfish's first NNUE net, and the search now has all of Stockfish's
standard mechanisms, why was the engine still ~800 Elo below what that
should support — is it the NNUE (training), the serialization/quantization
format, or the search? This section is the answer, arrived at by direct
measurement, not inference from training-loss numbers.

### Diagnosis method

Loaded the trained float checkpoint (`checkpoints/nnue_best.pt`) directly
in Python via `model.py`, ran its forward pass on real positions using the
exact same HalfKP feature encoding `data.py` uses for training, and
compared the output (which `train.py`'s loss function treats as
centipawns — see `WDL_SCALE`/`nnue_loss`) against the C++ engine's `eval`
UCI debug command for the *same positions* via the deployed
`checkpoints/model.nnue`. They diverged by 30–160cp per position,
sometimes with a different sign — far beyond int8 quantization rounding
noise (which should be a few cp at most).

Bisected in two steps, both empirical, not just code review:

1. **Byte-exact comparison: does `model.nnue` match what `serialize.py`
   should produce from `nnue_best.pt`?** Re-ran `serialize.py`'s
   `quantise_ft/l1/l2/out` functions in Python on the checkpoint and
   compared the resulting bytes, byte-for-byte, against the actual
   `checkpoints/model.nnue` payload. **Identical.** This ruled out "wrong
   checkpoint was serialized" and "serialize.py writes something different
   than what it computes" — whatever was wrong, it was wrong in the
   quantization *design*, not a mismatch between intent and execution.
2. **Weight-clipping audit.** Checked what fraction of each layer's
   weights get clamped to the int8 boundary (±127) during quantization at
   each layer's chosen scale constant. `ft`/`l1` weights: ~0% clipped.
   `l2`: 3.2% clipped (minor). **`output.weight`: 96.875% clipped.**
   The output layer's trained weights (median |w|=48.5, max=194.6, std=68.5)
   are an order of magnitude larger than every other layer's (all under
   ~9 in magnitude) — expected in hindsight, since the output layer's
   32 inputs are bounded to [0, L2_SCALE] by `clipped_relu`, so producing a
   cp-scale output (tens to hundreds of cp) from 32 *bounded* inputs
   mathematically requires large weights. `OUT_SCALE=600` was chosen
   assuming the same small-weight regime as the other layers and crushed
   ~97% of this layer's learned weights down to ±127/±128, destroying
   almost all the magnitude information the final layer had learned.

**Fixing `OUT_SCALE` alone made the eval visibly *worse* (jumped to
thousands of cp)**, which surfaced a second, independent, previously-masked
bug: the output-layer *bias* was quantized as `bias * OUT_SCALE`, but by
the same pattern used for every other layer (bias scale = product of all
upstream scales — see `l1 bias scale = FT_SCALE * L1_SCALE`, `l2 bias
scale = L1_SCALE * L2_SCALE`), it needed to be `bias * L2_SCALE *
OUT_SCALE`. The dot-product term (the weighted sum of the 32 L2 activations)
was inflated by exactly `L2_SCALE=64x` relative to the bias term the whole
time. `OUT_SCALE=600`'s clipping had coincidentally crushed the dot-product
term down to a plausible-looking magnitude, masking this second bug —
which is exactly why fixing bug #1 in isolation made things visibly worse
before fixing bug #2 revealed the actual correct output.

### Fix

- `serialize.py`: `OUT_SCALE` changed from `600` to `0.6` (derived from the
  measured weight range, with headroom for future retrains — recompute if
  a retrain meaningfully shifts the output layer's weight scale).
  `quantise_out`'s bias scaling changed from `bias * OUT_SCALE` to
  `bias * L2_SCALE * OUT_SCALE`.
- `nnue_engine.cpp`: `OUT_SCALE` changed from `static constexpr int` `600`
  to `static constexpr double` `0.6`. Final descale changed from
  `score / OUT_SCALE` to `score / (L2_SCALE * OUT_SCALE)`, with proper
  rounding (`std::lround`) instead of the previous integer-truncating
  division.
- `checkpoints/model.nnue` regenerated from `checkpoints/nnue_best.pt` with
  both fixes (`python3 serialize.py checkpoints/nnue_best.pt
  checkpoints/model.nnue`).

**Verified:** quantized C++ eval now matches the float model within a few
cp (quantization-noise level) across 6 diverse test positions — was off by
30–160cp before, sometimes flipping sign.

### Measured impact

- Internal SPRT (fixed vs. pre-fix `.nnue`, otherwise identical binaries,
  `tc=8+0.08`): **+211.62 ± 103.71 Elo**, `elo0=0 elo1=50`, H1 accepted,
  LOS 100%, resolved in just **46 games** (the effect is large enough that
  SPRT didn't need anywhere near its normal game budget). A cheap 40-game
  sanity check beforehand already showed 27 wins–1 loss–12 draws, 82.5%
  score — this was never a borderline result.
- Stockfish@2750 anchor: **-92.46 → +61.43 Elo (+153.89)** — see the table
  at the top of this document. This engine now scores above 50% against
  Stockfish's own "2750" self-rating at this time control.

### What this means for the user's question

**It was the serialization/quantization formatting — not the training,
and not primarily the search.** The training loss (0.0073, smooth,
plateauing) was telling the truth about the *float* model; the deployed
*quantized* model the engine actually played with had never faithfully
represented that float model, from the very first time `model.nnue` was
generated. Every SPRT result earlier in this document — the +159.65 Elo
correction-history/check-extension result, the +53.28 Elo time-management
result, the entire ~2683 `tournament.py` measurement — was measured with
this bug present. The search-side fixes were still real (each was
independently SPRT-verified against a fixed eval, so they weren't
artifacts of the eval bug), but they were all improving a search that was
reading from a badly corrupted evaluation function the entire time. This
single fix outweighs the combined effect of every search fix this session
found, on both the internal harness and the external anchor.

`verify_quantization.py` (checked-in script covering all three of this
section's diagnostic steps — byte-exactness, per-layer clipping, and
quantized-vs-float eval agreement) reports the current state every time it
runs; use it after any retrain or any change to `serialize.py`'s
quantization, before trusting the resulting `.nnue` file in the engine.

### Follow-up audit: is there anything else to fix in the quantization pipeline?

Asked directly by the user after the fix above landed — full answer,
including one thing that looked worth fixing and, on direct empirical
test, turned out not to be:

- **`l2.weight`'s 3.2% clipping — investigated, tested, confirmed NOT worth
  fixing.** `L2_SCALE` does double duty: it's both `l2.weight`'s
  quantization scale *and* the activation-resolution scale for the 32
  values feeding the output layer (same architectural pattern as
  `FT_SCALE`/`L1_SCALE`). Lowering it to reduce clipping also coarsens
  activation precision — a real tradeoff, not a one-directional fix.
  Tested empirically (simulated the quantized pipeline in Python at
  several `L2_SCALE` values, measured divergence from the float model on
  the same 6 test positions used in the section above):

  | `L2_SCALE` | mean \|float − quant\| | max \|float − quant\| |
  |---|---|---|
  | 64 (current) | 2.86cp | 4.44cp |
  | 32 | 2.75cp | 3.83cp |
  | 20 | 4.13cp | 7.14cp |
  | 14 (zero clipping) | 5.83cp | 10.14cp |

  Going lower to chase zero clipping makes agreement with the float model
  *worse* — resolution loss outweighs the clipping benefit below ~32.
  64 is already close to optimal. **Left unchanged; don't revisit without
  new evidence.** (32 shows a marginally smaller max-diff on this n=6
  sample, but the gap is within likely noise for that sample size — not
  worth the risk of touching `quantise_l2`, `quantise_out`, and the C++
  constant for an unconfirmed sub-1cp improvement.)
- **`ft`/`l1` clipping**: ~0% and 0.006% respectively — both already
  near-optimal (measured `L1_SCALE` headroom: current 64 vs. a
  zero-clipping threshold of ~58.8, i.e. already close to the ceiling).
  No action needed.
- **Dead code with the same historical bug, not touched.** `engine/`
  (a separate, older prototype — `evaluate.h`, `types.h`, `main.cpp`, a
  compiled `engine/engine` binary) has its own hardcoded `OUT_SCALE=600`,
  unfixed. Confirmed it is *not* part of the active pipeline — `tournament.py`,
  `bench.py`, and every fastchess SPRT this whole project reference
  `nnue_engine.cpp`/`nnue_engine`, not `engine/`. Only `debug.py`
  references `engine/engine`, and nothing calls `debug.py`. Left alone
  deliberately (fixing dead code is wasted effort and risks confusion
  about which binary is "the" engine) — worth a cleanup pass or deletion
  at some point, but not an Elo-bearing task.
- **Backup files** (`bakcup.cpp`, `nnue_engine_backup.cpp`,
  `nnue_engine_backup_2.cpp`) also carry the old `OUT_SCALE=600` — these
  are pre-session historical snapshots, referenced by nothing. Same
  treatment as `engine/`: not touched, not relevant.

### Does the search's eval scale now match what its ported-from-Stockfish pruning margins assume?

This was flagged as an open, unresolved concern in the immediately
preceding session turn — resolved this pass with cleaner data. **Short
answer: no strong evidence of the scale mismatch that was worried about.
Don't rescale the pruning margins based on the earlier (confounded)
signal.**

**What was wrong with the earlier signal:** `net_eval_diagnostic.py`
compares this engine's *static* eval against Stockfish's *searched* score
(`sf.search_score_cp()`, i.e. after real search, not a raw eval) on
real-game positions. Search scores are systematically more extreme than
static eval on the same position — search finds forced wins, converts
static advantages into decisive material, spots mate — none of which a
one-shot NNUE evaluation can see. The earlier regression (slope=0.207) and
median-ratio (1.9x) disagreed with each other (Pearson r only 0.52)
precisely because they were measuring "how much more decisive is search
than static eval," not a clean eval-to-eval scale factor — an apples-to-
oranges comparison, not a scale bug.

**Clean re-measurement:** modern Stockfish exposes its own *static* eval
directly via its `eval` console command (not a UCI command — run
interactively: `position fen ...` then `eval`, read the `Final evaluation`
line, in pawns). Compared directly against this engine's static eval (via
the `eval` UCI debug command) on the same 6 test positions, no search
involved on either side:

| position | ours (cp) | Stockfish static (cp) | ratio |
|---|---|---|---|
| startpos | 31.2 | 15.0 | 2.078 |
| r1bqk2r... (mg) | 139.2 | 178.0 | 0.782 |
| r1bqkb1r... (mg) | 31.8 | 41.0 | 0.776 |
| 2kr3r... (mg) | -84.8 | -102.0 | 0.831 |
| KPvK endgame | 16.3 | 0.0 | undefined (÷0) |
| Kiwipete-like | -171.7 | -171.0 | 1.004 |

Four of five defined ratios cluster in **[0.78, 1.20]** — reasonably
close to 1.0, not the 2–7x gap the search-score-based comparison implied.
The one outlier (startpos, 2.08x) is on a small absolute eval (15cp) where
ratio is a noisy metric for a small absolute difference (16cp) — not
strong evidence of a systematic multiplicative bias.

**Conclusion: the Stockfish-ported pruning margins (RFP=234, futility
margins, NMP formula, aspiration window, etc.) are probably reasonably
well-matched to this engine's eval scale, on the evidence available.**
There is no confirmed case for a blanket rescale. n=6 positions is a small
sample — if this matters enough to someone to chase a tighter answer, get
more data points before acting, not before to guard against the small
sample happening to look reassuring by chance.

### Is it worth pruning more aggressively now that the eval is trustworthy?

Reasoned answer, not yet tested — a fair question but not a slam-dunk
"yes" the way the user's framing suggested, for a specific reason worth
being precise about:

**The quantization bug's error was ~30–160cp of *systematic distortion*,
not noise.** Pruning margins like `RFP_MARGIN=234` exist to buffer against
a fundamentally different problem: a static eval, however precisely
computed, can still be *wrong relative to what a deeper search would find*
(missed tactics, positional complexity one ply of NNUE can't see) — that
gap is a property of net quality and search depth, not of quantization
precision. Fixing quantization noise (~4cp now, was 30–160cp) doesn't
directly imply the *net's fundamental static-vs-searched accuracy* changed
at all — that was never what was broken.

**That said, there is a real, more indirect argument for retesting margins
now, not because of a scale theory, but because the eval itself materially
changed.** Every existing pruning constant in this codebase was ported
from Stockfish's source without dedicated tuning against *this* net, and
whatever behavior seemed "fine" earlier this session (the RFP/futility
firing-rate check in the "Session follow-up" section further down,
showing "a large, healthy fraction" of applicable nodes triggering) was
measured against the **old, badly-distorted eval** — not evidence the
current margins are well-tuned for the *fixed* eval's actual behavior.
That check needs re-running post-fix before trusting its conclusion again.

**Recommendation: don't blanket-loosen or tighten margins on a theory.**
If this is worth pursuing, treat it exactly like every other change this
session — pick one concrete, bounded hypothesis (e.g., tighten
`RFP_MARGIN` by a modest amount, or re-run the RFP/futility firing-rate
node-count check against the current eval to see if it still looks
healthy), 40-game sanity gate, then SPRT. Not done this session — flagging
as the most promising remaining lever if more Elo is wanted, but it needs
its own dedicated test, not a bundled guess.

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

## Session follow-up: is search depth the ceiling, and is pruning too conservative to reach it?

The user asked directly: is it worth pruning more aggressively, and how do
we get this engine searching deeper — is depth the actual Elo ceiling right
now? Answered with real measurement, not the n=5–6-position spot-checks
used earlier in this document, per explicit instruction to test at scale
("thousands of positions," not five).

### Depth is confirmed to be a large, real lever

Doubled the per-move time budget (`tc=8+0.08` → `tc=16+0.16`, otherwise
identical binaries) and played it out for real, twice:

- First attempt (`depth_value_sprt.log`) ran at `-concurrency 4`
  *simultaneously* with an unrelated position-sweep job competing for the
  same cores — contaminated by resource contention, discarded. (H0
  accepted at 36 games, Elo −203.97 ± 99.07 — direction was right but the
  magnitude isn't trustworthy given the contention.)
- Clean re-run (`clean_depth_value.log`/`.pgn`, `-concurrency 2`, no
  competing jobs, fixed 200 games, zero forfeits/time losses verified):
  **Elo: −143.07 ± 34.51** for the faster (8+0.08) side, 16W–94L–90D,
  30.50% score. I.e. **one doubling of thinking time is worth about +143
  Elo** at this engine's current strength. This confirms the premise:
  depth (search time, at fixed search quality) is currently the single
  biggest lever available, well above anything a pruning-margin tweak
  could plausibly deliver on its own.

### Node-mass-by-depth measurement: extending pruning to *higher* depths is not the way there

Built new, checked-in instrumentation (`nnue_engine.cpp`, commit
`3212cb5`) to answer "where do search nodes actually live, and how often
does each pruning technique fire, at real game positions and a realistic
time budget" — env-var-gated (`NNUE_PRUNE_DEBUG=1`), zero behavior/perf
cost when unset, exposed via a new `prunestats` / `prunestats reset` UCI
debug command (mirrors the existing `NNUE_TIME_DEBUG`/`eval` pattern).
Tracks: nonleaf node count by remaining depth, RFP/NMP/ProbCut
checked-vs-hit counts (including a depth-bucketed breakdown and a
simulated hit-count at 8 candidate RFP margin coefficients, so different
coefficients can be compared from one sweep without rebuilding), futility/
LMP/history-prune skip counts by depth.

Ran this across **3,903 real-game positions** (deduped FENs, ply 10–70,
not in check, sampled from `games/vs_sf2750.pgn`; positions decided by
`|eval| > 600cp` skipped as uninformative), 330ms/move (the `tc=8+0.08`
anchor's typical per-move budget), **543M total nonleaf nodes** measured.

**Finding: node mass is heavily concentrated at shallow remaining depth —
extending any pruning technique's depth cap higher would touch almost no
nodes.**

- 92%+ of all nonleaf nodes sit at remaining depth ≤ 4.
- Only **0.63%** of nonleaf nodes occur at remaining depth ≥ 9 (RFP's
  current cutoff, `depth<9`) — raising that cap, or LMP's `depth<=8` cap,
  to cover more of the tree literally cannot move the needle; there's
  almost nothing left up there to prune.

This directly refutes the "prune more aggressively by reaching further up
the tree" framing — the pruning caps already cover essentially all the
node mass. **The lever, if there is one, is tightening pruning where the
nodes already are (shallow remaining depth), not extending pruning's
reach.**

### RFP margin coefficient: a depth-isolated, real signal — but pruning more aggressively is a mixed bag, not a free win

RFP's margin coefficient (`static_eval - 234 * (depth - improving) >=
beta`) is `234`, ported from generic Stockfish source, never tuned against
this net (flagged as an open gap in this document's own earlier "Is it
worth pruning more aggressively now that the eval is trustworthy?"
section). Simulated 8 candidate coefficients (234 down to 80) against the
same sweep data.

**First pass (aggregate, not depth-bucketed) looked like a small effect —
this was an analysis artifact, not the real signal.** ~44% of all node
mass sits at remaining depth 1, where RFP's margin is already 0 whenever
`improving=true` regardless of coefficient — diluting any aggregate
comparison. Re-ran depth-bucketed: isolated to the remaining-depth 2–4
band (48% of all node mass, where the margin is actually restrictive at
234–702cp), coefficient **165** (chosen because it's within Stockfish's
own historically-tuned range, not an arbitrary pick) fires **+8.77
percentage points** more often than the current 234 — a real, substantial,
depth-isolated effect, not noise.

**But "fires more often" is not the same as "is stronger" — a tighter RFP
margin also means more false-positive prunes (real threats missed because
the shallow-search verification never runs the tactics deeply enough to
see them).** This is exactly the same class of tradeoff this document
already resolved once, for `CHECK_EXT_BUDGET`: more pruning/fewer nodes
is not automatically more Elo, and mid-run SPRT trends on a small effect
are not trustworthy (see the `CHECK_EXT_BUDGET=24` cautionary tale
earlier in this document — it looked like +9–17 Elo at 700–1100 games and
decayed to +3.82±12.05/LLR≈0 by 1660 games). So this was taken to game
testing, per the project's established protocol, not adopted on the
node-count signal alone.

**Result: inconclusive — not a confirmed win, not a confirmed loss, no
production change made.**

- Built `nnue_engine_rfp165_candidate` (one-line change,
  `234` → `165`, from a scratch copy — never applied to the committed
  `nnue_engine.cpp`).
- **40-game sanity check passed cleanly**: `sanity_rfp165.log`,
  tc=8+0.08, concurrency=4 — Elo 70.44 ± 78.35, 14W–6L–20D, 60.00% score,
  LOS 96.75%. No red flags, cleared to proceed to a real SPRT per the
  project's "Operational lessons" protocol.
- **Real SPRT (`elo0=0 elo1=10`, tc=8+0.08, concurrency=4) did not
  resolve — it crashed.** `sprt_rfp165.log`: ran cleanly to 306 finished
  games (Elo 6.95 ± 29.13, LLR 0.11, only ~3.7% of the way to the +10
  Elo bound — nowhere near either bound, indistinguishable from zero at
  this sample size), then **game 308 ended in `{White disconnects}`** and
  fastchess halted the tournament ("stalled / disconnected and no recover
  option set for engine, stopping tournament"). This has not been
  root-caused — could be a real engine crash under sustained concurrent
  load (worth checking for before trusting either binary further), or an
  environment/resource hiccup unrelated to search logic. **Don't treat
  this candidate as tested-and-rejected or tested-and-accepted — it's
  simply unresolved.** Whoever picks this up next should: (1) check
  whether the disconnect reproduces (rerun with `-recover`, or grep
  stderr/core dumps around game 308 in this log for a crash signature),
  and (2) if the engines are stable, resume/extend the SPRT
  (`./fastchess -config file=config.json` per the log's own suggestion,
  or relaunch fresh) until LLR actually crosses a bound — per this
  document's own established discipline, a point estimate this close to
  zero this early is not a result to act on either way.
- **The live engine was unchanged at the end of this session**:
  `nnue_engine.cpp`'s RFP coefficient stayed `234` at HEAD (`3212cb5`); the
  only committed change this session was the `prunestats` instrumentation
  itself, which is purely additive and does not alter engine behavior when
  `NNUE_PRUNE_DEBUG` is unset. No SPRT-confirmed Elo change from this
  session — the Stockfish@2750 anchor table at the top of this document was
  still current and unchanged at that point.
  **Update, later session: this changed — see "RFP margin tightening:
  234 → 100, adopted" below. RFP is no longer 234 at HEAD.**

### Net takeaway

Depth is confirmed to be the dominant lever (+143 Elo per time-doubling,
measured cleanly). Extending pruning to reach *higher* remaining depths is
ruled out (almost no node mass lives there). Tightening RFP's margin at
the depths that actually matter (2–4) is a plausible, cheap, defensible
next test — the depth-isolated node-count signal is real — but is not yet
an Elo win; it needs a clean, uninterrupted SPRT run to actually resolve
one way or the other before being adopted or discarded.

## Lichess time losses — root cause found and fixed (later session)

The user reported the lichess bot losing a lot of games on time. Read the
actual losses out of `lichess-bot/lichess_bot_auto_logs/lichess-bot.log`
(grep for `outoftime`) rather than guessing: every one is a long bullet or
blitz game (40–100+ plies) that flags near the end — the signature of a
slow clock drain, not one catastrophic move.

**Root cause, verified both mathematically and empirically.** The `go`
handler's time formula (`nnue_engine.cpp`, in `main()`'s `go` case) was:

```cpp
soft_limit = std::max(50, myTime / movestogo + myInc * 0.8);
hard_limit = std::min(myTime / 2, soft_limit * 4);
hard_limit = std::max(hard_limit, soft_limit);  // never below soft
```

The last line was meant to enforce "hard is never below soft," but it also
silently **defeats the `myTime/2` safety cap** whenever the raw formula's
increment term pushes `soft_limit` above half the remaining clock — which
happens whenever `myTime` drops below roughly `1.6×` the increment, entirely
normal territory for a bullet/blitz game 40+ moves in (lichess never sends
`movestogo`, so the engine's hardcoded default of 30 combined with a
shrinking `myTime` walks the clock right into this regime over a long game).
Verified empirically on the previously-deployed binary via hand-fed UCI
commands (`NNUE_TIME_DEBUG=1`, `go wtime 500 winc 1000`): it actually
searched for **817ms — 317ms more than the entire remaining clock**. At
`wtime 200`, it searched **808ms, over 4x the remaining clock**. That's a
guaranteed forfeit once a game's clock decays into this zone, and every
`outoftime` loss in the logs matches this exact pattern.

**Fix** (`nnue_engine.cpp`, commit `57ce89e`): cap `soft_limit` at
`safety_cap = max(10, myTime/2)` *before* deriving `hard_limit` from it,
so `hard_limit = min(safety_cap, soft_limit*4)` can never exceed the
half-clock cap — no separate max-with-soft step needed, since a
`soft_limit` already `<= safety_cap` makes `hard_limit >= soft_limit`
automatic by construction. At normal (fastchess-anchor-scale) time budgets
this is a no-op — verified the formula produces byte-identical output to
before at `myTime=60000` — it only changes behavior in the low-time regime
that was broken.

**Companion fix, `lichess-bot/lib/engine_wrapper.py`** (gitignored, not in
the git history — note it here since CLAUDE.md is this project's memory
across sessions and that directory isn't tracked): `first_move_time()`
previously returned a flat `Limit(time=10)` for the first move of every
game regardless of time control — a guaranteed, unconditional ~17% of a
60-second bullet clock burned on move 1 before the game even starts. Now
scales with the actual base clock (`min(10s, max(500ms, base_time * 0.05))`).

**Verification approach, deliberately not SPRT.** This is a correctness
fix, not an Elo experiment — the fastchess anchor (`tc=8+0.08`, zero time
losses across 8126+ sampled moves in the original time-management SPRT)
never runs long or low enough to trigger this bug, so an SPRT of this fix
would be blind to the thing it fixes and would likely read as a small,
meaningless regression (an engine that stops overspending searches
marginally less). Verified instead via hand-fed `NNUE_TIME_DEBUG=1` UCI
commands at bullet-shaped inputs (`wtime 500/200/1/60000`, `winc 1000`),
confirming the fix caps correctly at low `myTime` and is unchanged at
normal `myTime`. Both `./nnue_engine` and `nnue_engine_baseline` were
rebuilt with this fix (no lichess-bot process was running at the time,
confirmed via `ps aux` before overwriting the live binary).

## RFP margin tightening: 234 → 100, adopted (later session)

Follow-up to "Session follow-up: is search depth the ceiling..." above,
same session as the time-management fix. The user asked directly to keep
cutting/tightening for more depth, having already been told the framing
"94% of nodes at depth ≤3 is a bug" was wrong — that's normal alpha-beta
tree shape (exponential node-mass falloff toward the leaves in any engine,
Stockfish included), not something to fix. The real, already-identified
lever from the prior session's node-mass measurement stands: node mass
concentrates at remaining depth ≤4, so tightening RFP there (not extending
pruning's reach to higher nominal depths, where there's nothing left to
prune) is where the leverage is.

Measured the marginal firing-rate gain of a further 165→100 step (500
sampled real-game positions, depth 2–4 band, same methodology as the
234→165 measurement above): **+11.2 percentage points** — larger than the
234→165 step's +8.77pp, i.e. not a diminishing-returns case.

Built `nnue_engine_rfp100_candidate` (one-line change, `234`→`100`, RFP
gating line only — the separate, already-existing `RFP_MARGIN` constant at
the top of the file is unrelated dead code, see caution below). 40-game
sanity check passed clean (14W–10L–16D, 55%, no forfeits/crashes). Full
SPRT (`elo0=0 elo1=10`, `tc=8+0.08`, `concurrency=4`, `-recover` — the
`165` attempt died to an unrelated disconnect at game 308 without that
flag) launched against the baseline.

**Stopped by explicit user request at 260 games, not run to LLR
resolution.** Progression across checkpoints:

| Games | Elo | LOS | LLR (of the way to +10 bound) |
|---|---|---|---|
| 138 | +45.58 ± 38.16 | 99.13% | 24.9% |
| 180 | +48.57 ± 34.20 | 99.77% | 33.2% |
| 220 | +41.25 ± 30.82 | 99.60% | 34.3% |
| 260 | +38.91 ± 27.94 | 99.71% | 38.9% |

Unlike the `CHECK_EXT_BUDGET=24` false positive (which looked like +9–17
Elo at 700–1100 games and decayed to +3.82±12.05/LLR≈0 by 1660 games), this
result held stable in the high-30s to high-40s across the entire
138→260-game range rather than decaying. The user made an informed call,
aware of that precedent, to adopt on this basis rather than wait for full
resolution. **Adopted, not formally resolved** — flagging the provenance
honestly rather than presenting +38.91 as a confirmed number. If this ever
needs re-verifying, rebuild a baseline from before commit `3c8418f` and
resume the same SPRT command.

Deployed: `nnue_engine.cpp`'s RFP gate is now `static_eval - 100 * (depth -
improving) >= beta` (was `234`), committed as `3c8418f`. Both
`./nnue_engine` and `nnue_engine_baseline` rebuilt from it.

**Caution for next session: `RFP_MARGIN = 120` (top of file, marked
`[[maybe_unused]]`) is dead code, unrelated to the actual RFP gate.** It's
been present and unused since the initial commit — the real gate has always
used a hardcoded literal (`234`, now `100`), never this constant. Don't
assume it reflects the live margin; it doesn't and never has. Worth wiring
up or deleting at some point, not done yet.

**Stockfish@2750 anchor table at the top of this document is now stale** —
it predates both the time-management fix and this RFP change, and was
never a fixed-margin comparison to begin with (it was measured with RFP at
234, the pre-time-fix formula, both since changed). Re-running that anchor
is still open, same as it's been since the multithreading section above —
not done this session either.

## Multithreading (Lazy SMP) — implemented, one real bug found and fixed, internal SPRT strongly positive but not formally resolved

Item 5 from the prioritized list above (previously "deliberately deferred")
is now implemented. Trigger: the RFP-margin experiment above turned out to
be a dead end (inconclusive, then an unrelated crash), and the user asked
directly whether multithreading was worth doing instead, with the explicit
goal of moving the Stockfish@2750 anchor forward. Decision made with the
user up front: the target number is **engine at N threads vs. Stockfish@2750
at 1 thread**, tracked as a new anchor row *alongside* the existing
single-thread row, not replacing it — re-measuring that anchor is still
open, see "What's left" below.

### Design

Standard Lazy SMP: every search thread (main + helpers) gets its own
independent `Board`/`Accumulator`/`SearchStack`/history tables; the **only**
state genuinely shared between threads is the transposition table.

- `g_main_history`, `g_cont_history`, `g_capture_history`, `g_countermoves`,
  `g_pawn_corrhist`, `g_killers`, and the per-thread node/timing counters are
  all `thread_local`. Measured the cost of this before committing to it
  (converting just `g_main_history` and comparing NPS against a global):
  <2% difference, within noise — so this was the right call over a
  Worker-struct-and-thread-a-pointer-through-everything refactor, which
  would have touched ~8-10 function signatures for no measurable benefit.
- The TT (`g_tt`) is shared with no locking — the standard lockless-SMP
  design. Traced through the actual code before trusting this: a torn read
  across threads can hand `score_moves`/`singular_ext` a garbage `tt_move`,
  but that value is only ever used for a `==` comparison against moves from
  a *freshly generated* legal move list (`board.gen_moves`) — it is never
  `do_move`'d directly. The one place a raw `tt_move` gets written into a
  `pv[]` array without that check (`alpha_beta`'s `TT_EXACT` early return)
  is gated to `!is_pv` nodes, which never includes the root's real PV chain.
  So a torn TT read can waste an ordering hint or corrupt a discarded
  non-root display string, never produce an illegal move or corrupt the
  actual chosen move. Confirmed empirically too: a ThreadSanitizer build run
  under real 4-thread search found races *only* inside `tt_store`/TT-entry
  reads — zero races anywhere in the history tables, confirming the
  `thread_local` conversion has no leaks.
- `g_stop` is `std::atomic<bool>` (relaxed ordering — it's a liveness flag,
  every thread already re-polls its own node count/clock before trusting
  anything gated on it). `g_start_time`/`g_time_limit_ms`/`g_soft_limit_ms`
  are set once by the orchestrating thread *before* any worker is launched,
  then read-only for the search's duration — safe without atomics.
- No persistent thread pool: helper threads are spawned and joined once per
  `go`. Deliberate tradeoff, not an oversight — this means helper threads'
  history tables restart cold every move (only the main thread's history
  persists across the whole game), which is a real but expected-to-be-modest
  quality cost, accepted in exchange for a much simpler implementation.
  Revisit only if a future measurement suggests it actually matters.
- New `Threads` UCI option (was hardcoded `min 1 max 1`; now `min 1 max 64`).

### Bug found: helper threads were crashing (stack overflow), caught by the first real SPRT attempt

`std::thread`'s default stack size is platform-dependent and, on this
machine, dramatically smaller for a spawned thread than for the process's
main thread — measured directly: **8176KB for main vs. 524KB for a plain
worker thread**. `alpha_beta`'s ordinary recursion depth fits comfortably in
8MB but overflowed the 512KB helper-thread stack. This is exactly why 100+
single-threaded games across this whole project never surfaced it: the main
thread always had the big stack. The first Lazy SMP SPRT attempt hit this
within the first ~10 games (macOS crash reporter: `nnue_engine_mt4_candidate`,
`EXC_BAD_ACCESS` / `SIGBUS`, "Thread stack size exceeded due to excessive
recursion", deep inside `alpha_beta`'s own recursion).

**Fix**: replaced `std::thread` with raw `pthread_create` +
`pthread_attr_setstacksize(16MB)` for helper threads (`std::thread` has no
portable way to request a non-default stack size). Verified: 10 games at
Threads=8 post-fix, zero new crash reports (macOS `DiagnosticReports`,
checked directly, not inferred) vs. 2 crashes in ~7 games pre-fix.

**Lesson for future sessions**: a short TSan run or a handful of manual
`go movetime` probes at high thread counts is not sufcient to catch this
class of bug — it needs enough real recursion depth (a real game, not a
800ms-1s smoke test) to actually exhaust a small stack. Any future change
to per-node stack usage (larger `SearchStack`, bigger move-ordering arrays,
etc.) should be re-checked against this 16MB budget, not assumed safe
because single-threaded testing passed.

### Internal SPRT result: strongly positive, stopped before formal resolution

`nnue_engine` (Threads=4) vs. `nnue_engine_baseline` (Threads=1, this
project's normal single-threaded state), `tc=8+0.08`, `-concurrency 1`
(required — Threads=4 candidate + Threads=1 baseline + any additional
concurrent game pair would oversubscribe this machine's 10 cores and
invalidate the comparison), `elo0=0 elo1=30`, `-recover` enabled.

Stopped by explicit user request at 108 games (100-game checkpoint is the
last full stats block):

| Games | Elo | LOS | LLR (of the way to +30 bound) |
|---|---|---|---|
| 60 | +58.45 ± 65.47 | 96.45% | 31.7% |
| 80 | +70.44 ± 61.90 | 98.98% | 47.0% |
| 100 | **+99.95 ± 58.05** | **99.98%** | **81.8%** |

Zero crashes, zero disconnects across the entire run (post stack-size fix).
**Not formally resolved** — LLR never crossed the +2.94 bound before the run
was stopped, so there is no "H1 accepted" event to point to. But the trend
strengthened at every single checkpoint rather than decaying, which is the
opposite pattern from this project's one known false-positive precedent
(`CHECK_EXT_BUDGET=24`, which looked like +9–17 Elo at 700–1100 games and
decayed to +3.82±12.05/LLR≈0 by 1660 games — see above). Treat +100 Elo as
a strong, credible-but-not-fully-confirmed estimate, not a precise number.
If more certainty is wanted later, re-run the same SPRT command and let it
actually cross a bound before trusting a specific Elo figure.

### Current deployment state

`./nnue_engine` has been rebuilt from the current (multithreaded,
stack-fixed) `nnue_engine.cpp` and is live — this is a change from every
earlier session, where `./nnue_engine` lagged behind `nnue_engine.cpp` by
convention (candidates were built under separate names until SPRT-confirmed).
Given the strength and consistency of the trend above, the decision was made
to ship Threads=4 into production now rather than wait for full SPRT
resolution. `nnue_engine_baseline` was **not** rebuilt to include
multithreading at the time — it was kept as the pre-MT, single-threaded
reference build on purpose, so it stayed useful as the fixed comparison
point if this SPRT was ever resumed or re-run.

**Update, later session: this invariant no longer holds.** A follow-up
session's time-management fix and RFP=100 change (see "Lichess time losses"
and "RFP margin tightening" below) were both deployed to `nnue_engine_baseline`
as well as `./nnue_engine`, since the project's own standing rule ("rebuild
`nnue_engine_baseline` fresh after every commit that changes `nnue_engine.cpp`")
takes precedence for correctness-and-latest-adopted-change tracking.
`nnue_engine_baseline` is **no longer the pre-MT single-threaded reference** —
it is multithreaded, time-fix-included, RFP=100. If the Lazy SMP SPRT above is
ever resumed to seek formal resolution, rebuild a fresh pre-MT single-threaded
reference from commit `2781dc7`'s parent instead of using `nnue_engine_baseline`
as it currently stands.

The engine is also now wired up to run as a Lichess bot (`lichess-bot/`,
gitignored — see `README.md`'s "Publishing on Lichess" section and
`setup_lichess_bot.sh`), with `Threads: 4` in `lichess-bot/config.yml`.

### What's left

- **Formal SPRT resolution.** The 108-game run above was stopped early by
  request. If the number matters precisely later (e.g. before trusting a
  specific Elo figure in a public-facing claim), re-run
  `nnue_engine` (Threads=4) vs. `nnue_engine_baseline` (Threads=1) at the
  same settings and let LLR actually cross a bound.
- **The Stockfish@2750 anchor has not been re-measured with multithreading.**
  Everything above is internal (engine vs. itself at different thread
  counts) — the anchor table at the top of this document is still the
  single-thread `+61.43 Elo` result. Per the explicit decision with the
  user, the next anchor row should be **engine @ N threads vs. SF@2750 @ 1
  thread**, added alongside (not replacing) the existing single-thread row.
- **Thread count beyond 4 untested via SPRT.** Only Threads=4 was
  SPRT-tested; NPS/depth scaling was spot-checked up to Threads=8 (depth
  reached at a fixed 1s budget: 15→17→17→18 for 1/2/4/8 threads) but that's
  a proxy, not a game-based result. If chasing more Elo here, that's the
  next natural test, same protocol (40-game sanity check, then a real SPRT
  vs. the Threads=4 result, not vs. Threads=1 again).
- **Cold helper-thread history every move** (see "Design" above) — accepted
  as a reasonable v1 tradeoff, not measured directly. A persistent thread
  pool would fix it at the cost of real implementation complexity; only
  worth it if a future measurement suggests the cold-start cost is actually
  material.

## LMR reduction curve steepened: divisor 2.25 → 1.675, adopted

Triggered by the user asking directly: raw NPS is on par with Stockfish
(see the earlier NPS diagnostic section), so why does this engine reach
nowhere near Stockfish's nominal search depths (16-18) in a comparable
time budget? Answered with a direct measurement, not a repeat of the
earlier (correctly cautious, but ultimately hand-wavy) "cross-engine depth
units aren't comparable" conclusion.

**Measurement.** Same hardware, both Threads=1, Hash=64, startpos, `go
depth 18`:

| | nodes | time | NPS |
|---|---|---|---|
| Stockfish 18 | 177,777 | 120ms | ~1.48M |
| nnue_engine (pre-change) | 4,446,239 | 2,684ms | ~1.66M |

NPS is tied, but Stockfish needed **~25x fewer nodes** to complete the
same nominal depth, and the ratio climbs steadily with depth (~2x at
depth 2, ~9x at depth 13, ~25x at depth 17-18) rather than bouncing
around — a real, growing search-efficiency gap, not the noisy
non-monotonic signal the older "Cross-engine nominal-depth comparison is
invalid" section was right to distrust (that measurement used a
pre-time-fix binary and a much cruder methodology; treat this section as
superseding it for the specific claim "NPS parity ⇒ depth parity," while
that section's general caution about cross-engine depth units still
holds — this comparison only trusts the *ratio's growth trend*, not an
exact "Stockfish's depth 18 == our depth 18" claim).

**Suspected cause, read directly from the code:** `init_lmr()`
(`nnue_engine.cpp:2250-2254`) computes the LMR reduction table as
`log(d)*log(m) / 2.25 + 0.5`, and the reduction is only applied to quiet,
non-check, non-promotion moves (`nnue_engine.cpp:2679-2680`) — both
plausible, low-risk levers for a search that fans out faster than
Stockfish's per additional ply.

**Experiment, per explicit user instruction: sanity-check only, no SPRT.**
Tested four candidate divisors (all steeper than 2.25, i.e. larger
reductions), each as a standalone 40-game check (`tc=8+0.08`,
`-concurrency 4`, `-openings openings.epd order=random`) against the same
unmodified `nnue_engine_baseline` (divisor 2.25). Zero crashes, forfeits,
disconnects, or time losses across all four runs (grepped for
`disconnect|timeout|crash|terminated|illegal` in each log — none found).

| divisor | Score | Elo | LOS |
|---|---|---|---|
| 1.6 | 48.75% (8W-9L-23D) | -8.69 ± 71.11 | 40.40% |
| 1.7 | 52.50% (9W-7L-24D) | +17.39 ± 73.07 | 68.23% |
| **1.675 (adopted)** | **55.00% (12W-8L-20D)** | **+34.86 ± 72.38** | **83.26%** |
| 1.75 | 53.75% (12W-9L-19D) | +26.11 ± 86.61 | 72.76% |

**All four results are statistically indistinguishable from each other**
(±70-90 Elo error bars, heavily overlapping) — this is explicitly *not* a
resolved tuning result. Flagging this honestly rather than presenting
1.675 as confirmed-best: it has the highest point estimate and LOS of the
four, and the user made an informed call to adopt on that basis, aware
the sample size can't truly separate these four values. Same "adopted,
not formally resolved" provenance as the RFP=100 change above — if this
ever needs re-verifying, rebuild a divisor=2.25 baseline and run a real
SPRT (`elo0=0 elo1=10`) against it.

**Node-count effect, measured directly (single `go depth` calls, not
part of the game testing above):**

| position | depth | baseline (2.25) | 1.6 | 1.675 | 1.75 |
|---|---|---|---|---|---|
| startpos | 18 | 4,446,239 | 2,740,986 (-38.4%) | 4,083,192 (-8.2%) | 2,522,452 (-43.3%) |
| Kiwipete-like tactical mg | 14 | 401,562 | 362,842 (-9.6%) | 327,442 (-18.4%) | 310,639 (-22.6%) |

**Important finding, worth remembering before trying to tune this further:
node counts on a single fixed position are not a reliable proxy for
reduction aggressiveness once you're comparing nearby divisor values.**
The Kiwipete row is clean and monotonic (steeper divisor → fewer nodes,
as expected). The startpos row is not — 1.675 barely reduced nodes
despite being a steeper cut than 1.75. This isn't measurement error; it's
alpha-beta's well-known chaotic sensitivity to small parameter changes —
a slightly different reduction shifts which move gets explored first at
some shallow node, which shifts a cutoff, which cascades into a
meaningfully different tree by depth 18. Don't grid-search this parameter
by single-position node counts again; only a real game-based measurement
(and a large enough sample) means anything near an already-reasonable
value.

**Deployed:** `nnue_engine.cpp`'s `init_lmr()` divisor is now `1.675`
(was `2.25`), and both `./nnue_engine` and `nnue_engine_baseline` were
rebuilt fresh from this change (no `lichess-bot` process was running at
the time, confirmed via `ps aux` before overwriting the live binary) —
so `nnue_engine_baseline` no longer represents the pre-this-change state;
rebuild from before this change if a clean A/B reference is needed again.
Committed as `0ae1099`.

**If more Elo is wanted from this specific parameter later:** don't
repeat the one-value-at-a-time 40-game grid search — it's shown here to
be noise-limited (all four candidates landed within a ~6-point score
band of each other and of 50%, indistinguishable at this sample size).
Either commit to a real SPRT at a single suspected-best value, or invest
in SPSA-style local tuning across several parameters at once (the
approach mature engines like Stockfish actually use for this class of
constant), rather than more fixed-point sanity checks.

## Next steps for a future session: closing the search-efficiency gap further

The LMR change above is a first cut at the node-efficiency gap, not a
resolution of it — even at the adopted 1.675 divisor, this engine still
needs meaningfully more nodes than Stockfish to reach the same nominal
depth (the startpos measurement showed only an 8.2% node reduction at
1.675, well short of what 1.75's 43.3% showed, and neither has been
confirmed as a real Elo gain via SPRT). Concrete, not-yet-tried next
steps, roughly in order of expected leverage:

1. **Formally resolve whether 1.675 (or any tested divisor) is actually
   an improvement.** The 40-game checks were deliberately noise-limited
   sanity checks, not SPRTs. Before trusting any Elo claim about this
   change, run `nnue_engine` (divisor 1.675) vs. a freshly-built
   divisor=2.25 reference at `elo0=0 elo1=10` and let LLR actually cross
   a bound. `nnue_engine_baseline` no longer holds the 2.25 state (it was
   rebuilt to 1.675 alongside `./nnue_engine`) — rebuild the 2.25
   reference from `git show 3c8418f:nnue_engine.cpp` (the commit
   immediately before this session's change) if this is picked up.
2. **Extend LMR to captures.** Currently `nnue_engine.cpp:2679-2680`
   excludes every capture from reduction entirely — modern Stockfish
   reduces captures too (smaller magnitude, gated on capture history/SEE
   rather than being a flat exclusion). This is a plausible reason the
   Kiwipete (capture-heavy) node reduction was consistently smaller than
   startpos's across every divisor tested this session. Untested; would
   need its own gating logic (probably capture-history-based, mirroring
   the existing quiet-move stat-score adjustment at
   `nnue_engine.cpp:2686-2689`), its own sanity check, and its own SPRT —
   don't just flip the exclusion off without a magnitude scheme, that's
   likely to be a large regression.
3. **Re-run the Stockfish node-count comparison at scale, not on 2
   positions.** This session's "25x more nodes to reach depth 18" and the
   per-divisor node tables were both single-position spot checks — cheap
   but noisy (see the non-monotonic startpos finding above). The project
   already has the right instrumentation pattern for a real answer: the
   `prunestats` methodology from the "Node-mass-by-depth measurement"
   section (3,903 real-game positions, deduped FENs from
   `games/vs_sf2750.pgn`) measured pruning firing rates at scale — the
   same sampling approach, adapted to record nodes-to-fixed-depth instead
   of pruning hits, would give a trustworthy answer to "how much of the
   Stockfish gap is actually closed" instead of the 2-position estimate
   this session used.
4. **Consider whether other Stockfish-ported constants are similarly
   undertuned for this net**, the same way LMR's divisor and RFP's
   margin turned out to be. Candidates not yet touched this project:
   the NMP reduction formula (`R = (1062 + 68*depth)/256 + ...`,
   `nnue_engine.cpp:2417`), the cut-node/PV LMR adjustments (`r += 2` /
   `r -= 1`, `nnue_engine.cpp:2691-2693`), and ProbCut's margin. Same
   protocol each time: one bounded hypothesis, cheap sanity check,
   real SPRT before adopting — not a simultaneous multi-parameter sweep,
   which this session's noise-limited 40-game results suggest this
   project's current game-testing budget can't reliably rank anyway.
5. **If tuning several of the above together ever becomes worth it**,
   revisit the SPSA note above rather than more one-at-a-time grid
   searches — this session's four-divisor sweep used ~160 games total
   and still couldn't separate the candidates; a multi-parameter
   one-at-a-time sweep would cost proportionally more for the same
   noise floor.

## SEE-aware move-ordering candidate (item 2 above, attempted): inconclusive, not adopted

Later session, direct attempt at item 2 from the list above ("extend LMR to
captures"). Built `nnue_engine_seelmr_candidate`: SEE-aware capture scoring
in `score_moves` (good captures scored `1'000'000 + vv*6 + cap_hist`, bad/
losing-SEE captures demoted to `-1'000'000 + cap_hist`, instead of being
lumped in with quiets at a flat score) plus LMR eligibility extended to bad
(SEE<0) captures specifically, at a lighter `r = r * 2/3` reduction than
quiets get. Good (SEE≥0) captures are still excluded from LMR entirely —
this candidate only reduces the *bad* half of item 2, not the full change
modern Stockfish makes.

**Sanity check: clean on the third attempt, result inconclusive.** The
first two 40-game sanity-check attempts were contaminated by macOS system
sleep suspending the long-running background `fastchess` process mid-run
(spurious 15–30+ minute "overrun" timeouts unrelated to engine timing —
root-caused via `pmset -g log`/`pmset -g assertions`). Fixed by wrapping
the run in `caffeinate -i -w <pid> --`. Third attempt completed clean: 40/40
games finished, zero timeouts/disconnects/crashes/overruns (grepped the
log for all of those, no matches).

```
Results of seelmr vs baseline (8+0.08, 1t, 64MB, openings.epd):
Elo: 8.69 +/- 66.67, nElo: 14.21 +/- 107.67
LOS: 60.20 %, DrawRatio: 55.00 %, PairsRatio: 0.80
Games: 40, Wins: 10, Losses: 9, Draws: 21, Points: 20.5 (51.25 %)
```

No red flags (not a lopsided ~90% loss the way a genuinely broken change
looks), but also nowhere near a signal — ±66.67 Elo error bars around a
+8.69 point estimate is indistinguishable from zero at this sample size.
Per this project's own protocol a clean-but-flat sanity check like this is
normally a green light to proceed to a real SPRT, not a verdict either way
— but the node-count measurement below gave a concrete reason to doubt
this candidate before spending SPRT-scale compute on it.

**Node-count measurement (single-threaded, Hash=64, same methodology as
the LMR-divisor table above) explains why the game result was flat, not
positive: the effect is double-edged, not a clean win.**

| Position | Depth | Stockfish nodes | live `nnue_engine` nodes (ratio to SF) | seelmr candidate nodes (ratio to SF) |
|---|---|---|---|---|
| startpos (quiet opening) | 18 | 177,777 | 4,083,192 (23.0x) | 2,783,431 (**15.7x — better**) |
| Kiwipete (tactical, capture-heavy) | 16 | 122,426 | 596,043 (4.9x) | 797,131 (**6.5x — worse**) |

On the quiet position the candidate closes nearly half the node-count gap
to Stockfish. On the capture-heavy position it makes the gap *worse* than
the unmodified engine, not just less-improved. This is a plausible direct
consequence of the design: only bad captures get any reduction, and good
captures are still fully excluded from LMR (same as before) — on a
position dominated by captures, the SEE-based reordering can change which
captures get tried first without cutting the actual branching factor much,
while the extra scoring/SEE-check overhead and altered move order can
shift which lines get explored deeper. **Same "single-position node counts
are chaotically noisy near an already-reasonable value" caution the LMR-
divisor section already documented applies here too** — two positions is
not a scaled measurement — but the sign flip between a quiet and a sharp
position (not just a magnitude difference) is a more structural concern
than ordinary noise, and lines up with the game result landing flat
instead of positive.

**Verdict (user call, after reviewing both the sanity check and the
node-count table): a fail, inconclusive — not adopted.** Move ordering
alone (at least this specific, partial implementation of it) is not
confirmed to close the EBF/node-efficiency gap to Stockfish. No full SPRT
was run — the node-count sign-flip was reason enough not to spend that
compute on this specific candidate. `nnue_engine_seelmr_candidate` is left
in the repo as a reference/starting point, not merged into
`nnue_engine.cpp`; `./nnue_engine` and `nnue_engine_baseline` are
unchanged (still LMR divisor 1.675, RFP 100, per the state documented
above).

**Next-step guidance, per explicit user direction: stop iterating on move
ordering as the lever for the EBF gap; look at more/different pruning, or
consider that an actual bug (not just an undertuned parameter) may be
contributing to the ~5–23x node-count gap.** This reframes item 2 above
from "extend LMR to captures" (attempted, inconclusive) toward two
untried directions:

- **More/different pruning**, not just re-tuning existing margins (RFP/NMP/
  ProbCut margins are already documented as tuned or ruled out above) —
  e.g. history-based pruning gates not yet present, or futility/LMP
  extended further than currently scoped, evaluated the same way (bounded
  hypothesis, sanity check, real SPRT).
- **A genuine bug**, not a tuning gap, as a live hypothesis for the node-
  count blowup. Nothing specific has been found yet — this is flagged as
  worth auditing, not a confirmed finding. Candidate places to look, given
  what's already been ruled out elsewhere in this document (TT, IIR,
  history gravity, bucket formula, LMR sophistication are all previously
  confirmed clean — don't re-audit those): the move-loop's interaction
  between pruning/reduction gates and re-search triggers (an overly broad
  full-depth re-search condition would silently re-inflate the tree after
  a reduction, the same *class* of bug as the ungated-re-search timing bug
  found earlier this project, just for node count instead of clock time),
  and whether TT cutoffs are actually firing at the rate expected for a
  ~5-23x-worse-than-Stockfish tree (an unexpectedly low TT hit rate at
  shallow depth would point at move-ordering/replacement-scheme issues
  rather than pruning aggressiveness). Not yet investigated — next session
  should instrument and measure before guessing further.

## Futility/LMP tightening: node-count reduction confirmed real, but game-tested negative twice — reverted, not adopted

Direct follow-up to the node-count-gap investigation above, later session.
User's framing: node counts, not Elo, are the metric to chase first — build
node-count evidence at scale before spending any game-testing budget, and
don't run games without an explicit go-ahead (the session opened with a
sharp correction after an earlier candidate's sanity check was launched as
three separate foreground `fastchess` calls that each hit Bash's 2-minute
timeout — the fix, followed for the rest of the session, was `nohup ... &`
plus the `Monitor` tool's `until ! kill -0 <pid>` pattern for anything
long-running).

**RFP was left alone this pass** (explicit user instruction — "RFP seems
tight enough right now"), consistent with its own already-thin headroom
documented in the RFP=100 section above. Attention went to futility pruning
and Late Move Pruning (LMP) instead, per direct user request. A first
attempt at a different lever — negative/reduced singular extensions
(mirroring Stockfish's extension-budget mechanics) — was tried and
abandoned early: only a 4% median node reduction on a node-count sweep,
called out directly by the user as insufficient, and fully reverted from
the source before this section's work began.

### Method

New `nodecount_sweep.py` (checked into the repo root, reusable) compares
two engine binaries' node counts at a fixed `go depth N` across sampled
real-game FENs (same PGN/dedup/ply-window convention as the project's
existing `prunestats` sampling) and reports median/mean/IQR of the ratio —
a fast, game-free way to screen a search-tuning candidate before spending
any `fastchess` budget, consistent with this project's established
protocol.

Extended the existing `prunestats` instrumentation (env-gated, additive,
zero cost when off) with a candidate-scale simulation for futility and LMP,
mirroring the pattern already used for RFP's margin coefficient: alongside
the real (currently active) pruning decision, the same node also computes
what several *candidate* tighter scale factors *would* have decided,
without changing actual search behavior. This let the headroom of both
mechanisms be measured at scale (300 real-game positions) before building
any real candidate binary.

**Finding: futility pruning has substantially more unsaturated headroom
than LMP.** Simulated scale factors 1.0 down to 0.25 relative to a given
baseline: futility's skip count kept climbing all the way to 0.25 with no
sign of flattening (+51% skips at 0.25 vs. the reference), while LMP's
skip count visibly flattened past roughly 0.55-0.7 (+21% at 0.55, only
+21% more by 0.25 — diminishing returns setting in much earlier than for
futility). This shaped the scale choices below: LMP was tightened less
aggressively than futility whenever the two were varied together.

### Node-count sweep results

Built real candidate binaries (`FUTILITY_MARGIN` and `LMP_MOVES` both
scaled by a single factor from their original values — `{0,100,200,300,400}`
and the original depth-indexed LMP table respectively) at six scales,
200-250 sampled positions each, depth 12, vs. the (untouched, RFP=100)
baseline:

| scale | median ratio | mean | IQR | improved / worse |
|---|---|---|---|---|
| 0.55 | 0.822 | 0.903 | [0.581, 1.064] | 70.5% / 29.5% |
| 0.60 | 0.819 | 0.914 | [0.623, 1.100] | 69.0% / 30.5% |
| 0.65 | 0.824 | 0.902 | [0.617, 1.050] | 70.0% / 30.0% |
| 0.70 | 0.821 | 0.941 | [0.648, 1.165] | 64.0% / 35.5% |
| 0.75 | 0.840 | 0.957 | [0.644, 1.172] | 64.0% / 36.0% |
| 0.80 | 0.898 | 1.005 | [0.680, 1.228] | 57.0% / 42.5% |

0.55-0.65 were statistically indistinguishable from each other on median
(~18% node reduction, n=200 — below this project's own established noise
floor for chasing sub-percent node-count differences) but clearly better
than 0.70 on both mean and IQR tail — 0.65 was picked as the "sweet spot"
(tied for best median, best mean, tightest regression tail) and carried
forward as the leading candidate. A separate, more aggressive probe
(futility scaled to 0.35x original, LMP to 0.45x — pushed further after an
explicit "more aggressive" request) showed an even larger node reduction
(24.1% median vs. baseline, 9.8% further reduction vs. the 0.65 candidate)
but was set aside per direct user instruction to instead search the
neighborhood around the already-promising 0.65-0.70 area rather than
continue pushing toward the extreme.

### Game-tested: negative at two different magnitudes, with no recovery at the lighter one

Two 40-game sanity checks (`tc=8+0.08`, `-concurrency 4`, `-recover`,
`openings.epd order=random`), each vs. the unmodified (RFP=100, LMR=1.675)
baseline:

| candidate | Score | Elo | LOS |
|---|---|---|---|
| 0.65x (futility+LMP together) | 41.25% (5W-12L-23D) | -61.43 ± 81.19 | 6.13% |
| 0.85x (lighter touch, futility+LMP together) | 40.00% (4W-12L-24D) | -70.44 ± 64.26 | 1.27% |

Neither result is the "wildly lopsided, ~90%-loss" pattern this project
treats as an unambiguous bug signal — but both are a clear, consistent
lean negative, and **the lighter-touch 0.85x candidate did not recover
toward parity relative to 0.65x — if anything its point estimate was
slightly worse, and its LOS dropped from 6.1% to 1.3% (i.e. higher, not
lower, confidence that it's a real loss).** Two independent 40-game
samples (different random opening draws each), at two different
tightening magnitudes, both landing solidly negative without the expected
"back off the aggressiveness and it gets better" pattern, is a real signal
that tightening futility and LMP *together* costs more search quality
than the node savings are worth in this range — not simply a matter of
picking a gentler scale. Consistent with, and a concrete instance of, this
document's own standing caution (first raised in the RFP=234→165 section
above): a cheap node-count/firing-rate signal is not a reliable proxy for
Elo when tightening forward-pruning margins — it must be game-tested
before adopting, no matter how clean the node-count evidence looks.

**Decision: reverted, not adopted.** `nnue_engine.cpp`'s `FUTILITY_MARGIN`
and `LMP_MOVES` are back to their original, untouched values
(`{0,100,200,300,400}` and the original depth-indexed LMP table). RFP
remains at `100` (never touched this pass). `./nnue_engine` and
`nnue_engine_baseline` were rebuilt fresh from this reverted state — both
are byte-for-byte HEAD (`0ae1099`) again, i.e. this entire investigation
ends as a no-op on the live/production binaries.

**If this is revisited:** don't retry "tighten futility+LMP together" at
some other single scale — two data points in this range already argue
against that framing. The one untried, better-motivated next step
(discussed but not run, since the session ended here) is to **isolate
futility and LMP from each other** — tighten only one at a time at a
moderate scale (e.g. 0.85x) and game-test each separately, since this
session's two tests never separated the two mechanisms' individual
contribution to the loss. It's possible one of them (not both) is the
actual problem, or that the *combination* specifically is what costs
strength even though neither alone would. `nodecount_sweep.py` and the
`prunestats` candidate-scale instrumentation (both still in the repo) are
ready to reuse for that without rebuilding any tooling.

**Tooling kept from this session** (all additive, reusable, no effect on
default engine behavior): `nodecount_sweep.py` (repo root) and
`depth_nodecount_compare.py` (repo root) — the latter reconfirmed, on a
larger and more careful re-run (150 real-game positions, `maxdepth=16`,
sampled the same way as `prunestats`) than the small 20-position spot
check that originally suggested an ever-compounding node-count gap, that
**the node-count gap to Stockfish plateaus at roughly 3.5-4x from around
nominal depth 10 onward rather than compounding without bound** — it looks
"set" by around depth 6-10 and roughly flat after that. This corrects the
earlier small-sample impression (up to 23-28x, still growing at depth 16)
as noise-driven, not a real trend — re-run `depth_nodecount_compare.py`
directly if exact per-depth numbers are needed again, they weren't
preserved verbatim from this session's run.

## Polyglot opening book: real UCI-level support, now the engine's standard book mechanism

Earlier work (a prior session) added opening-book support only at the
lichess-bot Python layer (`get_book_move()` called before any UCI `go`) —
invisible to fastchess/UCI, so it could never be exercised by this
project's own SPRT harness. The user asked directly for **"an actual
measurement"** of the book's effect, which required building real
Polyglot support into the engine's own UCI interface so fastchess could
exercise it like any other engine feature.

### What was built

`nnue_engine.cpp` gained standard UCI `OwnBook` (`type check`, default
`false`) and `BookFile` (`type string`) options, plus a `polyhash` debug
command (mirrors the existing `eval` debug command). Implementation:

- The full standard 781-entry Polyglot `POLYGLOT_RANDOM_ARRAY` (Fabien
  Letouzey's original table), copied verbatim from python-chess's
  `chess/polyglot.py`.
- `polyglot_hash(Board)` — independently reimplements python-chess's
  exact Zobrist scheme (per-piece entries, castling-rights entries, an
  en-passant entry included only when a pawn can actually capture there —
  not just because `ep_square` is set — and the side-to-move entry).
  **Verified byte-exact against `chess.polyglot.zobrist_hash()` on 8 test
  cases** (`verify_polyglot_hash.py`), including both-side castling rights
  and the ep-square-set-but-not-actually-capturable edge case.
- `load_polyglot_book(path)` — parses the standard 16-byte big-endian
  entry format, sorted by key for binary search.
- `probe_book(Board)` — binary search by key, weighted-random selection
  among matching entries (uniform weights currently — no per-move
  engine-strength bias yet, see the earlier opening-book section's
  rationale for why), decodes the raw move and cross-checks it against
  the position's actual legally generated moves to recover castle/ep/
  promotion flags, returns "no move" defensively on any mismatch.
- Book consult is gated identically to the existing timed-game branch in
  the `go` handler (`wtime`/`btime` present, not `infinite`/`movetime`/
  `go depth`/`go nodes`) — fixed-depth analysis and `go depth`/`movetime`
  queries always search for real, matching how the lichess-bot Python
  book only ever fired during real play.

**One real bug found and fixed during this work**: the from/to square
decode was initially backwards (`from = raw & 63; to = (raw>>6) & 63`,
should be the reverse per python-chess's actual encoding) — this would
have made the book silently non-functional, always falling through to
search, despite the hash lookup itself working correctly. Caught by
manual UCI testing (a book-covered position returned `bestmove 0000`
instead of a real move) before it ever reached a sanity check or SPRT.

**Second issue found and fixed**: the book's coverage stopped exactly at
the same point `openings.epd`'s 49 FENs sit (both are generated from the
same `LINES` dict in `gen_opening_book.py`), so any fastchess run started
from `openings.epd` would fall out of book on the opponent's very first
move — largely defeating the point of testing it in an SPRT context.
Fixed by extending `build_polyglot_book.py` with an `EXTRA_CONTINUATIONS`
dict (4-6 more half-moves of hand-typed theory per line, past
`openings.epd`'s stopping point, `openings.epd`/`gen_opening_book.py`
themselves untouched) and regenerating `lichess-bot/engines/book1.bin`
(258 → 450 entries).

**Verification**: `verify_book_coverage.py` drives the real engine binary
over UCI and confirms, across all 403 book-covered positions spanning all
49 lines, zero wrong/non-book moves and zero unexpectedly-slow
(>50ms, i.e. fell through to search) responses — PASS. A direct check
also confirmed all 49 `openings.epd` FENs now get an instant book
response rather than falling out of coverage immediately.

### Measurement: sanity-checked, not SPRT-resolved

A 40-game sanity check (`tc=8+0.08`, `-concurrency 4`,
`-openings file=openings.epd order=random`, book-enabled candidate vs.
book-disabled baseline, otherwise identical binaries) completed cleanly —
zero crashes/disconnects/illegal moves/timeouts — with **+8.69 ± 71.11
Elo**, 9W-8L-23D, for the book-enabled side. This lands squarely inside
the predicted "pure clock-banking" range (book moves are free against an
8s clock, with no assumed opening-quality edge since move weights are
uniform by design) and shows no red flags, but the error bars are far too
wide at n=40 to call this a result on its own.

**A real SPRT was launched to get an actual resolved number, then killed
mid-run by explicit user interrupt ("okay stop") before any meaningful
data accumulated.** No SPRT-confirmed Elo figure for the book exists. If
one is wanted later, re-launch `nnue_engine` (OwnBook=true) vs.
`nnue_engine_baseline` (OwnBook=false) at `elo0=0 elo1=10`,
`tc=8+0.08`, `-openings file=openings.epd order=random`, `-recover` —
same protocol as every other SPRT in this document.

### Decision: shipped as the engine's standard book mechanism anyway

Per explicit user instruction ("make the book UCI the new thing the
engine uses and that's it"), the UCI-level book is now this project's
standard book mechanism, adopted without waiting for the SPRT above to
resolve — the sanity check showed no correctness risk (zero anomalies
across 40 games) and the clock-banking rationale for a modest positive
effect is sound even without a tight Elo number.

- `./nnue_engine` and `nnue_engine_baseline` were both rebuilt from the
  current `nnue_engine.cpp` (which now includes the book module) and are
  otherwise unchanged from their prior state (LMR divisor 1.675, RFP 100,
  Lazy SMP, all prior fixes) — `OwnBook` still defaults to `false` at the
  UCI level (standard opt-in-by-`setoption` behavior, consistent with
  every other option this engine exposes), so nothing about raw binary
  behavior changed without an explicit `setoption`.
- **`lichess-bot/config.yml`** (gitignored, documented here per this
  project's standing convention for that directory): the old Python-layer
  `polyglot.enabled` was flipped to `false` (it would otherwise silently
  intercept every book move before the engine's own UCI book ever saw the
  position — redundant now, not a second layer of defense), and
  `uci_options` gained `OwnBook: true` / `BookFile: <absolute path to
  book1.bin>` so the live lichess bot now sources book moves through the
  engine's native UCI path instead.
- `book_sanity_check.py` (the earlier session's Python-layer validator)
  is left in the repo as a reference for that now-disabled code path, not
  deleted — it still exercises real code (`lib.engine_wrapper`) that
  remains in `lichess-bot/`, just no longer the active book route.

### Files added this session

`build_polyglot_book.py` (rewritten — `EXTRA_CONTINUATIONS`),
`verify_polyglot_hash.py`, `verify_book_coverage.py`,
`book_sanity_check.py` (restored after an accidental deletion mid-session
— rewritten to note the now-superseded status of the Python-layer path
it validates).

## Bullet time-drain diagnosis: two real causes found, both fixed (partial fix, not full resolution)

The user reported the lichess bot spending the majority of a bullet clock
(50 of 60s) in just the first 20 moves and asked why. Diagnosed with two
distinct, real, independently-verified causes — not a single root cause —
both in the time-management/book path, not the search itself.

### Cause 1: best-move-instability stretch fires on almost every move, not just genuinely volatile ones

`iterate()`'s instability heuristic (`nnue_engine.cpp`, in the iterative
deepening loop) multiplies the move's effective soft time budget by 1.3x
every time the best move changes between two consecutive completed
iterations at depth ≥5, capped at `hard_limit = min(myTime/4,
soft_limit*2.5)`. The trigger was a bare best-move label change — no check
on whether the position's evaluation actually moved. Verified directly via
a real bullet self-play game (`NNUE_TIME_DEBUG=1`, 60+0, actual measured
think time decrementing the clock each move): once past the book, nearly
every move landed at 1.4x-2.5x its nominal `myTime/movestogo` share, with
several hitting the hard cap outright — by move 20, roughly 34-40s of a
60s clock was already gone.

Instrumented every instability-candidate event (added a temporary
`NNUE_INSTAB_DEBUG=1` env-gated print, since removed) across 205 events
sampled from 12 diverse `openings.epd` positions: median score delta at
trigger time was just **6cp**, and **83% of events were under 20cp** —
i.e. the overwhelming majority of "instability" was cosmetic reordering
among near-equal quiet moves (search noise), not a real re-evaluation of
the position. Genuine large swings (up to 118cp in this sample) were rare
(~5% of events above 40cp).

**Fix**: added `INSTABILITY_SCORE_DELTA = 20` (cp) and require the score
to have moved by at least that much, in addition to the best-move label
changing, before applying the 1.3x stretch (`nnue_engine.cpp`, `iterate()`
and the new constant near `CHECK_EXT_BUDGET`). This keeps the safety net
for genuinely volatile positions while filtering out the noise-driven
triggers that were dominating in practice.

### Cause 2: lichess-bot's first move bypassed the book entirely and burned a full fixed-time search

`lichess-bot/lib/engine_wrapper.py`'s `first_move_time()` (gitignored, not
in git history, noted here per this project's standing convention) sends
`go movetime X` — not `wtime`/`btime` — for **both** sides' very first
move of every game (`len(board.move_stack) < 2`), where `X = min(10s,
max(0.5s, base_time*0.05))` (3000ms for a 60s bullet base). The engine's
`OwnBook` consult gate (`nnue_engine.cpp`, the `go` handler) required
`wtime`/`btime` to be present, on the documented assumption that
lichess-bot always sends those for real play and only sends `movetime`
for fixed-time analysis queries. **That assumption was wrong** — verified
by reading `engine_wrapper.py` directly, not inferred. The practical
effect: every real game's first move for both colors silently skipped the
book (even though the position was book-covered) and ran a full,
non-time-managed fixed search instead (`soft_limit = -1` for movetime-based
searches, no early exit) — a straight, unconditional **3s burned on a
60s bullet clock (5% of the entire game) on a move the book already
answered for free**, on move 1, before the game had meaningfully started.

**Fix**: broadened the book-consult gate to fire on any real-time-bounded
`go` (movetime **or** wtime/btime present), excluding only unbounded
analysis (`go infinite`/`go depth`/`go nodes`). Verified directly: `go
movetime 3000` from `startpos` with `OwnBook=true` now returns the book
move in 0ms, matching exactly what lichess-bot sends for a 60s-bullet
first move.

### Measured effect: real but partial — read the numbers carefully, don't over-claim

Built a probe (`prod_probe.py`, not checked in) that reproduces the actual
lichess-bot flow exactly: first ply of each side via `go movetime`
(matching `first_move_time()`'s formula), every subsequent move via `go
wtime/btime`, `Threads=4` (matching `config.yml`), `OwnBook=true`. Ran 2
self-play games each, pre-fix vs. both-fixes-applied, from `startpos`:

| | avg time spent by White's move 20 | avg time spent by Black's move 20 |
|---|---|---|
| pre-fix | 38,691 ms | 39,798 ms |
| both fixes | 35,872 ms | 34,992 ms |
| reduction | -7.3% (-2,819ms) | -12.1% (-4,806ms) |

**This is a real, directionally-confirmed improvement, but a partial one,
not a full resolution — say so honestly if this comes up again.** n=2 runs
per side is small and self-play at `Threads=4` has some run-to-run
variance (Lazy SMP's unlocked TT reads aren't fully deterministic). The
event-count reduction from the score-delta gate (83% of trigger events
removed) does **not** translate to an 83% wall-clock reduction, because
stretch events compound multiplicatively (1.3x per surviving event) and
clip at the 2.5x hard cap — a move that previously had several
noise-level events, all now filtered, drops from a capped ~2.5x multiplier
to something much closer to 1.0-1.3x, which is a large per-move win: but a
move that already had exactly one genuine large-delta event is completely
unaffected by design, since the fix is specifically built to preserve that
case. The two effects (book fix + instability-gate fix) are bundled in
the table above, not isolated from each other.

**A structural cause was identified but deliberately left untouched**:
`movestogo` is hardcoded to `30` in the `go` handler regardless of actual
game length or ply count (lichess never sends it). For a bullet game that
runs materially longer or shorter than 30 total moves per side, this
alone front- or back-loads time allocation independent of anything fixed
this pass. Not touched this session — flagged as the next concrete lever
if more of this gap needs closing, but it changes the *pacing* formula
itself (not just how far a single move can overshoot it, which is what
both fixes above touched), so it needs its own dedicated
measurement-then-fix pass, not a bundled guess.

### Verification methodology note: the project's standard `tc=8+0.08` SPRT harness cannot see this class of fix

Same structural blind spot already documented for the original
"Lichess time losses" fix above: at `tc=8+0.08` (~8s base), the clock
never gets low enough during a game for either of these bugs to matter —
a fixed 3s first-move burn or a 2.5x-capped stretch are comparatively
tiny against an 8s budget that resets every move via increment. A 40-game
sanity check (`sanity_timefix2.log`, both-fixes candidate vs. a pre-fix
baseline, `tc=8+0.08`, `-concurrency 4`, `-recover`) completed cleanly —
40/40 games, zero crashes/disconnects/forfeits, 11W-6L-23D, Elo +43.66 ±
64.91, LOS 91.12% — but per this project's own established protocol this
is a **did-I-break-anything gate only, not an Elo verdict**: the error
bars comfortably include zero, and the harness that produced this result
structurally cannot exercise the actual bug being fixed. Don't cite
+43.66 as a confirmed Elo gain if this comes up again. If a real Elo
number is ever wanted for this specific class of fix, it would need a
longer/lower base time control (or a real low-time endgame test) to
actually stress the clock the way a real bullet game does — not
attempted this session.

**Operational note, logged for the next session**: while the sanity-check
`fastchess` run above was in flight, the candidate binary it referenced
(`nnue_engine_instabilityfix_candidate`) was rebuilt in place to add the
second (book/movetime) fix — a direct violation of this project's own
"never rebuild a binary an active run references" rule, made worse by
that run using `-recover` (which can respawn engine processes mid-run and
pick up the rebuilt file). Caught before trusting the result: the run was
killed, the combined-fix binary was renamed to a fresh path
(`nnue_engine_timefix2_candidate`), and the sanity check was re-run
cleanly from scratch against that fixed name. The result quoted above is
from that clean re-run, not the contaminated one.

### Deployment

Both fixes are committed to `nnue_engine.cpp`. `./nnue_engine` and
`nnue_engine_baseline` were both rebuilt from this state (no
`lichess-bot` process was running at the time, confirmed via `ps aux`
before overwriting) — so both now include: the score-delta-gated
instability stretch, the broadened book/movetime gate, and everything
previously documented above (Lazy SMP, RFP=100, LMR divisor 1.675, native
UCI Polyglot book, the original time-management overrun/low-clock fixes).

## Stockfish-style time management: three mechanics added, one tried and reverted; a real `stop`/pondering bug found but not fixed

Triggered by the user reporting that even blitz games (not just bullet)
consistently drain most of the clock down to a small reserve, and asking
directly how Stockfish manages time efficiently. Answered conceptually
first (continuous stability/score-trend scaling instead of a binary
trigger, an "obvious move" fast exit, pondering), then asked to implement
all of it and verify carefully.

### What was built

`nnue_engine.cpp`'s `iterate()` (the per-move iterative-deepening loop) and
its constants near `INSTABILITY_SCORE_DELTA` gained three mechanics, all
gated on `soft_limit_ms >= 0` (i.e. only active for real wtime/btime play,
same convention as every other early-exit in this function -- fixed
`go movetime`/`go depth`/`go infinite` are unaffected):

1. **Continuous instability-stretch dial** (`INSTABILITY_STRETCH_MIN=1.10`,
   `INSTABILITY_STRETCH_MAX=1.35`) -- replaces the old flat 1.3x-per-event
   multiplier with one scaled by how far the score actually moved. A delta
   right at `INSTABILITY_SCORE_DELTA` gets the gentle end; only a delta at
   or beyond 4x that threshold reaches the ceiling. Deliberately built so
   an ordinary borderline event ends up *cheaper* than the old flat 1.3x,
   not just smoother -- CLAUDE.md already documents the flat version as
   the direct cause of clock-draining stretches on near-equal quiet-move
   label flips, so a "refinement" that stretches more on average would be
   a regression dressed up as one.
2. **Falling-eval extension** (`FALLING_EVAL_DELTA=30`,
   `FALLING_EVAL_STRETCH=1.15`) -- extends thinking time when the score has
   declined for two consecutive completed iterations even though the best
   move's label hasn't changed, mirroring Stockfish's real behavior (which
   reacts to a worsening trend, not just a label flip). Softer signal than
   a label change, so a smaller single-shot stretch; the two mechanics
   don't compound in the same iteration (at most one fires per iteration).
3. **Single-legal-move fast exit** -- computed once per `iterate()` call
   (every thread, main and helpers, reaches the same conclusion
   independently since they see the same board, so no cross-thread
   coordination is needed for this specific case): if there's exactly one
   legal move, run one depth-1 search for a real score/PV, then stop.
   Verified: a real forced-move test position (`7k/8/8/8/8/8/6Q1/6RK b - -
   0 1`, one legal move) went from what would have been a ~2000ms
   allocation to 0ms, correct move returned.

### Tried and reverted: the fourth mechanic ("obvious move" stability-based fast exit)

A fourth mechanic, closer to Stockfish's actual "obvious/forced move" fast
exit, was also built and isolation-tested: track `stable_iters` (consecutive
completed iterations, from the same `d>=5` noise floor as the instability
trigger, where the best move's label didn't change AND the score moved by
less than `EASY_MOVE_QUIET_DELTA=15`cp); once `stable_iters` reached 6 at
`d>=10`, stop deepening immediately and signal `g_stop` (required --
without it, a helper thread's independent `effective_soft` means
`search()`'s `pthread_join` would still block on whichever helper takes
longest, and the main thread's early exit alone saves zero wall-clock
time).

**Isolated via a controlled A/B, not just a single combined test** (per
this project's own precedent -- "don't retry tighten-X-and-Y-together;
isolate them", from the futility+LMP episode below): a 40-game sanity
check with all four mechanics active scored 41.25% vs. baseline (Elo
-61.43 +/- 76.99, LOS 5.21%); an otherwise-identical 40-game run with only
this fourth mechanic disabled (same dial + falling-eval + single-legal-move)
scored a clean 50.00% (Elo 0.00 +/- 64.43). This isolates the regression
to this one mechanism specifically, not the other three. Confirmed by hand
too: it cut a genuine, still-contested middlegame search
(`r2q1rk1/ppp2ppp/2np1n2/2b1p3/2B1P1b1/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 8`)
to just 18% of its allotted soft budget (depth 20/544ms vs. an effective
soft limit that had grown to ~3034ms) purely because the label+score had
looked "settled" for 6 iterations -- evidently too weak a signal on this
engine; Stockfish's real equivalent gates on root-move *node-share*
dominance, which this engine has no per-root-move instrumentation for.
**Reverted, not re-tuned** -- consistent with this project's RFP/futility/
LMP tightening history, where one-at-a-time threshold grid searches on an
already-shown-negative heuristic rarely recover it. The dead constants and
logic are removed from `nnue_engine.cpp`; a comment at the `RFP_MARGIN`
declaration site records the isolation-test numbers and the node-share
alternative for anyone revisiting this.

### Real, unrelated bug found: `go infinite` + `stop` does not work at all

While scoping out whether to also implement pondering, found (and
reproduced directly, not inferred) that **the UCI main loop cannot process
a `stop` command while a search is in flight.** `main()`'s loop is
`while (std::getline(std::cin, line))` -- fully synchronous, one line
processed to completion (including blocking inside `engine.search()`)
before the next line is even read. Sent `go infinite` immediately followed
by `stop` by hand: the engine kept iterating past depth 19 and had to be
force-killed; it never once reads the `stop` line because the loop is
still blocked inside the first `go`'s `search()` call, which for
`go infinite` (`hard_limit=-1`, so `out_of_time()`'s time check never
trips) only terminates by reaching `depth=64`, i.e. effectively never in a
real position. This is also the actual architectural prerequisite for
pondering (which needs the engine to keep searching in the background
while remaining responsive to `stop`/`ponderhit`) -- not implemented this
session, deliberately scoped out (see below).

**Not fixed this session.** A correct fix means running `search()` on a
background thread from the `go` handler and having the main loop keep
reading stdin concurrently, which collides with a documented invariant
elsewhere in this file: `g_start_time`/`g_time_limit_ms`/`g_soft_limit_ms`
are "set once before any worker is launched, then read-only for the
duration of the search" -- true today because nothing hands the main
thread control back until the whole search (all helpers included) is
done. A real fix needs to preserve thread-safety for that invariant
(making `g_time_limit_ms`/`g_soft_limit_ms` atomic is the likely fix,
since `out_of_time()`'s hot-path read is already gated behind a
once-per-4096-nodes check -- negligible cost either way) and design
`ponderhit`'s semantics (convert an in-flight infinite ponder search into
a real time-managed one without restarting it and losing the work already
done). Flagged to the user as a separate, larger, higher-risk follow-up;
not started.

### Verification and deployment

Verified directly (not inferred) that neither new stretch mechanism can
violate the hard cap: hand-fed `NNUE_TIME_DEBUG=1` at bullet-shaped low
`wtime` values (100/200/500/1000/60000ms) showed 0-1ms overrun throughout
-- the same noise floor this project already treats as clean (see the
original time-management-overrun section above). `search_wall` was also
confirmed to actually drop on both the single-legal-move case (0ms) and
a real endgame position that settled quickly (8ms, was previously
projected to spend up to the full ~2000ms soft budget) -- i.e. the
`g_stop`-on-early-exit signaling actually saves wall-clock time across
threads, not just on the reporting thread.

**Elo signal at this project's `tc=8+0.08` fastchess anchor: flat, not
resolved as a win.** Two 40-game sanity checks plus a real SPRT
(`elo0=0 elo1=10`) totaling ~1050 games (stopped by explicit user request
at 978 games, never crossed either LLR bound) all landed within noise of
50% -- final tally 49.08% / approx -6.4 Elo. Zero crashes, zero time
losses, zero lopsided results across the entire ~1050 games, including the
SPRT's real games, which (worth being precise about, since this was
initially mis-stated mid-session) **were genuine full games played through
a live, continuously-decrementing UCI clock from move 1 to the end**
(fastchess sends real `wtime`/`btime`, same protocol path lichess-bot
uses) -- not a fixed-movetime or truncated test. `tc=8+0.08` is in fact
*faster* than real lichess bullet, so if anything these games hit low-clock
territory sooner than a real 60s+ bullet game would. **Do not repeat the
mid-session claim that this anchor "structurally can't see this class of
fix"** -- that reasoning is validly documented elsewhere in this file for
two *other*, narrower bugs (the `myTime/2` safety-cap defeat, lichess-bot's
first-move `movetime` bypass) with specific, verified trigger conditions
that really don't occur at this tc; it does not automatically transfer to
general time-allocation-shape changes like these three, and restating it
without re-verifying was a mistake caught only because the user asked a
direct, specific question about the test methodology. The defensible
reading of the flat result: either these three mechanics have a genuinely
small net effect at this specific (very fast) speed, or a real small
effect is under-resolved at ~1050 games -- this project's own
`CHECK_EXT_BUDGET=24` precedent took ~1660 games to fully resolve a
small effect down to noise, so 1050 games is not dispositive either way.

**Deployed anyway, on safety + direct verification, not an Elo claim** --
same category of decision as the original low-clock/overrun time fixes and
the Polyglot book: zero regressions/crashes/time-losses across every test
run, the underlying logic independently hand-verified correct, and the
change directly targets a real, user-reported problem (clock draining to a
small reserve) that a fast fixed-tc anchor is, at minimum, not positioned
to rule out helping with even if it can't confirm it here. `./nnue_engine`
and `nnue_engine_baseline` were both rebuilt from this state (no
`lichess-bot` process running at the time, confirmed via `ps aux` first)
-- both now include the three mechanics above on top of everything
previously documented in this file. If a tighter Elo answer is wanted
later, a real bullet/blitz time control closer to what the lichess bot
actually plays (not `tc=8+0.08`) is a more direct test than re-running
this same anchor.

## UCI_Elo bisection: engine strength lands at ~2900 on the `tc=8+0.08` anchor -- but this number is disputed, not final

Triggered by the user directly challenging two things at once: (1) whether
comparing this engine against **unlimited, full-strength** Stockfish 18
(dropping `UCI_LimitStrength` entirely, as briefly considered) would give a
useful number, and (2) later, whether the whole `tc=8+0.08` anchor is even
an accurate test given how fast it is, and why it disagrees with the live
lichess bot's own measured ~2600 rating.

### Why full-strength Stockfish was ruled out before testing it

Reasoned, not measured (no games were run against unlimited SF18): any
opponent 400+ Elo above the true rating produces a score statistically
indistinguishable from an opponent 1000+ Elo above it, because the
logistic Elo score curve saturates near 0% well before the gap gets that
large. A near-100% loss rate against full-strength SF18 (rated 3600+)
would confirm "much weaker" and nothing more precise than that -- it
cannot distinguish a true rating of 2900 from a true rating of 2000. An
Elo estimate needs an opponent *close* to the engine's own level, which is
exactly what `UCI_LimitStrength`/`UCI_Elo` at a chosen target provides.

### Method: bisect `UCI_Elo` upward from the existing 2750 anchor point

The existing anchor table at the top of this document already had one
real, non-degenerate data point: `UCI_Elo=2750`, 58.75% score (200 games),
+61.43 +/- 42.49 Elo -- i.e., this engine is confirmed stronger than 2750
on this scale. Rather than pick a single new target and hope it's close,
the right move (and the standard way rating lists actually work) is to
walk `UCI_Elo` upward until the score falls back toward ~50% -- that
crossing point is the real estimate. Confirmed the bundled Stockfish's
usable range first: `UCI_Elo min=1320 max=3190`.

Ran one rung at `UCI_Elo=2900`: `nnue_engine` at its real deployed
configuration (Threads=4, Hash=512, `OwnBook=true` with the real
`book1.bin`, matching `lichess-bot/config.yml` exactly) vs. Stockfish 18
(`UCI_LimitStrength=true UCI_Elo=2900`, Threads=1, Hash=64), same
`tc=8+0.08 timemargin=200` anchor TC, 50 games, `-concurrency 2`,
`openings.epd order=random`, `-recover`. Zero crashes/disconnects/time
losses across the full run (grepped the log for all of those, no matches).

**Result: 20W-21L-9D, 49.00% score, Elo -6.95 +/- 93.56, LOS 44.07% --
statistically dead even.** Combined with the existing 2750 point:

| `UCI_Elo` target | Score | Elo |
|---|---|---|
| 2750 | 58.75% (200 games) | +61.43 +/- 42.49 |
| **2900** | **49.00% (50 games)** | **-6.95 +/- 93.56** |

Comfortably above 2750, statistically even at 2900 -- the crossing point,
and this project's best current strength estimate on this anchor, is
**~2900**, not the previously-unset "~3000 goal."

**A live example of this project's own standing caution, observed while
this exact rung was still running:** the in-progress read at n=31 games
was 41.9% / -56.5 Elo (looked like a clear loss); by n=50 it had swung all
the way to 49.00% / -6.95 (dead even). Another instance of "don't trust a
mid-run number," this time caught and flagged in real time rather than
after the fact.

### Why ~2900 should not be trusted as a precise or final number

Raised directly by the user, and the concerns are valid, not just
hand-wringing:

1. **`tc=8+0.08` is an 8-second-per-side anchor, chosen for cheap
   iteration, not for realism.** Three concrete reasons this specific
   test undermines an absolute strength claim:
   - `UCI_LimitStrength` is already documented elsewhere in this file to
     **under-limit Stockfish at fast time controls** -- so the "2900"
     label itself is less trustworthy exactly at this speed, likely
     playing a bit stronger than a real 2900 would.
   - Fixed per-move overhead (process/IPC, movegen setup, book lookup, TT
     bookkeeping) eats a much bigger fraction of an 8-second budget than a
     60+-second one -- more noise relative to signal.
   - Every real low-clock bug this project has found and fixed this
     session and earlier ones (the `myTime/2`-then-`/4` safety-cap
     defeat, lichess-bot's first-move `movetime` book-bypass, the
     instability-stretch overspend) only manifests as a real clock runs
     down over dozens of moves -- **none of them can ever trigger in an
     8-second game**, so this anchor structurally cannot confirm whether
     those fixes are doing anything in the regime they were built for.
2. **This ~2900 number disagrees with the live lichess bot's own measured
   ~2600 rating, and the gap is not yet explained.** Candidate causes,
   none yet tested:
   - TC mismatch: the lichess bot plays real `60+1`/`60+2`/`180+1`/`180+2`
     (`lichess-bot/config.yml`'s `challenge_initial_time`/
     `challenge_increment`), not `8+0.08` -- a completely different
     regime, and the one that actually exercises the low-clock code paths
     above.
   - **`UCI_Elo=X` and lichess's own Glicko-2 rating are not the same
     scale even at matched TC** -- one is Stockfish's internal
     self-declared handicap curve against a single fixed opponent; the
     other is earned against a live, differently-calibrated pool of
     humans/bots, and may itself carry a high rating deviation if the bot
     account hasn't played many rated games yet (not checked this
     session -- worth confirming game count/RD on the bot's lichess
     profile before treating 2600 as more final than 2900).
   - Real lichess games pay real network round-trip latency per move that
     a local `fastchess` match never pays at all (`lichess-bot`'s
     `move_overhead` setting exists specifically to compensate for this)
     -- one more clock cost the `8+0.08` anchor cannot see.

**Recommended next step, not yet run:** repeat this same `UCI_Elo`
bisection at the lichess bot's actual real time controls (`60+1`/`60+2`
bullet, and separately `180+1`/`180+2` blitz), same real-deployment
engine config vs. Stockfish Threads=1. If the estimate drops toward ~2600
at matched TC, that points to TC/under-limiting as the main driver of the
gap. If it stays near ~2900 even at matched TC, that points instead to
the lichess-rating-pool/network-latency/RD side, and the bot's actual
game count and rating deviation should be checked before trusting 2600 as
a stable number either. Real `60+1` games will run far longer per game
than `8+0.08` (a 50-game batch could take 30-60+ minutes rather than the
~11 minutes the 2900 rung took) -- budget for that before launching it.

**Bottom line: ~2900 is this project's best current estimate on the
`tc=8+0.08` anchor specifically, not a confirmed absolute rating.** Both
the fast-TC caveat and the unexplained gap against the live ~2600 lichess
number should travel with this figure any time it's cited -- don't repeat
it as a settled "the engine is 2900" claim without those two caveats
attached.
