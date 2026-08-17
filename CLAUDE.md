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
- **The live engine is unchanged**: `nnue_engine.cpp`'s RFP coefficient is
  still `234` at HEAD (`3212cb5`); the only committed change this session
  was the `prunestats` instrumentation itself, which is purely additive
  and does not alter engine behavior when `NNUE_PRUNE_DEBUG` is unset.
  No SPRT-confirmed Elo change from this session — the Stockfish@2750
  anchor table at the top of this document is still current and
  unchanged.

### Net takeaway

Depth is confirmed to be the dominant lever (+143 Elo per time-doubling,
measured cleanly). Extending pruning to reach *higher* remaining depths is
ruled out (almost no node mass lives there). Tightening RFP's margin at
the depths that actually matter (2–4) is a plausible, cheap, defensible
next test — the depth-isolated node-count signal is real — but is not yet
an Elo win; it needs a clean, uninterrupted SPRT run to actually resolve
one way or the other before being adopted or discarded.

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
multithreading — it remains the pre-MT, single-threaded reference build, on
purpose, so it stays useful as the fixed comparison point if this SPRT is
ever resumed or re-run.

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
