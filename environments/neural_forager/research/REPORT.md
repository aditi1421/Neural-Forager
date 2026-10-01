# Selective updating of neural spatial memory

## What was built

A controlled Nengo experiment separates three questions: did the world change,
did a reward move, or is the current position estimate wrong? An observation
contains integrated odometry, local movement affordances, a sometimes-missing
aliased landmark category, and local reward presence. It contains no true pose,
food coordinates, or intervention labels.

Geometry, reward, and landmark memories occupy three independently gated PES
connections from 972 LIF neurons. Every read and plastic update runs through
Nengo. A conventional diagnostic heuristic compares recent observations with
decoded memory, corrects estimated position when appropriate, and defers writes
when uncertain. A conventional graph planner uses the learned map. This is a
hybrid system: diagnosis and planning are not spiking circuits.

Each method starts from the same local sensory survey. This isolates retention
and updating of established memories; it does not test initial exploration,
continuous SLAM, or autonomous discovery of the diagnostic hypotheses.

## Controls and evaluation

| Item | Setting |
|---|---|
| Development mazes | 7, 11, 19 |
| Held-out mazes | 201–212 |
| Neural seeds | 31, 47, crossed with every maze |
| Scenarios | Stable, moved food, blocked route, odometry bias, blocked route plus bias |
| Methods | Selective neural, ungated neural, neural without localization repair, selective tabular |
| Budget | 120 moves to first food; failures retain the full capped latency |
| Sensors | Four aliased landmark categories, 20% evaluation-time landmark dropout |
| Size | 9×9 mazes, 480 episodes grouped into 24 Prime tasks |
| Independent units | 12 mazes; neural replicates averaged within each maze |

The primary measure is damage to initially correct, still-valid memory
components. Legitimate environmental edits are excluded from its denominator.
The main contrast equally averages the four disrupted scenarios and compares
selective versus ungated neural memory. Recovery and latency prevent interpreting
frozen, nonfunctional memory as a success. Stable scenarios test false alarms.

The ungated control retains the same diagnostic algorithm and planner. It writes
all available channels at its current position estimate, including during
uncertainty. Thus this comparison tests the **combined selective-update policy**:
deferred writes plus channel-specific residual gates. It does not isolate the
contribution of each gate individually. Timing and subsequent trajectories can
also diverge between policies.

The tabular control shares diagnosis and planning but stores values conventionally.
Its update gain is 0.7, whereas neural memory uses PES dynamics. This is a useful
non-neural mechanism control, not a matched-compute or matched-learning-rate test.

## Results

All **24 Prime tasks / 480 episodes completed with zero execution errors**.
All interventions applied, and calibration categorical accuracy was 100% for
every method and maze. There are 96 disrupted episodes per method.

| Method | Reached food | Unchanged-memory damage | Mean capped moves |
|---|---:|---:|---:|
| Selective neural | 96/96 | 0.000% | 20.81 |
| Ungated neural | 96/96 | 0.150% | 18.94 |
| No localization repair | 50/96 | 4.056% | 68.29 |
| Selective tabular | 96/96 | 0.000% | 20.81 |

**The update gate did not improve recovery success.** It prevented a small amount
of final memory corruption observed in the ungated control: 10 of 96 episodes,
concentrated in only three of the 12 mazes (204, 206, 207). The primary paired
damage difference was **−0.150 percentage points**, with a 95% maze-bootstrap
interval of **[−0.329, 0.000]**. The interval reaches zero; this small experiment
does not give strong evidence for a broadly reliable memory-retention advantage.

The gate also cost **1.875 additional moves** on average, interval
**[0.708, 3.167]**. In route-only changes it took 22.92 moves versus 18.92 for
ungated memory. Pure drift recovery was slightly faster (14.25 versus 14.92).
This is a retention/adaptation tradeoff, not a general navigation improvement.

Removing localization repair had a much larger effect: 50/96 recoveries versus
96/96, a paired difference of 47.92 percentage points [43.75, 50.00]. It failed
22/24 drift episodes and all 24 combined episodes. The two successful drift
episodes were the two neural-seed runs of maze 201: food was reached after 27
moves while the position estimate remained wrong throughout. Reward success
alone would miss that localization failure.

**The tabular control matched selective neural memory on recovery, categorical
damage, and latency in every paired episode.** No spiking advantage is established.
Neural-seed replicates had the same categorical outcomes here; they are not
additional independent mazes. Zero-width intervals for equal observed recovery
do not prove equivalence outside this sample.

Stable worlds produced no change alarms, no damage, and 24/24 successes per
method. Selective memory's first diagnosis identified reward relocation in
24/24 reward episodes and drift in 24/24 drift episodes. It diagnosed a route
change in 20/24 route episodes; the remaining four completed without such a
diagnosis. Change detection and navigation recovery are different outcomes.
In combined maze 207, it first reported a route change at move 3 and repaired
position only at move 7; pose accuracy was 70.8%. Compound cases are not scored
as a single-cause classification problem.

Mean measured episode time was 0.261 s selective, 0.534 s ungated, 1.473 s without
localization repair, and 0.008 s tabular. These times include final evaluator
readbacks but exclude network construction and calibration. Fewer gated writes
mean fewer simulator steps, so this is not a hardware or energy-efficiency test.

See [all scenario tables and intervals](TABLES.md), the
[machine-readable summary](summary.json), and the compressed full traces.
**Recommendation:** retain this as a reproducible negative/mixed-result study.
A stronger next experiment needs realistic odometry errors, longer sequences of
changes, and a learned diagnostic mechanism; merely increasing the maze count
would not create algorithmic novelty.

## Novelty and limitations

The [prior-art review](LITERATURE.md) finds substantial precedent for separating
reward and spatial knowledge, spiking localization correction, and adaptive
cognitive maps. **No algorithmic novelty is established.** The contribution is a
reproducible implementation and a matched comparison of this particular policy.
Positive results alone cannot establish a new learning principle or a benefit
from spikes.

The hypothesis family and costs are hand-designed, and softmax confidence is an
uncalibrated heuristic. The drift intervention is a constant integer translation
within the diagnostic search range; heading and actual movement increments are
exact. Geometry sensing is noiseless. Evaluation rewards recovery of the first
food location, not sustained foraging or map consistency over repeated changes.
The combined disruption is outside the single-cause model family but still uses
the same simple component interventions. Twelve small mazes cannot establish
robustness in larger or continuous environments.

Memory damage is a final categorical readback metric. It can miss transient
corruption, analog drift that does not cross a threshold, and errors in newly
learned components. Reads of the full original survey map occur only in the
evaluator after the episode. The planner uses a cache of neural readbacks;
supervised targets are not copied into that cache.

## Reproduce and audit

From the repository root after installation:

```bash
prime eval run configs/eval/neural-forager-research.toml --disable-tui
.venv/bin/python environments/neural_forager/research/analyze.py \
  PATH_PRINTED_BY_PRIME/results.jsonl --output /tmp/forager-analysis
```

To reproduce analysis without simulation, substitute the committed
`environments/neural_forager/research/heldout.json.gz` for the input path.
It contains every episode, sensor/pose trace, diagnosis, write count, and outcome.
The original canonical records are also retained as `prime-results.jsonl.gz`.
The analysis checks completeness and uses 10,000 paired maze-bootstrap samples
(seed 20261001). Intervals are exploratory percentile 95% intervals without
multiple-comparison adjustment.

[PROTOCOL.md](PROTOCOL.md) and [manifest.json](manifest.json) record the local
pre-run protocol and source hashes. This is a local protocol freeze, not a
third-party preregistration. Prime 0.6.16 completed the first held-out run but did
not persist traces for the config-driven command. We repeated the same experiment
with `save_results = true`; the manifest records this logging-only amendment.
No algorithm, seed, or outcome definition was changed after held-out testing.
The duplicate execution is not additional statistical evidence.

Prime's default model label is unused by this local program; there are no LLM
calls. This CLI version reports that automatic upload is skipped for config-driven
runs. No upload opt-out was requested, and the environment is not on Prime Hub.
Exact Nengo, NumPy, SciPy, Pydantic, and Verifiers versions are in the manifest.

Validation: **42 tests passed**, including real Nengo weight changes and exact
protection of non-gated connections. Dashboard JavaScript syntax checks passed.
The copied repository matched the frozen implementation hashes. A separate
known-difference check verified the bootstrap's maze grouping, and rerunning the
analysis from the committed compressed traces reproduced the summary exactly.
