# Spiking advantage and novelty audit: protocol

Written before evaluation on mazes 301–316. Development/smoke uses seed 7.
This audits the existing mechanism; it does not assume a positive result and
does not introduce a new learning algorithm. The original study is unchanged.

## Matched intervention

Replace Nengo `LIF` with `LIFRate` while retaining direct place currents,
neuron count, random drives, output dimensions, dense PES weight count,
initial weights, gain/bias, 8 ms output/presynaptic filters, 1 ms time steps,
PES learning rate 0.25, local survey, and training duration. Readouts are means
of the last 25 simulation samples. Rate output goes through the same filters.

Train each backend separately. Then cross both learned weight sets into both
receivers with exact array copies. This separates the effect of spike-based
learning from spike-based retrieval. Both source weight sets are reported;
do not select whichever transfer direction favors a preferred conclusion.

Budgets: 1 and 12 neurons per cell (81 and 972 total). All methods use the same
81-cell dictionary; this changes representation redundancy, not map size.
The Nengo weights are dense float64 arrays. Report weights and cache bytes
separately; these are persistent-array counts, not total resident memory.

## Conditions

1. Clean: original 65 ms cue, mean of its last 25 ms.
2. Current noise: every simulation step multiplies each cued neuron's current
   by `1 + Normal(0, 0.5)`. Inactive currents stay zero. Training is clean.
3. Neuron loss: uniformly select exactly floor(N/2) input units and suppress
   their drive after training. No survivor rescaling or retraining in recall
   assays. Masks are identical for matched spiking/rate receivers.
4. Short read: 20 ms per cue, average all 20 samples, no reuse of samples from
   the previous cue. Filters/voltage retain ordinary sequential-read state.
   Both receivers use the same schedule. This tests a latency constraint,
   not a carefully optimized temporal code.

Conditions are engineering perturbations, not a full sensor/hardware model.
Noise RNG and lesion masks depend on maze, neural seed, budget, condition, but
never backend or trained-weight source. Each assay restores weights and neural
state. Reads cannot change weights. Shared noise realizations are guaranteed
for identical step schedules; online navigation can diverge in update counts
and consequently in later positions along the noise stream.

## Sample size and endpoints

16 fresh maze seeds (301–316) × neural seeds 31 and 47 = 32 Prime tasks.
Each task contains 2 budgets × 2 training sources × 2 receivers × 4 conditions
= 32 memory assays, for **1,024 assays** total. Every assay reads all locally
surveyed cells. Calibration errors are retained, not discarded.

Primary contrast: spiking minus rate categorical read accuracy, each using its
own trained weights, at budget 12, equally averaged over the four conditions
and then over neural seeds within each maze. A meaningful primary benefit
requires an improvement of at least **1 percentage point** and a paired 95%
maze-bootstrap interval excluding zero in the positive direction.

Secondary: per-channel accuracy, analog MSE, reward-site detection, calibration
accuracy, resource counts, timings, both budgets, and identical-weight transfer.
Categorical accuracy alone can hide reward failure because most cells contain
no reward. Always include reward-site detection and navigation.

For searching a condition-specific accuracy advantage across the 2 budgets ×
4 conditions, use one-sided exact paired sign-flip tests of the 16 maze means
with Holm adjustment over all eight comparisons. Report effect sizes and raw
intervals too. These tests assume exchangeable paired differences under the
null; they do not establish general SNN superiority. Transfer comparisons are
mechanistic/descriptive, with no additional confirmatory claims.

## Behavioral check

At budget 12, run the same selective controller and planner using independently
trained spiking/rate memories: 3 conditions (clean/noise/lesion) × 2 scenarios
(drift/combined) × 2 receivers × 32 tasks = **384 navigation episodes**.
120-move horizon. Plasticity stays active as in the original controller.
After restoration, re-read every known location under the perturbation so
the clean Python cache cannot hide damaged neural memories. Both backends
get the same refresh procedure; that is additional setup work for this assay.

Report paired food-recovery rate and capped latency. Do not call a small
readout gain a useful navigation advantage if it does not improve behavior.
Include all assigned interventions and disclose unsuccessful injections.

## Statistical and claim boundaries

Resample the **16 mazes**, not cells, moves, or neuron-seed replicates. Use
20,000 bootstrap replicates, RNG seed 20261002, percentile 95% intervals.
The conventional table's original task performance and current clean survey
accuracy/storage are reference points. Neuron loss is not applied to a table:
there is no justified mapping from neuron failure to digital memory errors.

Runtime is CPU simulation wall time, not hardware energy. Spike counts do not
measure joules. No energy or neuromorphic-hardware claim will be made without
measurements on such hardware. Count rate integrals separately from spikes.

Novelty is assessed through a source-level mechanism decomposition and a
targeted primary-literature review, independently of performance. A favorable
benchmark would not make established PES, rate coding, residual gates, or
finite-hypothesis model selection new. An unsuccessful literature search is
not proof of novelty. Record the strongest defensible project-specific verdict.

No tuning after held-out outcomes. Save full decoded arrays, navigation traces,
canonical Prime records, source hashes, and analysis. A local source freeze is
an audit trail, not formal third-party preregistration.
