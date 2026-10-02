# Does Neural Forager benefit from spikes?

This follow-up audits the existing project. It separates three claims:
task performance, resource efficiency, and algorithmic novelty. A positive result
on one does not automatically establish either of the others.

## What changed in the comparison

The first study compared neural and conventional memories but did not isolate
spiking. This audit adds Nengo `LIFRate`, with the same place inputs, neuron
counts, dense weight dimensions, PES learning rule/rate, training observations,
output filters, and readout schedule as the spiking `LIF` network.

Each backend learns from the same local survey. We then also transfer the
learned weight arrays exactly in both directions. Frozen-memory assays test
clean reads, fluctuating current, 50% neuron loss, and a 20 ms read deadline,
at one or twelve neurons per cell. Both receivers get identical fault masks
and input-noise streams for the same simulation schedule.

The behavioral check runs the existing selective controller under clean,
noisy-current, and neuron-loss conditions, with drift and combined map/pose
changes. It refreshes the entire learned cache under the perturbation before
navigation, so a clean cached map cannot hide a damaged neural representation.
The same refresh is applied to both backends.

| Design | Value |
|---|---|
| Independent maze seeds | 301–316 (16 new mazes) |
| Neural seeds | 31 and 47 |
| Prime tasks | 32 |
| Memory assays | 1,024; includes both weight-transfer directions |
| Navigation episodes | 384, each capped at 120 moves |
| Primary endpoint | Own-trained spiking minus rate recall accuracy, 12 neurons/cell, pooled over four conditions |
| Meaningful primary benefit | At least +1 percentage point, with paired 95% interval above zero |
| Condition search | Eight own-trained accuracy comparisons, exact paired sign flips with Holm adjustment |
| Uncertainty | 20,000 bootstrap resamples of 16 maze means |

The neuron-seed replicates, cells, and moves are not independent statistical
units. Secondary analog-error and weight-transfer comparisons are descriptive.
All calibration outcomes are retained; none is filtered out for being poor.

## Performance results

All 32 Prime tasks completed with zero execution errors; all 1,024 memory
assays and 384 navigation episodes are present. All interventions applied and
all calibration categorical accuracies were 100%.

**No categorical-recall or navigation advantage was established.** The primary
spiking-minus-rate accuracy difference was **−0.0112 percentage points**, with
a 95% interval of **[−0.0223, 0.0000]**. It fails the predeclared +1 percentage
point benefit criterion. None of the eight condition comparisons favored
spikes after the declared adjustment (all one-sided Holm p-values were 1).

| Condition | Neurons/cell | Spiking accuracy | Rate accuracy |
|---|---:|---:|---:|
| Clean | 1 or 12 | 100% | 100% |
| Current noise | 1 or 12 | 100% | 100% |
| 50% neuron loss | 1 | 80.268% | 80.268% |
| 50% neuron loss | 12 | 99.911% | 99.911% |
| 20 ms read | 1 | 97.396% | 100% |
| 20 ms read | 12 | 99.955% | 100% |

Redundancy helped tolerate neuron loss, but the rate circuit received the same
benefit. Short windows hurt the spiking readout, especially with one neuron per
cell. Holding the learned weight arrays identical did not reveal a categorical
spiking advantage in either transfer direction.

**Both backends reached food in 192/192 navigation episodes.** Recovery latency
matched in every paired episode: 18.69 moves on average in clean/noisy-current
conditions and 22.00 with neuron loss. Equal outcomes in this finite sample do
not prove equivalence in all environments.

There is a **narrow analog-accuracy result** worth investigating. At twelve
neurons/cell under noisy current, spiking MSE was approximately **0.00060**
versus **0.00244** for instantaneous rate units, about 76% lower. The paired MSE
difference was −0.00185 [−0.00190, −0.00180]. These are secondary unadjusted
intervals. This improvement did not change categorical accuracy or navigation.
It cannot yet be attributed specifically to discrete spikes, because the rate
control also lacks LIF membrane integration. A separately frozen dynamic-rate
follow-up tests that explanation; it must not be confused with the original
primary endpoint or a post-hoc change to this evaluation.

See [all tables](TABLES.md) and [machine-readable results](summary.json), including
analog error, channel accuracy, reward detection, and both transfer directions.

## Resource claim

The implemented 12-neuron/cell circuit allocates 8,748 float64 plastic weights
(69,984 bytes) **plus** the same 729-value decoded cache (5,832 bytes) that a
conventional table uses. Those two arrays alone total **13 times** the table's
value storage. Both spiking and rate circuits have this same allocation.
At one neuron/cell the two arrays together are twice the table's value storage.

These are directly counted persistent arrays, not total process-memory
measurements. They omit neural state, currents, filters, learning-rule state,
simulation bookkeeping, and Python objects. They therefore cannot support a
memory-efficiency advantage for this implementation. This does not exclude
different savings from another representation or specialized hardware.

Wall times are instrumented CPU-simulator times. The activity monitor counts
LIF spikes but reports a separate rate integral for LIFRate. Neither measurement
is hardware energy. We did not measure power or run on Loihi/SpiNNaker, so no
energy or deployment-latency advantage is established.

## Algorithmic novelty

The [mechanism and prior-art audit](NOVELTY.md) identifies no defensible established
algorithmic novelty in the current implementation. PES associative learning,
separate reward/spatial knowledge, localization correction, adaptive maps,
and matched spike/rate weight-transfer experiments all have precedents.

The custom diagnostic function is an engineered finite-hypothesis scorer; it is
not a newly learned neural inference algorithm. The exact assembly may differ
from published systems, but an implementation difference is insufficient evidence
for a substantive new algorithm. This conclusion is independent of whether one
of the memory conditions favors spikes.

## Boundaries of the conclusion

This is a small, discrete, fully surveyed map. Neural inputs are place-current
cues rather than asynchronous sensory event streams. Heading and movement
increments are otherwise exact. There is no recurrence or decision rule that
directly reads relative spike timing. The chosen learning rate and filters are
shared; neither backend received a separate hyperparameter search.

`LIFRate` removes membrane/refractory state as well as discrete events. If LIF
does better under a perturbation, that comparison alone cannot separate a benefit
of membrane integration from a benefit of spikes. A dynamic non-spiking control
would be required for that stronger attribution.

Mean categorical accuracy can obscure loss of the one rewarded location or
particular landmark/geometry components. Per-channel accuracy, reward-site
detection, analog MSE, full decoded arrays, and navigation traces are retained.
Short reads include transient mixing from sequential cues; they are not an
optimized temporal code. Neuron-loss comparisons concern redundant neural
representations, not digital memory bit failures.

No result here proves that spikes can never help elsewhere. The conclusions
apply to this implementation, these perturbations, and the specified metrics.
Prior work reports spiking benefits using different temporal encodings and
training methods, as discussed in the novelty audit.

## Reproduce and inspect

From the repository root after installation:

```bash
prime eval run configs/eval/neural-forager-audit-smoke.toml --disable-tui
prime eval run configs/eval/neural-forager-audit.toml --disable-tui
.venv/bin/python environments/neural_forager/audit/analyze.py \
  PATH_PRINTED_BY_PRIME/results.jsonl --output /tmp/forager-audit
```

Reproduce the analysis alone using the committed
`environments/neural_forager/audit/prime-heldout.jsonl.gz` as the input.
The script rejects missing/duplicate episodes, incomplete tasks, and errors.
[PROTOCOL.md](PROTOCOL.md) and [manifest.json](manifest.json) record the local
pre-run freeze. This is an audit trail, not third-party preregistration.

The harness makes no LLM calls. Prime's default model label is unused. Configs
explicitly save full state, with no upload opt-out; this installed Prime version
does not automatically upload config-driven evaluations. All evidence is kept
in this repository. The first study remains available at commit `d5845a0`.
