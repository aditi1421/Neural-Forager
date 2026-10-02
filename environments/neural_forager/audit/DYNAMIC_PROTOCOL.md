# Targeted follow-up: can non-spiking current smoothing explain the noise result?

The main audit at commit `f7e1e56` found no categorical/navigation advantage,
but LIF had lower analog MSE than instantaneous LIFRate under noisy current.
This follow-up was selected **after seeing that result**. It uses fresh mazes
and its own source freeze; it does not retroactively change the main endpoint.

Add first-order exponential smoothing of input current before LIFRate's
nonlinearity. Fixed time constants: **8 ms** (the existing output-filter time
constant) and **20 ms** (the LIF membrane time constant). Neither value is tuned
on the new mazes. Retain the original 65 ms cue and last-25-ms average, with no
extra warmup or survivor rescaling. Smoothing state resets at the start of an
assay and evolves normally through sequential cell cues, including transients.

Train the original spiking and instantaneous-rate memories independently from
zero on the same local survey. Transfer each weight set exactly to four
receivers: LIF, instantaneous LIFRate, 8 ms smoothed LIFRate, and 20 ms smoothed
LIFRate. The filtered receivers receive no extra training or learning-rate
tuning. All use 12 neurons per cell and the existing 8 ms output filter.

Evaluate clean and multiplicative-current-noise conditions on maze seeds
401–408 crossed with neural seeds 31 and 47: **16 tasks / 256 full-map assays**.
Development/smoke seed 7 remains separate. Preserve all decoded arrays.

Primary follow-up comparisons: noisy-current MSE, **spiking minus each filtered
rate control**, equally averaging both weight sources and neural seeds within
each maze. Positive differences favor the non-spiking control. Use paired
20,000-replicate percentile 95% bootstrap intervals (RNG 20261002) over the eight
mazes, exact one-sided sign-flip tests, and Holm correction across the two
filter comparisons. A superiority claim requires an interval above zero and
adjusted p < 0.05. Null findings do not prove equivalence or unique spiking benefit.

Report clean errors, categorical accuracy, each training source, and both filter
settings even when unfavorable. This is a targeted mechanism control, not a
new navigation benchmark or an optimization of the best possible rate model.
A filter that recovers the noise benefit would show that the benefit is not
exclusive to discrete spike events in this task. It would not prove exact
equivalence of a linear smoother and LIF membrane dynamics.
