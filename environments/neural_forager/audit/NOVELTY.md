# Algorithmic novelty audit

Reviewed 2026-10-02. Scope: the algorithm implemented in Neural Forager, not
whether spiking neural networks can ever offer an advantage. This is a targeted
primary-source review and code analysis, not an exhaustive novelty search.

## Mechanism-level finding

The current code has **no defensible established algorithmic-novelty claim**.
It implements a particular combination of established neural associative
learning, conventional hypothesis scoring, residual-based update gates, and
graph search. Its reproducible benchmark and controls are useful engineering
contributions. They should be described as such.

The exact combination of constants and hypotheses may be uncommon. Showing
that its ingredients have precedents does not logically prove that every
combination has appeared before. However, this project has not identified a
new inference principle, learning rule, theorem, learned diagnostic mechanism,
or experimentally established capability that supports a substantive new
algorithm claim. Lack of a search hit would not establish one either.

## What the code computes

1. An estimated integer coordinate selects a dedicated group of place units.
   This index is designed, not learned. Different locations do not share a
   compressed feature code.
2. The group drives three linear output projections. Separate output channels
   encode local geometry, reward presence, and landmark identity.
3. PES updates follow the established local error-times-activity form, with a
   mask: `delta(weight[k,j]) ∝ -gate[k] * error[k] * filtered_activity[j]`.
4. A Python function scores three explanations for a four-observation history:
   the current map/pose, one toggled edge, or a constant coordinate offset.
   It applies penalties and a softmax, then abstains below a fixed threshold.
   This is a heuristic score, not a calibrated Bayesian posterior.
5. Writes are deferred during abstention. Once confident, channel residual
   thresholds gate the same PES rule. A Python BFS plans through cached neural
   readbacks. No learned neural circuit infers the cause of a mismatch.

After cue/filter transients settle, a decoded value at cell `x` has the form
`memory[x,k] ≈ sum_j(weight[k,j] * mean_activity[j])` over the units assigned to
that cell. In the long-time constant-current limit, replacing spike trains by
their firing rates produces the same type of static local association. Finite
windows, membrane dynamics, filtering, and learning noise can create measurable
differences; this approximation is not a proof of identical finite-time behavior.

The planner receives averaged values, never an event sequence or relative
spike timing. The current task has no event-camera input, temporal ordering
objective, learned recurrent state, or explicitly timing-dependent decision.
Consequently, a matched rate circuit is a necessary control before attributing
successful navigation to spikes.

## Claim-by-claim prior art

| Possible claim | Primary source | What it means for this project |
|---|---|---|
| Error-driven learning of neural associations | Davies, Stewart, Eliasmith & Furber, **Spike-based learning of transfer functions with the SpiNNaker neuromimetic simulator**, IJCNN 2013: [author publication page](https://www.compneuro.uwaterloo.ca/publications/davies2013.html) | PES-family learning already predates this implementation. We use Nengo's implementation; we did not invent the rule. |
| Separate reward and environmental knowledge | Barreto et al., **Successor Features for Transfer in Reinforcement Learning**, NeurIPS 2017: [paper](https://arxiv.org/abs/1606.05312) | Separating reward from dynamics has established precedent. Our code is not a successor-feature implementation or comparison. |
| Detect/correct localization errors with spikes | Kreiser et al., **Error Estimation and Correction in a Spiking Neural Network for Map Formation in Neuromorphic Hardware**, ICRA 2020: [author-hosted paper](https://services.ini.uzh.ch/admin/extras/doc_get.php?id=85538) | Spiking map/error correction is established. Our pose diagnosis is conventional Python, a weaker neural claim. |
| Combine path integration, landmarks, and neural spatial memory | Dumont, Furlong, Orchard & Eliasmith, **Exploiting semantic information in a spiking neural SLAM system**, 2023: [author-hosted paper](https://compneuro.uwaterloo.ca/files/publications/dumont2023slam.pdf) | SSP-SLAM already combines spiking localization and semantic/spatial memory. We do not reproduce its architecture or compare its performance. |
| Maintain maps when the environment changes | Qian et al., **POCD**, RSS 2022: [paper](https://arxiv.org/abs/2205.01202) | Probabilistic geometry/semantic change detection and map maintenance are established robotics topics. This is thematic precedent, not an assertion of the exact same gate. |
| Adaptive cognitive maps using reward/boundary signals | Danieli & Lepperød, **Flexible navigation with neuromodulated cognitive maps**, PLOS Computational Biology 2026: [paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013487) | Particularly close motivation: online spatial learning, reward changes, and detours in a rate-network model. Those capabilities alone are not new. |
| Matched spike/rate networks and weight transfer as an audit | Newton & Nicola, **Comparison of FORCE trained spiking and rate neural networks shows spiking networks learn slowly with noisy, cross-trial firing rates**, PLOS Computational Biology 2025: [paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013224) | Parameter matching and transferring learned weights between spike/rate systems are also established controls. Our audit uses PES and a different task; it is not a replication of FORCE results. |

## Positive spiking results do exist elsewhere

Ding, Yu, Liu & Huang's **Neuromorphic computing paradigms enhance robustness
through spiking neural networks** (Nature Communications 2025) reports robustness
benefits using temporal encoding, early exits, and training that captures temporal
dependencies. These mechanisms are materially different from this project's
static place cues and fixed averaged readout. That paper is evidence against a
blanket claim that spikes can never help, not evidence that Neural Forager already
implements the same benefit. [Paper](https://www.nature.com/articles/s41467-025-65197-x)

Nengo explicitly provides `LIFRate` as a non-spiking version of LIF; we also
checked the installed Nengo 4.1.0 implementation when constructing the matched
control. The rate model has an instantaneous current-to-rate nonlinearity,
whereas LIF includes membrane/refractory state. Thus any positive LIF-versus-rate
result still needs care: it may reflect membrane filtering rather than the
discrete spikes themselves. A dynamic non-spiking control would be required
to isolate that distinction. [Nengo API](https://www.nengo.ai/nengo/frontend-api.html#nengo.LIFRate)

The completed [dynamic-rate follow-up](REPORT.md#follow-up-the-noise-benefit-is-not-exclusive-to-spikes)
provides that additional control for the observed analog-noise effect. A fixed
8 ms non-spiking current smoother lowered noisy-current MSE further than LIF
on fresh mazes, while a 20 ms smoother performed worse. This demonstrates that
the measured noise benefit is not exclusive to spike events in this assay;
it is not a proof that all spiking dynamics reduce to linear filtering.

## Search and decision record

Queries included combinations of spiking/rate parameter matching, robustness,
neuron deletion, PES/delta-rule learning, selective plasticity with localization
or map changes, and the closest cognitive-map and semantic-SLAM paper titles.
Sources were checked through publisher pages, author-hosted papers, arXiv, and
official Nengo documentation. Search-engine snippets alone were not used to
claim that our exact algorithm appears in a prior paper.

A future novelty claim would need a precisely specified new mechanism and a
comparison with its closest alternatives. Merely replacing the Python classifier
with neurons, adding more neurons, or finding one favorable benchmark condition
would not by itself demonstrate algorithmic novelty. The separate performance
audit determines which narrower empirical claims are justified now.
