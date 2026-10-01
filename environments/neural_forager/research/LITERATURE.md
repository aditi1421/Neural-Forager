# Focused prior-art review

Scope: selective updating of spatial, reward, and localization knowledge in a
Nengo navigation agent. This is a focused search, not a systematic review or
proof that a mechanism has never been published. Sources below are research
papers or model-author documentation. Reviewed 2026-10-01.

| Prior work | Already established | Consequence for our claim |
|---|---|---|
| Barreto et al., **Successor Features for Transfer in Reinforcement Learning** (NeurIPS 2017), [paper](https://arxiv.org/abs/1606.05312) | Separates reward from environmental dynamics to transfer between tasks. | Separating reward and spatial knowledge is not novel. |
| Kreiser et al., **Error Estimation and Correction in a Spiking Neural Network for Map Formation in Neuromorphic Hardware** (ICRA 2020), [author-hosted paper](https://sandamirskaya.eu/resources/ICRA_2020-7.pdf) | Spiking mechanisms detect errors on landmark revisits and correct map/path-integration estimates. | Spiking localization-error correction is not novel. |
| Dumont et al., **Exploiting semantic information in a spiking neural SLAM system** (2023), [model documentation](https://zoo.nengo.ai/submissions/ssp-slam/index.html) | Combines spiking path integration, landmark correction, and learned associative spatial memory. | Combining those neural modules is not itself a contribution. |
| Qian et al., **POCD: Probabilistic Object-Level Change Detection and Volumetric Mapping in Semi-Static Scenes** (RSS 2022), [paper](https://arxiv.org/abs/2205.01202) | Combines geometry and semantics in probabilistic map maintenance under environmental changes. | Evidence-dependent map updates have substantial robotics precedent. |
| Danieli & Lepperød, **Flexible navigation with neuromodulated cognitive maps** (PLOS Computational Biology, 2026), [paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013487) | Uses reward/boundary modulation in online cognitive maps and tests altered rewards and boundaries. The paper describes a rate-network model. | Especially close prior art: neuromodulated adaptive cognitive maps and these interventions are already studied. |

## What this implementation actually contributes

A reproducible, small controlled assay of a particular update policy: compare
stable-world, local-edge-change, and constant-pose-offset explanations of recent
aliased observations; defer plastic writes when evidence is ambiguous; apply
pose correction before writing; then update only inconsistent memory channels.
The paired control has the same memories, inference, and planner but writes
unconditionally. A conventional memory control tests whether any advantage
requires spiking dynamics at all.

The diagnostic scores and graph search are ordinary Python. Only associative
memory storage, retrieval, and PES plasticity run in a Nengo LIF network. This
is not an all-spiking cause-inference system. The diagnostic hypothesis family,
costs, confidence threshold, and separate output channels are designed by us.

## Novelty verdict before testing

**No established algorithmic novelty.** The defensible near-term contribution is
an implementation and causal comparison for this constrained task. A positive
result would support the value of this gate relative to its matched control;
it would not establish a new principle, a neuromorphic advantage, or superiority
to the cited systems. We do not reproduce those papers' full architectures, so
our baselines must not be presented as direct comparisons against them.

A stronger future claim would require a new learned or neural diagnostic
mechanism, more realistic noisy localization, direct comparisons to suitable
prior methods, and broader held-out evaluation. Merely moving the hand-coded
classifier into a neural implementation would not by itself establish novelty.
