# Selective memory updating: frozen evaluation protocol

Written before running held-out seeds. Development seeds: 7, 11, 19. Test maze
seeds: 201–212, crossed with neural seeds 31 and 47. Do not tune on test outcomes.

Fixed parameters: 120 moves per episode, 9×9 maze, four aliased landmark
categories, 20% landmark dropout during evaluation, four-observation history,
0.75 confidence threshold. There are 480 episodes: 12 mazes × 2 neural seeds ×
5 scenarios × 4 methods. Development used only seeds 7, 11, and 19.

## Question and hypothesis

Does diagnosing observation mismatch before committing plastic updates reduce
damage to valid spatial and reward memories, compared with writing all sensory
observations at the current position estimate?

Primary outcome: fraction of previously learned, unchanged memory components
whose decoded categorical content becomes incorrect after a disruption.
Secondary outcomes: first-reward recovery within a fixed move budget, capped
recovery latency, position accuracy, diagnosis accuracy, false alarms in stable
worlds, and wall-clock simulation time. Report failures and null results.

## Controlled setup

The experiment first gives every method the same local sensory survey. A DFS
walk uses only adjacent-wall observations and actual traversed displacements;
the controller never receives the world's occupancy array. This deliberately
isolates *retaining and updating learned knowledge* from initial exploration.
The survey includes observations of the original reward and aliased landmarks.
It is a controlled assay, not a demonstration of autonomous map discovery.

Following the survey, each method receives an identical fresh copy of learned
memory and starts at home. It receives integrated odometry, four local movement
affordances, an aliased landmark category (sometimes absent), and local reward
presence. No true pose, intervention type, event flag, or food coordinate is
passed to an agent. True state is evaluator-only. Heading and displacement are
otherwise exact; the localization intervention adds a persistent odometry bias.

Separate episodes: unchanged world, reward relocated, passage closed,
odometry bias, and simultaneous passage closure plus odometry bias. The latter
is a stress test outside the single-cause hypothesis family. Missing landmarks
are deterministic by maze, time, and physical location, independent of method.

## Mechanism and controls

The neural memory stores movement affordances, reward presence, and landmark
identity in separate PES-plastic Nengo connections driven by LIF place cells.
All retrievals and updates run through the simulator. Previously retrieved
decoded values are cached for conventional planning and diagnosis; this cache
is refreshed from neural output, not from supervised targets.

A conventional likelihood calculation compares local map-change hypotheses
against constant localization-offset hypotheses across recent observations.
Selective plasticity delays writes during uncertainty, applies position
corrections before writes, and updates only inconsistent memory channels.
This inference and the graph planner are Python code, not neural circuits.

Controls: neural memories with unconditional writes but identical diagnosis;
neural memories without localization correction; and a conventional tabular
memory using the same selective inference and planner. The comparison isolates
gating more closely than comparison with a random walker or a different planner.

## Analysis

Pair methods by maze/neuron seed/scenario. Average neural-seed replicates within
each maze before bootstrap confidence intervals; resample mazes, not individual
moves or deterministic reruns. Primary paired contrast: selective neural minus
unconditional neural memory damage. Lower is better. Include an interval for
the recovery-rate difference and latency, so suppressing updates cannot earn a
success claim merely by freezing everything.

The primary contrast equally averages the four disrupted scenarios, then the
two neural-seed replicates, within each maze. Stable worlds are a separate
negative control. Retain failed intervention attempts in the assigned-scenario
analysis and disclose their counts; also report an applied-intervention
sensitivity analysis if any occur. Use 10,000 paired cluster-bootstrap samples
with RNG seed 20261001 and percentile 95% intervals. These are exploratory
intervals, with no multiple-comparison correction. A manifest freezes source
hashes and this protocol before the held-out run.

For attribution, distinguish first non-stable diagnosis from the final belief;
record abstentions rather than forcing a cause label. Compound cases cannot be
scored as a single-cause classification success. Evaluate unchanged-memory
damage only where the calibration memory was initially correct and ground truth
did not change, excluding legitimate edits from the denominator.

No claim of novelty, biological realism, energy efficiency, or general
superiority follows from this test alone. The literature review must constrain
any contribution claim even if the hypothesis is supported.
