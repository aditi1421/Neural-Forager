# Neural Forager

A Nengo project studying learning and memory in changing mazes. Runs locally on
CPU with real LIF neurons and PES plasticity. The agent needs **no LLM or API key**.

Two experiments live here:

- **Interactive foraging:** a spiking value learner explores, finds food, and
  adapts to moved food and blocked passages. A browser dashboard shows the maze,
  spikes, action values, and actual learning traces.
- **Selective memory updating:** a controlled experiment asks whether diagnosing
  position errors before writing observations protects valid neural memories.
  Matched controls separate update gating, localization correction, and neural
  versus conventional memory.

Read the [spiking-advantage audit](environments/neural_forager/audit/REPORT.md),
[algorithmic-novelty review](environments/neural_forager/audit/NOVELTY.md),
[original selective-memory results](environments/neural_forager/research/REPORT.md),
[frozen protocol](environments/neural_forager/research/PROTOCOL.md), and
[prior-art review](environments/neural_forager/research/LITERATURE.md).

## Start the dashboard

Python 3.10+ and the [Prime CLI](https://docs.primeintellect.ai/) are required
for the workspace workflow. The tested interpreter was Python 3.10.

```bash
git clone https://github.com/aditi1421/Neural-Forager.git
cd Neural-Forager
prime lab setup --agent codex
prime env install neural-forager
.venv/bin/neural-forager serve
```

Open **http://127.0.0.1:8765**. Choose a controller and seed, create an experiment,
then run, pause, step, move food, or close a passage. Export the observed run
as JSON. The dashboard displays the original value-learning experiment; the
selective-memory experiment runs through the research harness below.

## Reproduce the research

```bash
prime eval run configs/eval/neural-forager-research-smoke.toml --disable-tui
prime eval run configs/eval/neural-forager-research.toml --disable-tui
```

The full evaluation runs 24 Prime tasks containing **480 episodes**: 12 held-out
mazes × 2 neural seeds × 5 scenarios × 4 methods. Allow several minutes on a CPU.
The harness is a local Python program and makes no model calls, even if Prime
prints its default model name. Both configs explicitly retain episode traces.
Prime 0.6.16 skips automatic upload for config-driven runs; no upload opt-out is
used. The environment has not been published to Prime Hub.

To inspect the committed evidence without running simulations:

```bash
.venv/bin/python environments/neural_forager/research/analyze.py \
  environments/neural_forager/research/heldout.json.gz --output /tmp/forager-analysis
```

For a fresh run, give that script the `results.jsonl` path printed by Prime.
It checks that all 480 episodes are present and computes paired confidence
intervals over independent maze seeds, rather than treating moves as samples.

## Does spiking help?

The follow-up audit replaces spikes with matched continuous firing rates, then
transfers the same weights in both directions. It tests current noise, neuron
loss, short reads, and reduced neuron counts on 16 fresh mazes: 1,024 memory
assays and 384 navigation episodes. See the [audit report](environments/neural_forager/audit/REPORT.md)
for the measured outcome and claim boundaries.

**Current finding:** no navigation advantage from spikes. A noise-related
readout improvement over instantaneous rate units was surpassed by a simple
non-spiking filter in 256 additional tests on fresh mazes. No defensible new
algorithmic principle is established. The report includes the unfavorable
results and the filter's own tradeoffs.

```bash
prime eval run configs/eval/neural-forager-audit-smoke.toml --disable-tui
prime eval run configs/eval/neural-forager-audit.toml --disable-tui

# Reproduce analysis from the committed full evidence:
.venv/bin/python environments/neural_forager/audit/analyze.py \
  environments/neural_forager/audit/prime-heldout.jsonl.gz --output /tmp/forager-audit
```

## Tests

```bash
uv pip install --python .venv/bin/python -e 'environments/neural_forager[dev]'
.venv/bin/python -m pytest environments/neural_forager/tests -q
```

53 tests cover the world, real plastic weights, observations, localization,
memory gates, reward edits, dashboard sessions, matched rate controls, weight
transfer, shared perturbations, cache invalidation, smoothing controls, and paired
statistical analysis. See the
[implementation guide](environments/neural_forager/README.md) for architecture
and the original foraging pilot.

## Scope

This is a research prototype. In the selective-memory experiment, storage and
plastic updates run in a 972-neuron Nengo network; diagnosis and graph planning
are conventional Python. A local survey initializes memory before disruptions.
The project does not establish a new learning principle, an all-spiking agent,
or an advantage of spiking over conventional memory. The report makes those
boundaries explicit.
