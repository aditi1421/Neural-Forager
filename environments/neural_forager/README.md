# Neural Forager

A local Nengo experiment: a spiking agent learns to find food with position sensing, adjacent-wall sensing, and rewards. A live dashboard shows its behavior and neural activity. No LLM, model endpoint, GPU, or API key is needed for the agent.

This guide describes the original interactive value-learning experiment. The
new [selective-memory study](research/REPORT.md) uses a separate controller and
controlled survey protocol, with its own matched baselines. Its results should
not be confused with the pilot below.

## Run

From the existing Prime Lab workspace:

```bash
cd /path/to/Neural-Forager
prime env install neural-forager
.venv/bin/neural-forager serve
```

Open **http://127.0.0.1:8765**. The server binds only to localhost. Select a seed and controller, click **New experiment**, then **Run experiment**. Pause, step, close passages, move food, and export actual run data as JSON.

The full maze is visible to the human observer. **Observed cells** limits the display to sensed occupancy. Neither setting changes the agent's inputs. Settings take effect when you click **New experiment**.

Every browser connection has its own experiment. Refreshing creates a new network. Weights persist across trials within a run, not across refreshes. Collection or timeout automatically resets the agent to home; it does not yet learn a physical return-home journey.

## The architecture

```text
position / adjacent walls
          ↓
localized LIF place-cell populations
          ↓
PES plastic connections → four learned action values
          ↓
exploration + local wall mask + contrast normalization
          ↓
four LIF motor populations → decoded direction → world step
          ↑
reward + estimated next value → TD error → PES update
```

The default 9×9 network contains **1,552 LIF neurons**: 81 place populations of 16 neurons, plus four motor populations of 64. Wall-coordinate populations remain unused. Place populations receive direct sensory current.

**Neural memory:** place–action values live in plastic Nengo connection weights initialized to zero. The neural agent has no hidden Python Q-table or shortest-path planner. This is synaptic value memory, not a learned occupancy map or recurrent working-memory circuit.

**Conventional scaffolding:** perfect position sensing, place indexing, reward calculation, epsilon exploration, legal-action masking, utility normalization, eligibility bookkeeping, and TD teaching signals. The motor interface selects the strongest decoded legal direction.

The target is `reward + gamma * max_legal Q(next)`, with zero bootstrap at terminal transitions. Recent place–action eligibility decays by `gamma * 0.8`; entries above 0.12 are re-cued briefly and updated through Nengo PES. This is a truncated trace-assisted TD implementation, not a reproduction of a named biological learning algorithm. Neural and tabular learning have different effective update sizes; this is not a compute-matched comparison.

## Protocol

- Connected procedural mazes have loops to permit safe passage closure.
- Agents receive only `(position, legal directions)`. They detect food by entering its cell and receiving reward, never by getting its coordinates.
- Food gives +1, legal moves cost 0.01, collisions cost 0.08. Built-in controllers mask blocked moves.
- Trials finish at food or after 120 moves; the agent resets home and retains learned weights.
- At one-third of the move budget, an edge on the old shortest route closes if connectivity can be preserved. Otherwise, the log records an unapplied intervention.
- At two-thirds, food moves to another reachable location. World randomness is independent of controller behavior, giving paired interventions for each seed.
- Changes are not announced to the agent. Only the UI and evaluator inspect privileged world state.

## Controllers

| Controller | Purpose |
|---|---|
| `neural` | LIF place/motor populations and online PES value memory |
| `frozen` | Same neural architecture, plasticity disabled |
| `tabular` | Conventional TD learner with a NumPy table and eligibility traces |
| `random` | Uniform random legal actions |

Frozen means no learning throughout the run, not a lesion of trained memory. Neural and frozen simulations spend different amounts of simulated time on updates, so the control does not isolate every timing effect.

## Reproduce the comparison

```bash
cd /path/to/Neural-Forager
.venv/bin/neural-forager evaluate \
  --seeds 101 102 103 --steps 1800 \
  --output environments/neural_forager/results/heldout.json
```

Use `--static` to disable scheduled changes, `--agents neural frozen` for fewer controllers, or `--size 11` for a larger world. Results save incrementally. Seed 7 was used for development; 101–103 were fresh pilot mazes.

Food collected over 1,800 moves in the initial pilot:

| Controller | Seed 101 | Seed 102 | Seed 103 | Median |
|---|---:|---:|---:|---:|
| Neural | 26 | 34 | 165 | 34 |
| Frozen | 4 | 8 | 27 | 8 |
| Tabular | 33 | 25 | 139 | 33 |
| Random | 13 | 6 | 25 | 13 |

**Interpretation:** learning helps on these runs, but variance is substantial. The neural agent collected no food during rerouting on seed 101 and none after relocation on seed 102. Seed 103 recovered much better. Three maze seeds are a pilot, not evidence of statistical superiority. Every maze starts a fresh network trained online; this does not show transfer of learned weights or zero-shot generalization.

The JSON includes configuration, phase counts, interventions, trial outcomes, and move-level reward/error traces. The weight norm is relative to zero initialization, not the latest update magnitude. Coverage counts visited cells. Shortest-path efficiency is evaluator-only; trials spanning a change have `efficiency: null`. Final incomplete trials are excluded from completed-trial success rates, while their moves remain in phase denominators.

## Prime integration

Environment ID: **neural-forager**. The loader uses v1 `vf.Env` / `vf.Taskset` / `vf.Harness` with a local Python program. Task rows cover seeds 101–150, defaulting to 600 moves each. These are procedural instances, not a downloaded dataset.

```bash
cd /path/to/Neural-Forager
prime eval run neural-forager --disable-tui -c 1

# Integration smoke test, not an adaptation benchmark:
prime eval run neural-forager \
  --env-args '{"config":{"taskset":{"steps":60}}}' \
  -n 1 -r 1 --disable-tui
```

Override `config.taskset.steps` and `config.taskset.agent` for custom checks. Prime's default model label may appear in output, but this harness never invokes that model. Same-seed rollouts are deterministic replicates, not independent samples. Prime saves artifacts under `outputs/evals/`. The local environment is unpublished, so the CLI cannot upload results until linked upstream. No upload opt-out flag is used.

Tested Nengo and Verifiers versions are pinned because the Verifiers v1 API has changed across releases. Dependency versions are recorded in `results/versions.txt`.

## Tests

```bash
cd /path/to/Neural-Forager
uv pip install --python .venv/bin/python -e 'environments/neural_forager[dev]'
.venv/bin/python -m pytest environments/neural_forager/tests -q
```

Tests cover connectivity across 20 seeds, observation boundaries, rewards, resets, determinism, real PES weight changes, frozen weights, neural action selection, and WebSocket session isolation/export. The Nengo backend-testing pytest plugin is disabled for these application tests; they still run real Nengo simulators.

## Next research milestones

1. **Reliable adaptation:** more held-out seeds, recovery-latency metrics, reward-omission-driven exploration, trained-memory lesions, and independent world/neuron seeds.
2. **Neural spatial state:** replace exact coordinates with recurrent path integration under noisy odometry, then test drift and landmark correction.
3. **Reusable spatial memory:** learn topology or spatial semantic pointers; separately lesion reward memory and map memory.
4. **Continuous embodiment:** velocity control, carrying food, learned return navigation, moving obstacles, and energy needs.

Each milestone should beat an appropriate baseline on a predeclared test set before expanding. See `DESIGN.md` for implementation boundaries and current limits.

## References

- [Nengo](https://github.com/nengo/nengo)
- [Learning examples](https://www.nengo.ai/nengo/examples.html)
- [PES learning](https://www.nengo.ai/nengo/examples/learning/learn-communication-channel.html)
- [NengoSPA](https://www.nengo.ai/nengo-spa/) for potential future spatial memory

Required environment variables: **none** for the local dashboard or benchmark.
