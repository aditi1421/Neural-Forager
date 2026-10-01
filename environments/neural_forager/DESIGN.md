# Implementation boundaries

`world.py` owns privileged state and seeded changes. The immutable `Observation` contains only position and four local movement affordances. Shortest paths are used to choose interventions and compute metrics, never by an agent.

`agents.py` owns policies and Nengo simulations. The neural agent stores values in plastic connections; the tabular baseline stores them in a NumPy array. Spike and decoded-value samples use bounded deques, avoiding probe-history accumulation. The weight-norm monitor reads a Nengo simulator signal for diagnostics, protected by the version pin.

`experiment.py` owns trials, scheduling, and metrics. Each move follows observe/select, transition, and learn. Simulation time also includes re-cuing recent places for credit assignment; it does not equal wall time or world moves. Logs are bounded by the configured move budget.

`server.py` owns isolated WebSocket sessions. Simulation runs in a worker thread and session mutations use a lock. Disconnect cleanup uses the same lock so it cannot close a simulator mid-move. Pydantic validates commands. The browser renders real simulation outputs and never simulates the agent.

`prime.py` adapts the same local experiment to the installed Verifiers v1 API. Reward is food per move; coverage is an additional metric. The framework program does not call an LLM endpoint.

`cli.py` provides serving and paired comparisons. JSON logs are the reproducibility artifact. The dashboard and batch evaluator share the same Experiment class.

## Scientific limits

- Position is exact and place indexing is fixed; localization and place-field learning are unsolved.
- Memory is synaptic action-value memory. There is no learned occupancy map, recurrent working-memory circuit, language understanding, hunger model, or continuous motor control yet.
- Python eligibility traces and re-cuing supply temporal credit. This is a hybrid prototype, not a fully biological brain.
- No intrinsic curiosity or model-based planning: sparse rewards and stale values can trap the agent after changes.
- A seed controls world and neural initialization, with a separate policy RNG stream. Future experiments should vary these factors independently.
- Fresh networks train online in each test maze. This is not transfer learning or zero-shot generalization.
- The reward is designed by us. No claim of biological validation, energy efficiency, or superiority to conventional reinforcement learning.
