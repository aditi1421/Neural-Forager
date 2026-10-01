# Validation record

This is the original dashboard/pilot validation record. The subsequent
selective-memory implementation brings the test suite to **42 passing tests**;
its evaluation is documented in [research/REPORT.md](research/REPORT.md).
The `outputs/evals/` directories below are local run locations, excluded from
Git. Portable research evidence is committed under `research/`.

Environment: `neural-forager`, `environments/neural_forager`.

## Installation and correctness

```bash
prime env install neural-forager
.venv/bin/python -m pytest environments/neural_forager/tests -q
node --check environments/neural_forager/neural_forager/static/app.js
```

Installation passed. All **34 tests passed**, including an actual neural step through the WebSocket API that produced nonzero firing activity and changed weights. JavaScript syntax check passed. Nengo emits one upstream NumPy deprecation warning during tests.

The dashboard was opened in Safari and visually inspected. Its maze contrast and initial activity message were corrected after inspection. API tests cover stepping, changes, reset, JSON export, and session isolation. A subsequent native-browser click was not attempted after the computer-use tool reported that the user had changed the app; this avoided interfering with their browser.

## Canonical Prime evaluation

```bash
prime eval run neural-forager --disable-tui -c 1
```

The unchanged base eval defaults ran 5 examples × 3 rollouts, with 600 world moves per rollout: **15 completed, 0 errors**. Repeated same-seed rollouts produced identical scores, as expected. These are deterministic replicates, not 15 independent worlds.

Artifacts: `outputs/evals/neural-forager--openai--gpt-4.1-mini/84cc46d9/`.

```bash
prime eval run neural-forager \
  --env-args '{"config":{"taskset":{"steps":60}}}' \
  -n 50 -r 1 -c 1 --disable-tui -s
```

Broader integration smoke: **50 completed, 0 errors**. This deliberately short test validates execution across seeds, not navigation competence or adaptation.

Artifacts: `outputs/evals/neural-forager--openai--gpt-4.1-mini/45e1c862/` and `results/prime-smoke.log`.

Both harnesses ran local Nengo programs and made no LLM calls despite the default model label in Prime output. Prime saved local artifacts and reported that automatic platform upload requires publishing/linking the environment. No `--skip-upload` flag was used. Nothing was published.

## Behavioral comparison

```bash
.venv/bin/neural-forager evaluate \
  --seeds 101 102 103 --steps 1800 \
  --output environments/neural_forager/results/heldout.json
```

This compares neural, frozen, tabular, and random controllers on paired maze/intervention seeds, with fresh weights for each run. See the README table and the full JSON for outcomes. The behavior is deterministic within the recorded software stack. A final rerun after correcting visited-cell telemetry preserves the same policy and food counts.

The JSON pilot is more informative about behavior than either short Prime integration test. Three seeds are insufficient for a superiority claim. Adaptation failures are retained in the report rather than filtered out.
