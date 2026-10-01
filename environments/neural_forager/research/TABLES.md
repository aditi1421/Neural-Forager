# Generated result tables

Disrupted episodes only; 96 episodes per method.

| Method | Recovery | Unchanged-memory damage | Capped moves | Pose accuracy |
|---|---:|---:|---:|---:|
| selective | 96/96 | 0.000% | 20.81 | 98.8% |
| ungated | 96/96 | 0.150% | 18.94 | 97.0% |
| no_localization | 50/96 | 4.056% | 68.29 | 50.0% |
| tabular | 96/96 | 0.000% | 20.81 | 98.8% |

| Scenario | Method | Recovery | Damage | Capped moves | First diagnoses |
|---|---|---:|---:|---:|---|
| stable | selective | 24/24 | 0.000% | 14.08 | {'none': 24} |
| stable | ungated | 24/24 | 0.000% | 14.08 | {'none': 24} |
| stable | no_localization | 24/24 | 0.000% | 14.08 | {'none': 24} |
| stable | tabular | 24/24 | 0.000% | 14.08 | {'none': 24} |
| reward | selective | 24/24 | 0.000% | 22.00 | {'reward': 24} |
| reward | ungated | 24/24 | 0.000% | 22.00 | {'reward': 24} |
| reward | no_localization | 24/24 | 0.000% | 22.00 | {'reward': 24} |
| reward | tabular | 24/24 | 0.000% | 22.00 | {'reward': 24} |
| route | selective | 24/24 | 0.000% | 22.92 | {'route': 20, 'none': 4} |
| route | ungated | 24/24 | 0.000% | 18.92 | {'none': 24} |
| route | no_localization | 24/24 | 0.000% | 18.92 | {'none': 24} |
| route | tabular | 24/24 | 0.000% | 22.92 | {'route': 20, 'none': 4} |
| drift | selective | 24/24 | 0.000% | 14.25 | {'drift': 24} |
| drift | ungated | 24/24 | 0.278% | 14.92 | {'drift': 24} |
| drift | no_localization | 2/24 | 8.690% | 112.25 | {'none': 22, 'reward': 2} |
| drift | tabular | 24/24 | 0.000% | 14.25 | {'drift': 24} |
| combined | selective | 24/24 | 0.000% | 24.08 | {'drift': 22, 'route': 2} |
| combined | ungated | 24/24 | 0.321% | 19.92 | {'drift': 24} |
| combined | no_localization | 0/24 | 7.532% | 120.00 | {'none': 22, 'reward': 2} |
| combined | tabular | 24/24 | 0.000% | 24.08 | {'drift': 22, 'route': 2} |

Paired differences: selective minus control; percentile 95% cluster bootstrap over 12 mazes.

| Control | Metric | Difference | 95% interval |
|---|---|---:|---:|
| ungated | memory_damage (percentage points) | -0.150 | [-0.329, 0.000] |
| ungated | success (percentage points) | 0.000 | [0.000, 0.000] |
| ungated | capped_latency (moves) | 1.875 | [0.708, 3.167] |
| no_localization | memory_damage (percentage points) | -4.056 | [-5.242, -2.990] |
| no_localization | success (percentage points) | 47.917 | [43.750, 50.000] |
| no_localization | capped_latency (moves) | -47.479 | [-50.750, -42.562] |
| tabular | memory_damage (percentage points) | 0.000 | [0.000, 0.000] |
| tabular | success (percentage points) | 0.000 | [0.000, 0.000] |
| tabular | capped_latency (moves) | 0.000 | [0.000, 0.000] |
