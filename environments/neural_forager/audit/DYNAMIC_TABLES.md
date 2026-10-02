# Dynamic-rate follow-up

Means pool both exact weight sources.

| Condition | Receiver | MSE | Categorical accuracy |
|---|---|---:|---:|
| clean | spiking | 0.000032 | 100.000% |
| clean | rate | 0.000015 | 100.000% |
| clean | rate_filter_8ms | 0.000089 | 100.000% |
| clean | rate_filter_20ms | 0.006384 | 100.000% |
| current_noise | spiking | 0.000559 | 100.000% |
| current_noise | rate | 0.002353 | 100.000% |
| current_noise | rate_filter_8ms | 0.000321 | 100.000% |
| current_noise | rate_filter_20ms | 0.006693 | 100.000% |

Positive MSE differences favor the non-spiking filter. Paired maze-bootstrap intervals.

| Filter | Spiking minus filtered MSE | 95% interval | Holm p |
|---|---:|---:|---:|
| rate_filter_8ms | 0.000238 | [0.000192, 0.000286] | 0.00781 |
| rate_filter_20ms | -0.006135 | [-0.006324, -0.005959] | 1.00000 |
