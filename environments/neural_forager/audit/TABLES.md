# Generated spiking audit results

Primary spiking-minus-rate accuracy: -0.0112 percentage points; 95% interval [-0.0223, 0.0000].

| Neurons/cell | Condition | Spiking accuracy | Rate accuracy | Difference (pp) | Holm p (spiking > rate) |
|---:|---|---:|---:|---:|---:|
| 1 | clean | 100.000% | 100.000% | 0.000 | 1.0000 |
| 1 | current_noise | 100.000% | 100.000% | 0.000 | 1.0000 |
| 1 | lesion_50 | 80.268% | 80.268% | 0.000 | 1.0000 |
| 1 | short_read | 97.396% | 100.000% | -2.604 | 1.0000 |
| 12 | clean | 100.000% | 100.000% | 0.000 | 1.0000 |
| 12 | current_noise | 100.000% | 100.000% | 0.000 | 1.0000 |
| 12 | lesion_50 | 99.911% | 99.911% | 0.000 | 1.0000 |
| 12 | short_read | 99.955% | 100.000% | -0.045 | 1.0000 |

| Condition | Backend | Recovery | Capped moves | Pose accuracy |
|---|---|---:|---:|---:|
| clean | spiking | 64/64 | 18.69 | 96.16% |
| clean | rate | 64/64 | 18.69 | 96.16% |
| current_noise | spiking | 64/64 | 18.69 | 96.16% |
| current_noise | rate | 64/64 | 18.69 | 96.16% |
| lesion_50 | spiking | 64/64 | 22.00 | 96.81% |
| lesion_50 | rate | 64/64 | 22.00 | 96.81% |

Analog errors are secondary descriptive endpoints; these intervals are unadjusted.

| Neurons/cell | Condition | Spiking MSE | Rate MSE | Difference, 95% interval |
|---:|---|---:|---:|---:|
| 1 | clean | 0.00060 | 0.00000 | 0.00060 [0.00058, 0.00062] |
| 1 | current_noise | 0.00426 | 0.00480 | -0.00053 [-0.00095, -0.00015] |
| 1 | lesion_50 | 0.27728 | 0.27701 | 0.00027 [0.00025, 0.00030] |
| 1 | short_read | 0.14174 | 0.11028 | 0.03146 [0.02937, 0.03371] |
| 12 | clean | 0.00004 | 0.00000 | 0.00004 [0.00004, 0.00004] |
| 12 | current_noise | 0.00060 | 0.00244 | -0.00185 [-0.00190, -0.00180] |
| 12 | lesion_50 | 0.15365 | 0.15325 | 0.00040 [-0.00009, 0.00088] |
| 12 | short_read | 0.13604 | 0.11028 | 0.02576 [0.02467, 0.02691] |

| Neurons/cell | Condition | Training source | Same-weight receiver difference (pp) |
|---:|---|---|---:|
| 1 | clean | spiking | 0.0000 |
| 1 | clean | rate | 0.0000 |
| 1 | current_noise | spiking | 0.0000 |
| 1 | current_noise | rate | 0.0000 |
| 1 | lesion_50 | spiking | 0.0000 |
| 1 | lesion_50 | rate | 0.0000 |
| 1 | short_read | spiking | -2.6042 |
| 1 | short_read | rate | -3.2292 |
| 12 | clean | spiking | 0.0000 |
| 12 | clean | rate | 0.0000 |
| 12 | current_noise | spiking | 0.0000 |
| 12 | current_noise | rate | 0.0000 |
| 12 | lesion_50 | spiking | 0.0000 |
| 12 | lesion_50 | rate | 0.0000 |
| 12 | short_read | spiking | -0.0446 |
| 12 | short_read | rate | -0.0446 |
