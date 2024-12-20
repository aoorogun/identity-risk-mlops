# Verified evaluation

All event-level records are synthetic and privacy-safe. Results demonstrate the workflow, not production performance.

| Measure | Result |
|---|---:|
| Selected model | Logistic regression |
| Test ROC-AUC | 0.715 |
| Test average precision | 0.352 |
| Test Brier score | 0.128 |
| Maximum F1 | 0.400 |
| Automatic approval rate | 60.6% |
| Manual-review rate | 39.1% |
| Reject/escalate rate | 0.27% |

The 2024 launch scenario raises mean predicted failure probability from 12.8% to 18.3%. Together with a 1.35× traffic multiplier and stricter thresholds, scaled manual-review volume increases from 1,922 baseline sample cases to 4,502 scenario cases. Current numerical-feature drift is healthy.

Three tests, static lint, source compilation and the complete generate/train/scenario/monitor workflow passed. Production use would require probability calibration, legal/fairness review, real outcome labels and client-specific cost validation.

