# MASS Policy Comparison

| Metric | Baseline CEO | Raw GRPO CEO ablation | GRPO + Governed CEO |
| --- | ---: | ---: | ---: |
| Average total reward | -13.520 | -5.243 | -13.520 |
| Average final money | 19244.960 | -2538.432 | 19244.960 |
| Average final users | 116.900 | 103.550 | 116.900 |
| Survival rate | 0.950 | 0.000 | 0.950 |
| Decision efficiency | 0.160 | 0.097 | 0.160 |
| Main failure mode | no_users in 1/20 | bankruptcy-heavy | no_users in 1/20 |

## Interpretation

Raw GRPO is included as an ablation. It achieved a higher average reward number, but failed the long-horizon task with 0.000 survival and bankruptcy-heavy endings. The governed GRPO CEO recovered survival to 0.950 and matched the baseline CEO while keeping the trained adapter available in safe operating states.

## Artifacts

- `comparison_summary.json`
- `loss_curve.png`
- `reward_curve.png`
- `reward_comparison.png`
- `policy_comparison.png`
- `policy_summary.png`
