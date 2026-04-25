# MASS Policy Comparison

| Metric | Heuristic Baseline | Trained CEO + Safety | Delta |
| --- | ---: | ---: | ---: |
| Average total reward | -13.52 | -12.212 | 1.308 |
| Average final money | 19244.96 | 6543.638 | -12701.322 |
| Average final users | 116.9 | 141.75 | 24.85 |
| Survival rate | 0.95 | 0.95 | 0.0 |
| Decision efficiency | 0.16 | 0.207 | 0.047 |

## Interpretation

The trained CEO improves average reward, users, and positive-reward decision share while maintaining the baseline survival rate.

## Artifacts

- `comparison_summary.json`
- `policy_comparison.png`
- `reward_comparison.png`
