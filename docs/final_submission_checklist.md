# Final Submission Checklist

Use this checklist before submitting. These are the automated validation gates and judge-facing deliverables.

## Automated Validation Gates

- [ ] Public Hugging Face Space URL is submitted.
- [ ] Space is public and cloneable.
- [ ] Space works from a logged-out browser.
- [ ] Space URL does not return 404.
- [ ] `openenv.yaml` is parseable.
- [ ] Environment has `reset`, `step`, and `state`.
- [ ] OpenEnv validation helper passes.
- [ ] Training evidence is committed as image files.
- [ ] Loss curve is committed as `.png` or `.jpg`.
- [ ] Reward curve is committed as `.png` or `.jpg`.
- [ ] Plots are not only in Colab or WandB.
- [ ] Runnable training script exists.
- [ ] Public Colab notebook link is in README.
- [ ] README links every deliverable.
- [ ] README embeds key plots inline.

## Links To Fill

- GitHub repo: TODO
- Hugging Face Space: TODO
- Public Colab notebook: TODO
- Hugging Face mini-blog / video / slides: TODO

## Final Artifact Copy

After Colab evaluation, copy these into the repo:

- `docs/assets/loss_curve.png`
- `docs/assets/reward_curve.png`
- `docs/assets/reward_comparison.png`
- `docs/assets/policy_comparison.png`
- `docs/comparison_summary.json`
- `docs/comparison_report.md`

## Final Smoke Tests

Run locally:

```bash
python3 scripts/validate_openenv_package.py
python3 simulate.py --horizon 5
python3 evaluation.py --episodes 2 --horizon 10 --save-dir outputs/smoke_eval
python3 app.py
```

Run in Colab or local GPU environment:

```bash
python3 train_ceo_grpo.py --dataset outputs/ceo_grpo.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --output-dir outputs/models/ceo-grpo-smoke --max-steps 10
```

## README Must Show

- Problem and motivation.
- What the agent observes.
- Valid actions.
- Reward design.
- Training pipeline.
- Baseline vs trained results.
- Embedded loss and reward plots.
- All submission links.
