# Final Submission Checklist

Use this checklist before submitting. These are the automated validation gates and judge-facing deliverables.

## Automated Validation Gates

- [x] Public Hugging Face Space URL is submitted.
- [x] Space is public and cloneable.
- [x] Space works from a logged-out browser.
- [x] Space URL does not return 404.
- [x] `openenv.yaml` is parseable.
- [x] Environment has `reset`, `step`, and `state`.
- [x] OpenEnv validation helper passes.
- [x] Training evidence is committed as image files.
- [x] Loss curve is committed as `.png` or `.jpg`.
- [x] Reward curve is committed as `.png` or `.jpg`.
- [x] Plots are not only in Colab or WandB.
- [x] Runnable training script exists.
- [x] Public Colab notebook link is in README.
- [x] README links every deliverable.
- [x] README embeds key plots inline.

## Links To Fill

- GitHub repo: https://github.com/FarhanImtiaz/multiagent_startup_simulation_env
- Hugging Face Space: https://huggingface.co/spaces/Techiester83/mass-startup-simulator
- Public Colab part 1, training run: https://colab.research.google.com/drive/1CrJwaFnwbxXTfkwnaTecEiFlgvx1nIcm?usp=sharing
- Public Colab part 2, recovery/evaluation/artifacts: https://colab.research.google.com/drive/1ak2B8CFUIaCk4m-rvYeaAXskTZfDMX3u?usp=sharing
- Combined repo notebook: `notebooks/MASS_CEO_Training_Colab.ipynb`
- Mini-blog / writeup: https://github.com/FarhanImtiaz/multiagent_startup_simulation_env/blob/main/docs/miniblog_draft.md

## Final Artifact Copy

Committed final artifacts:

- `docs/assets/loss_curve.png`
- `docs/assets/reward_curve.png`
- `docs/assets/reward_comparison.png`
- `docs/assets/policy_comparison.png`
- `docs/assets/policy_summary.png`

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
