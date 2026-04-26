# Hugging Face Space Deployment

This project already includes the files needed for a Gradio Space:

- `app.py`
- `space_demo.py`
- `requirements.txt`
- `docs/assets/*.png`
- `docs/comparison_summary.json`
- `docs/comparison_report.md`

## Create The Space

1. Go to Hugging Face.
2. Create a new Space.
3. Suggested settings:
   - SDK: Gradio
   - Hardware: CPU basic is enough for the demo
   - Visibility: Public
4. Push this repository to the Space repo, or connect it from GitHub.

## Space README Front Matter

If Hugging Face asks for Space metadata, use this at the top of the Space `README.md`:

```yaml
---
title: MASS Startup Simulator
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---
```

## Local Smoke Test

Run this before publishing:

```bash
pip install -r requirements.txt
python app.py
```

The app should open with three tabs:

- Live Episode
- Training Result
- OpenEnv

## What To Update After Colab Training

After the final Colab run, copy these generated artifacts back into the repo:

- `docs/assets/loss_curve.png`
- `docs/assets/reward_curve.png`
- `docs/assets/reward_comparison.png`
- `docs/assets/policy_comparison.png`
- `docs/comparison_summary.json`
- `docs/comparison_report.md`

Then commit and push. The Space will show the updated training results.

## Submission Checklist

- Space loads without dependency errors.
- Live Episode tab returns a trace.
- Training Result tab shows all plots.
- Comparison table has final trained metrics.
- README links include the Space URL.
