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

# MASS Startup Simulator

MASS is a multi-agent startup decision environment for training and evaluating an LLM CEO.

Tech, Growth, and Finance co-founders propose actions from noisy partial observations. The CEO chooses the final strategy and receives reward from the simulated company outcome.

## Space Tabs

- **Live Episode:** run a seeded startup episode and inspect the day-by-day trace.
- **Training Result:** view training plots and baseline vs trained CEO metrics.
- **OpenEnv:** inspect the environment interface and valid action space.

## Environment Loop

The OpenEnv manifest is `openenv.yaml`.

The environment exposes:

- `reset()`
- `step(action)`
- `state`

Valid CEO actions:

- `hire_employee`
- `fire_employee`
- `invest_in_product`
- `run_marketing_campaign`
- `do_nothing`
- `pivot_strategy`

## Training

The CEO policy is trained with Hugging Face TRL GRPO. Final comparison metrics and plots are shown in the Training Result tab after the Colab evaluation artifacts are copied into `docs/`.
