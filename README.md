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

# MASS: Multi-Agent Startup Simulator

MASS is an OpenEnv-compatible reinforcement-learning environment where an LLM CEO learns to run a startup over a long horizon. Tech, Growth, and Finance co-founders each propose actions from noisy partial observations, and the CEO chooses the final company strategy.

The project is built for the OpenEnv hackathon themes of **multi-agent interaction**, **long-horizon planning**, and **world modeling**. The agent has to balance product quality, user growth, cash runway, employee count, burn rate, market demand, competition, and stochastic external events.

## Deliverables

Replace these links before final submission:

- Hugging Face Space: TODO
- Colab training notebook: TODO
- Mini-blog / writeup: TODO
- Code repository: https://github.com/FarhanImtiaz/multiagent_startup_simulation_env

Supporting docs:

- [Hugging Face Space deployment checklist](docs/huggingface_space_deployment.md)
- [Final submission checklist](docs/final_submission_checklist.md)
- [Mini-blog draft](docs/miniblog_draft.md)
- [Training steps](TRAINING_STEPS.md)

## Why This Environment

Many LLM demos optimize for one-shot text quality. MASS instead asks whether an LLM can act inside a dynamic system, receive feedback, and improve a policy over repeated decisions.

Startup strategy is useful for this because decisions are coupled and delayed:

- hiring improves execution but increases burn
- marketing can grow users but wastes cash if product quality is weak
- product investment improves future retention but may not pay off immediately
- pivots can help under bad market conditions but carry risk
- survival requires balancing growth against runway

This makes the environment more interesting than a single-turn classification or formatting task. The CEO must reason over competing proposals, noisy observations, and delayed consequences.

## Environment Design

At each timestep:

1. The environment exposes a noisy observation of the startup.
2. Three co-founder agents propose actions:
   - Tech Co-founder
   - Growth Co-founder
   - Finance Co-founder
3. The CEO chooses one final action.
4. The simulator applies delayed effects, the chosen action, recurring business dynamics, and a possible external event.
5. The environment returns a reward and the next observation.

Episodes terminate when the startup reaches the horizon, runs out of money, or hits a success/failure condition.

## What The Agent Observes

The CEO sees a partial, noisy view of the company:

- cash / runway signal
- user count
- product quality
- employee count
- burn rate
- market and competition signals
- recent events
- co-founder proposals and rationales

Hidden state such as true market demand and some event dynamics is not directly exposed.

## Action Space

The CEO selects one of:

- `hire_employee`
- `fire_employee`
- `invest_in_product`
- `run_marketing_campaign`
- `do_nothing`
- `pivot_strategy`

These actions affect company state immediately and through delayed effects.

## Reward Design

The reward function is shaped to encourage durable startup performance rather than one-dimensional growth. It accounts for:

- survival
- user growth
- product quality
- financial discipline
- cash runway
- action efficiency
- penalties for bankruptcy or poor strategic tradeoffs

The key design goal is to prevent trivial policies such as spending all cash on growth or doing nothing forever. The trained CEO is also evaluated with a safety gate so a high-growth policy cannot collapse into repeated bankruptcy behavior.

## Training Pipeline

The project uses Hugging Face TRL GRPO for the CEO policy.

Pipeline:

1. Run the simulator to collect CEO decision trajectories.
2. Export GRPO-ready examples.
3. Train a Qwen2.5 LoRA CEO policy with TRL GRPO.
4. Evaluate the trained CEO against the heuristic baseline.
5. Commit final loss/reward plots and comparison metrics.

Generate training data:

```bash
python3 train.py \
  --episodes 100 \
  --horizon 30 \
  --output outputs/trajectories.json \
  --grpo-output outputs/ceo_grpo.jsonl
```

Train with GRPO:

```bash
python3 train_ceo_grpo.py \
  --dataset outputs/ceo_grpo.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir outputs/models/ceo-grpo \
  --epochs 3 \
  --batch-size 4 \
  --num-generations 4 \
  --gradient-accumulation-steps 8 \
  --save-steps 50 \
  --max-steps 500
```

Evaluate baseline and trained CEO:

```bash
python3 evaluation.py --episodes 20 --horizon 30 --save-dir outputs/eval_baseline
python3 evaluation.py --episodes 20 --horizon 30 --agent-mode trained_ceo --save-dir outputs/eval_trained_safety
python3 compare_policies.py --output-dir outputs/comparison
```

## What Improved After Training

Final GRPO training is in progress. This section will be updated with the completed Colab run.

Planned comparison:

| Metric | Heuristic Baseline | Trained CEO + Safety Gate |
| --- | ---: | ---: |
| Average total reward | TODO | TODO |
| Average final users | TODO | TODO |
| Survival rate | TODO | TODO |
| Decision efficiency | TODO | TODO |

Plots to include after the final run:

- training loss curve
- baseline vs trained reward curve
- before/after metric comparison
- policy comparison summary

## Hugging Face Space Demo

The Gradio demo is implemented in `app.py`.

It includes:

- **Live Episode:** run a multi-agent startup episode and inspect the day-by-day trace.
- **Training Result:** view training plots and baseline vs trained CEO metrics.
- **OpenEnv:** inspect the environment interface and action space.

Run locally:

```bash
pip install -r requirements.txt
python3 app.py
```

Deployment notes are in [docs/huggingface_space_deployment.md](docs/huggingface_space_deployment.md).

## OpenEnv Compatibility

The OpenEnv manifest is [openenv.yaml](openenv.yaml). The package wrapper lives in `mass_startup_env/`.

The environment exposes the standard loop:

- `reset()`
- `step(action)`
- `state`

Local smoke test:

```python
from mass_startup_env import StartupAction, StartupOpenEnv

env = StartupOpenEnv(max_days=3, seed=7)
obs = env.reset()
obs = env.step(StartupAction(action="invest_in_product"))
print(obs.reward, obs.done, env.state.step_count)
```

Validation helper:

```bash
python3 scripts/validate_openenv_package.py
```

Server entrypoint:

```bash
python3 -m mass_startup_env.server.app
```

## Quick Start

Install:

```bash
pip install -r requirements.txt
```

Run one episode:

```bash
python3 simulate.py
```

Run a short debug episode:

```bash
python3 simulate.py --horizon 10 --show-hidden-state
```

Run baseline evaluation:

```bash
python3 evaluation.py --episodes 20 --horizon 30 --save-dir outputs/eval
```

Run the Space app locally:

```bash
python3 app.py
```

## Main Files

- `environment.py`: core startup simulator, hidden state, delayed effects, events, rewards, and termination.
- `agents.py`: heuristic Tech, Growth, Finance, and CEO policies.
- `simulate.py`: single-episode runner.
- `evaluation.py`: multi-episode evaluation and report export.
- `train.py`: trajectory collection and dataset export.
- `train_ceo_grpo.py`: Hugging Face TRL GRPO training script.
- `llm_agents.py`: trained CEO loading, prompt scaffolding, and safety fallback logic.
- `compare_policies.py`: baseline vs trained CEO comparison artifacts.
- `mass_startup_env/`: OpenEnv-style package wrapper and server.
- `app.py`: Gradio Hugging Face Space demo.
- `notebooks/MASS_CEO_Training_Colab.ipynb`: Colab training workflow.

## Results Artifacts

Final artifacts should be committed under:

- `docs/assets/loss_curve.png`
- `docs/assets/reward_curve.png`
- `docs/assets/reward_comparison.png`
- `docs/assets/policy_comparison.png`
- `docs/comparison_summary.json`
- `docs/comparison_report.md`

Generated training/evaluation outputs are written to `outputs/` and are intentionally ignored by Git.

## Known Limitations

- The simulator is compact and designed for hackathon-scale training, not a full business benchmark.
- The trained model is a LoRA CEO policy and should be evaluated through the provided safety gate.
- Final metrics are pending the current GRPO Colab run.

## Next Improvements

- Add more reward components and anti-reward-hacking checks.
- Add curriculum variants with easier and harder market conditions.
- Compare GRPO against prompt-only and heuristic baselines.
- Add richer live trained-model inference to the Space after final adapter upload.
