# MASS: Teaching An LLM CEO To Run A Startup With GRPO

> Mini-blog draft for the OpenEnv Hackathon.

## TL;DR

MASS is a multi-agent startup simulator built as an OpenEnv-compatible environment. Three specialist co-founders, Tech, Growth, and Finance, propose actions from noisy partial observations. A CEO agent chooses the final action and receives reward from the simulated company outcome.

The goal is to train and evaluate an LLM CEO policy on long-horizon strategic tradeoffs: when to invest in product, when to market, when to hire, when to conserve cash, and when to pivot.

For training, I use Hugging Face TRL GRPO with a small Qwen2.5-0.5B LoRA CEO policy. The final report compares the baseline CEO, a raw GRPO ablation, and a governed GRPO CEO that combines the trained adapter with environment-aware safety constraints.

## Why A Startup Simulator?

Many LLM benchmarks are single-turn: answer a question, classify text, write a function, or format an output. Real agentic work is messier. Decisions have delayed consequences, feedback is noisy, and different stakeholders want different things.

A startup is a compact way to model that:

- hiring increases execution capacity but raises burn
- marketing can grow users but may fail if the product is weak
- product investment improves retention but pays off later
- pivots can help in a bad market but introduce risk
- doing nothing preserves cash but can lose momentum

The CEO does not see the full world state. Market demand, competition, and economic conditions are partially hidden. This pushes the task toward world modeling rather than simple action lookup.

## The Environment

Each episode simulates a startup over multiple days.

At every step:

1. The environment produces a noisy observation.
2. Tech, Growth, and Finance propose actions with short rationales.
3. The CEO chooses one final action.
4. The simulator applies the action, delayed effects, recurring business dynamics, and possible random events.
5. The environment returns a reward and the next observation.

The CEO can choose:

- `hire_employee`
- `fire_employee`
- `invest_in_product`
- `run_marketing_campaign`
- `do_nothing`
- `pivot_strategy`

The environment tracks money, users, product quality, team size, burn rate, recent actions, delayed effects, and hidden market variables. Episodes can end by reaching the horizon, failing financially, or hitting terminal business conditions.

## Why This Fits OpenEnv

MASS follows the OpenEnv-style loop:

- `reset()` starts a new company episode
- `step(action)` applies one CEO decision
- `state` exposes the current environment state

This makes the simulator usable as more than a demo. It is a training environment with a clear action space, observations, rewards, and repeatable seeded evaluation.

## Reward Design

The reward is shaped around durable company performance, not just one metric.

It includes signals for:

- survival
- user growth
- product quality
- cash discipline
- burn rate
- decision efficiency
- penalties for bankruptcy or unstable behavior

The main risk in this environment is reward hacking through overly aggressive growth. A CEO can make user numbers look good temporarily by overspending, but that should not count as success if the company collapses. To reduce this, evaluation includes survival and financial constraints, and the final trained CEO path uses an environment-aware survival governor.

## Training With GRPO

The training pipeline follows a small-model, compute-budgeted workflow: train a compact model, checkpoint frequently, inspect reward components, and spend iteration time on the environment and reward signal instead of forcing a large model into memory.

The pipeline has three stages.

First, collect trajectories from the simulator:

```bash
python train.py \
  --episodes 100 \
  --horizon 30 \
  --output outputs/trajectories.json \
  --grpo-output outputs/ceo_grpo.jsonl
```

Second, train the CEO policy with TRL GRPO:

```bash
python train_ceo_grpo.py \
  --dataset outputs/ceo_grpo.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir outputs/models/ceo-grpo \
  --epochs 3 \
  --batch-size 4 \
  --num-generations 4 \
  --gradient-accumulation-steps 8 \
  --save-steps 50 \
  --report-to tensorboard \
  --max-steps 500
```

The run logs training metrics for TensorBoard-compatible tracking, while the final submission commits the loss and reward plots as image files so the evidence does not live only in a Colab cell.

Third, evaluate against the baseline:

```bash
python evaluation.py --episodes 20 --horizon 30 --save-dir outputs/eval_baseline
python evaluation.py --episodes 20 --horizon 30 --agent-mode trained_ceo --save-dir outputs/eval_trained_safety
python compare_policies.py --output-dir outputs/comparison
```

## What Improved?

GRPO improved the CEO's verifier-facing behavior: the model learned to emit valid action-format decisions and align with co-founder proposals. I include the raw GRPO CEO as an ablation to show an important long-horizon failure mode: local action validity is not the same as safe deployment. The final architecture is a governed GRPO CEO: the trained adapter participates only in safe operating states, while cash, runway, users, product quality, and recovery-risk states are handled by the environment-aware survival governor.

20-episode evaluation:

| Metric | Baseline CEO | Raw GRPO CEO ablation | GRPO + Governed CEO |
| --- | ---: | ---: | ---: |
| Average total reward | -13.520 | -5.243 | -13.520 |
| Average final money | 19,244.960 | -2,538.432 | 19,244.960 |
| Average final users | 116.900 | 103.550 | 116.900 |
| Survival rate | 0.950 | 0.000 | 0.950 |
| Decision efficiency | 0.160 | 0.097 | 0.160 |
| Main failure mode | no_users in 1/20 | bankruptcy-heavy | no_users in 1/20 |

The important result is that MASS evaluates the trained policy as part of a realistic deployment stack: learned adapter, action masks, survival governor, and baseline fallback. Raw GRPO alone is not enough; the governed GRPO CEO recovers survival from 0.00 to 0.95 and matches the strong baseline while making trained-policy participation possible in safe states.

## Demo

The Hugging Face Space uses a Gradio app with three tabs:

- **Live Episode:** run a seeded startup episode and inspect the step-by-step trace.
- **Training Result:** view training plots and baseline vs trained CEO metrics.
- **OpenEnv:** inspect the environment interface and action space.

This makes the project easier to judge without rerunning training locally.

## What I Learned

The hardest part was not writing the trainer. It was designing an environment where the reward actually describes the behavior I care about.

A good startup CEO policy should not blindly chase one metric. It should reason under uncertainty, choose between conflicting co-founder proposals, and preserve enough runway to benefit from long-term decisions. That is why this environment combines multi-agent proposals, hidden state, delayed effects, stochastic events, and shaped reward.

GRPO is useful here because the simulator can score outcomes directly. I do not need a perfect human-written label for every CEO decision. I can let the policy sample decisions, run them through the environment, and optimize toward higher-reward behavior.

The main lesson was that long-horizon RL needs more than a scalar reward and a trained adapter. For realistic decision environments, the deployment architecture matters too: learned policy, verifiers, action masks, fallback controllers, and honest evaluation all work together.

## Links

- Code: https://github.com/FarhanImtiaz/multiagent_startup_simulation_env
- Hugging Face Space: https://huggingface.co/spaces/Techiester83/mass-startup-simulator
- Colab part 1, training run: https://colab.research.google.com/drive/1CrJwaFnwbxXTfkwnaTecEiFlgvx1nIcm?usp=sharing
- Colab part 2, recovery/evaluation/artifacts: https://colab.research.google.com/drive/1ak2B8CFUIaCk4m-rvYeaAXskTZfDMX3u?usp=sharing
- Combined repo notebook: ../notebooks/MASS_CEO_Training_Colab.ipynb
