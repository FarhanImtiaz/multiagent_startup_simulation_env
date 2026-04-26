# MASS: Teaching An LLM CEO To Run A Startup With GRPO

> Mini-blog draft for the OpenEnv Hackathon. Replace the TODO links and final metrics after the Colab run completes.

## TL;DR

MASS is a multi-agent startup simulator built as an OpenEnv-compatible environment. Three specialist co-founders, Tech, Growth, and Finance, propose actions from noisy partial observations. A CEO agent chooses the final action and receives reward from the simulated company outcome.

The goal is to train an LLM CEO policy that gets better at long-horizon strategic tradeoffs: when to invest in product, when to market, when to hire, when to conserve cash, and when to pivot.

For training, I use Hugging Face TRL GRPO with a Qwen2.5 LoRA CEO policy. The trained CEO is compared against a hand-written heuristic CEO baseline in the same simulator.

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

The main risk in this environment is reward hacking through overly aggressive growth. A CEO can make user numbers look good temporarily by overspending, but that should not count as success if the company collapses. To reduce this, evaluation includes survival and financial constraints, and the trained CEO path uses a safety gate.

## Training With GRPO

The training pipeline has three stages.

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
  --max-steps 500
```

Third, evaluate against the baseline:

```bash
python evaluation.py --episodes 20 --horizon 30 --save-dir outputs/eval_baseline
python evaluation.py --episodes 20 --horizon 30 --agent-mode trained_ceo --save-dir outputs/eval_trained_safety
python compare_policies.py --output-dir outputs/comparison
```

## What Improved?

Final GRPO training is still running. This section will be updated with the completed Colab evaluation.

Target comparison:

| Metric | Heuristic Baseline | Trained CEO + Safety Gate |
| --- | ---: | ---: |
| Average total reward | TODO | TODO |
| Average final users | TODO | TODO |
| Survival rate | TODO | TODO |
| Decision efficiency | TODO | TODO |

The ideal result is not just a higher reward number. I want the trained CEO to improve reward, growth, or efficiency while preserving survival. A model that grows quickly but bankrupts the company is not a good CEO.

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

## Links

- Code: TODO
- Hugging Face Space: TODO
- Colab notebook: TODO
- Final mini-blog / video / slides: TODO
