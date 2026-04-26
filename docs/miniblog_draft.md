# Training A CEO Agent In A Multi-Agent Startup Simulator

## Summary

MASS is a compact world-modeling environment for startup decision-making. A simulated company is run over multiple days by four roles: Tech, Growth, Finance, and CEO. The three co-founder agents propose actions from partial observations, and the CEO chooses the final action. The goal is to survive while improving users, product quality, and long-term reward under noisy market conditions.

For this project, I built the simulator, exported CEO decision trajectories, and trained a Qwen LoRA CEO policy with Hugging Face TRL GRPO. The trained CEO is evaluated against a heuristic CEO baseline with the same co-founder proposals and a safety gate.

## Why This Environment

Startup strategy is a useful toy domain for long-horizon agency because decisions are delayed, noisy, and cross-functional. Hiring improves execution but raises burn. Marketing can grow users but may waste cash if product quality is weak. Product investment helps retention and future growth but creates delayed payoff. A good CEO policy has to balance these tradeoffs instead of greedily selecting one metric.

MASS includes:

- hidden market state
- partial and noisy observations
- stochastic external events
- delayed action effects
- role-specific co-founder proposals
- shaped rewards for survival, growth, quality, and financial discipline

## Training Setup

The training pipeline has three stages.

First, the simulator collects trajectories from the heuristic multi-agent system:

```bash
python train.py \
  --episodes 100 \
  --horizon 30 \
  --output outputs/trajectories.json \
  --sft-output outputs/ceo_sft.jsonl \
  --preference-output outputs/ceo_preferences.jsonl \
  --grpo-output outputs/ceo_grpo.jsonl
```

Second, the CEO policy is optimized with TRL GRPO:

```bash
python train_ceo_grpo.py \
  --dataset outputs/ceo_grpo.jsonl \
  --model Qwen/Qwen3-0.6B \
  --output-dir outputs/models/ceo-grpo \
  --epochs 3 \
  --batch-size 4 \
  --num-generations 4 \
  --gradient-accumulation-steps 8 \
  --save-steps 50 \
  --max-steps 500
```

Third, the trained CEO is evaluated in the same simulator:

```bash
python evaluation.py --episodes 20 --horizon 30 --save-dir outputs/eval_baseline
python evaluation.py --episodes 20 --horizon 30 --agent-mode trained_ceo --save-dir outputs/eval_trained_safety
python compare_policies.py --output-dir outputs/comparison
```

## Results

Replace this table with the final Colab numbers:

| Metric | Heuristic Baseline | Trained CEO + Safety Gate |
| --- | ---: | ---: |
| Average total reward | TODO | TODO |
| Average final users | TODO | TODO |
| Survival rate | TODO | TODO |
| Decision efficiency | TODO | TODO |

The main result to look for is not just raw reward improvement. The trained CEO should preserve survival while improving growth, reward, or decision efficiency. If the model grows faster but bankrupts the company more often, the safety gate and reward shaping need adjustment.

## Demo

The Gradio demo has two main views:

- Live Episode: run a heuristic multi-agent startup episode and inspect the day-by-day decisions.
- Training Result: inspect the training curve and baseline vs trained CEO comparison.

The Hugging Face Space is intended to make the simulator inspectable without running training locally.

## Lessons Learned

GRPO is a good fit for this environment because the CEO action can be scored by simulator outcomes instead of requiring a fixed supervised label. The hard part is reward design: a naive reward can over-prioritize growth, while a conservative reward can make the policy indistinguishable from the heuristic baseline.

The most useful engineering choice was keeping the simulator deterministic under seeds. That made before/after comparisons, debugging, and Colab training failures much easier to reason about.

## Links

- Code: TODO
- Hugging Face Space: TODO
- Colab notebook: TODO
- Final report/demo video: TODO
