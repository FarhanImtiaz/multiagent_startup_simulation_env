# MASS Training Steps

This project currently trains the CEO decision policy first. The co-founder agents
produce proposals, and the CEO chooses the final action, so CEO fine-tuning gives
the clearest before/after comparison.

## 1. Run A Baseline

```bash
venv/bin/python evaluation.py --episodes 20 --horizon 30 --save-dir outputs/eval
```

Record the baseline values from `outputs/eval/evaluation_summary.json`.

## 2. Collect Training Trajectories

```bash
venv/bin/python train.py \
  --episodes 50 \
  --horizon 30 \
  --output outputs/trajectories.json \
  --sft-output outputs/ceo_sft.jsonl \
  --preference-output outputs/ceo_preferences.jsonl \
  --grpo-output outputs/ceo_grpo.jsonl
```

This creates:

- `outputs/trajectories.json`: full simulator rollouts
- `outputs/ceo_sft.jsonl`: supervised CEO decision examples
- `outputs/ceo_preferences.jsonl`: chosen/rejected action pairs for later DPO-style tuning
- `outputs/ceo_grpo.jsonl`: prompt-only CEO records with simulator context for GRPO rewards

## 3. Install Training Dependencies

Use this on a GPU machine or Colab runtime:

```bash
python -m pip install -r requirements-training.txt
```

## 4. Optimize The CEO With GRPO

Small GPU-friendly first run:

```bash
python train_ceo_grpo.py \
  --dataset outputs/ceo_grpo.jsonl \
  --model Qwen/Qwen3-0.6B \
  --output-dir outputs/models/ceo-grpo \
  --epochs 1 \
  --batch-size 4 \
  --num-generations 4 \
  --gradient-accumulation-steps 8
```

For a smoke test, add `--max-steps 10`.

## 5. Evaluate After Training

After the adapter is saved in `outputs/models/ceo-grpo`, run:

```bash
python evaluation.py --episodes 20 --horizon 30 --agent-mode trained_ceo --save-dir outputs/eval_trained
```

Compare `outputs/eval/evaluation_summary.json` with
`outputs/eval_trained/evaluation_summary.json`.
