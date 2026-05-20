# MASS Training Steps

This project currently trains the CEO decision policy first. The co-founder agents
produce proposals, and the CEO chooses the final action, so CEO GRPO training gives
the clearest before/after comparison.

## Fresh Notebook Setup

Use a GPU runtime, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-training.txt
```

If you are resuming from a previous notebook session, restore the previous
`outputs/models/ceo-grpo` directory before training again. The trainer saves
checkpoints every 50 optimizer steps in that directory.

## 1. Run A Baseline

```bash
python evaluation.py --episodes 20 --horizon 30 --save-dir outputs/eval_baseline
```

Record the baseline values from `outputs/eval_baseline/evaluation_summary.json`.

## 2. Collect Training Trajectories

```bash
python train.py \
  --episodes 100 \
  --horizon 30 \
  --output outputs/trajectories.json \
  --grpo-output outputs/ceo_grpo.jsonl
```

This creates:

- `outputs/trajectories.json`: full simulator rollouts
- `outputs/ceo_grpo.jsonl`: prompt-only CEO records with simulator context for GRPO rewards

## 3. Smoke Test The Trainer

Before spending GPU time, run a short training check:

```bash
python train_ceo_grpo.py \
  --dataset outputs/ceo_grpo.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir outputs/models/ceo-grpo \
  --epochs 1 \
  --batch-size 4 \
  --num-generations 4 \
  --gradient-accumulation-steps 8 \
  --save-steps 50 \
  --max-steps 10 \
  --report-to tensorboard
```

## 4. Optimize The CEO With GRPO

Main run with a checkpoint saved every 50 optimizer steps:

```bash
python train_ceo_grpo.py \
  --dataset outputs/ceo_grpo.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir outputs/models/ceo-grpo \
  --epochs 10 \
  --batch-size 4 \
  --num-generations 4 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-6 \
  --save-steps 50 \
  --logging-steps 10 \
  --max-steps 1000 \
  --report-to tensorboard
```

This produces checkpoints such as:

- `outputs/models/ceo-grpo/checkpoint-50`
- `outputs/models/ceo-grpo/checkpoint-100`
- `outputs/models/ceo-grpo/checkpoint-150`

## 5. Resume Training

To continue from the newest checkpoint in `outputs/models/ceo-grpo`:

```bash
python train_ceo_grpo.py \
  --dataset outputs/ceo_grpo.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir outputs/models/ceo-grpo \
  --epochs 10 \
  --batch-size 4 \
  --num-generations 4 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-6 \
  --save-steps 50 \
  --logging-steps 10 \
  --max-steps 2000 \
  --report-to tensorboard \
  --resume-from-checkpoint latest
```

To resume from a specific checkpoint instead:

```bash
python train_ceo_grpo.py \
  --dataset outputs/ceo_grpo.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir outputs/models/ceo-grpo \
  --epochs 10 \
  --batch-size 4 \
  --num-generations 4 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-6 \
  --save-steps 50 \
  --logging-steps 10 \
  --max-steps 2000 \
  --report-to tensorboard \
  --resume-from-checkpoint outputs/models/ceo-grpo/checkpoint-500
```

## 6. Evaluate After Training

After the adapter is saved in `outputs/models/ceo-grpo`, run:

```bash
python evaluation.py --episodes 20 --horizon 30 --agent-mode trained_ceo --save-dir outputs/eval_trained
```

Compare `outputs/eval_baseline/evaluation_summary.json` with
`outputs/eval_trained/evaluation_summary.json`.

For a fuller comparison report:

```bash
python compare_policies.py --output-dir outputs/comparison
```
