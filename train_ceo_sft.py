import argparse
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a CEO decision model from MASS SFT JSONL records."
    )
    parser.add_argument("--dataset", default="outputs/ceo_sft.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", default="outputs/models/ceo-sft")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--no-lora", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Run train.py with --sft-output first."
        )

    train_dataset = load_dataset(
        "json",
        data_files=str(dataset_path),
        split="train",
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        report_to="none",
    )

    peft_config = None
    if not args.no_lora:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    trainer = SFTTrainer(
        model=args.model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved CEO SFT model to {args.output_dir}")


if __name__ == "__main__":
    main()

