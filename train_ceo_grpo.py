import argparse
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from environment import StartupEnvironment
from llm_agents import parse_action


VALID_ACTIONS = tuple(StartupEnvironment.ACTIONS)
CRISIS_DISALLOWED = StartupEnvironment.CRISIS_DISALLOWED_ACTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the MASS CEO decision model with TRL GRPO."
    )
    parser.add_argument("--dataset", default="outputs/ceo_grpo.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", default="outputs/models/ceo-grpo")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-completion-length", type=int, default=24)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
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
            f"Dataset not found: {dataset_path}. Run train.py with --grpo-output first."
        )

    train_dataset = load_dataset(
        "json",
        data_files=str(dataset_path),
        split="train",
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
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

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[
            action_format_reward,
            simulator_proxy_reward,
            proposal_alignment_reward,
        ],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved CEO GRPO model to {args.output_dir}")


def action_format_reward(
    completions: List[Any],
    log_extra=None,
    log_metric=None,
    **_: Any,
) -> List[float]:
    rewards: List[float] = []
    actions: List[str] = []
    for completion in completions:
        text = _completion_text(completion)
        action = parse_action(text, VALID_ACTIONS)
        actions.append(action or "invalid")

        if action is None:
            rewards.append(-1.0)
            continue

        exact_one_line = re.fullmatch(r"\s*Action:\s*[a-z_]+\s*", text) is not None
        rewards.append(0.6 if exact_one_line else 0.25)

    _log_batch(log_extra, log_metric, actions, "parsed_action", "format_valid_rate")
    return rewards


def simulator_proxy_reward(
    completions: List[Any],
    reference_action: List[str],
    reference_reward: List[float],
    money: List[float],
    product_quality: List[float],
    burn_rate: List[float],
    recent_user_growth: List[float],
    last_3_growth: List[List[float]],
    trend_direction: List[str],
    ad_performance: List[str],
    runway_hint: List[float],
    crisis_level: List[str],
    recent_actions: List[List[str]],
    log_extra=None,
    log_metric=None,
    **_: Any,
) -> List[float]:
    rewards: List[float] = []
    actions: List[str] = []

    for index, completion in enumerate(completions):
        text = _completion_text(completion)
        action = parse_action(text, VALID_ACTIONS)
        actions.append(action or "invalid")
        if action is None:
            rewards.append(-2.0)
            continue

        reward = 0.0
        reward += float(reference_reward[index]) if action == reference_action[index] else -0.15
        reward += _state_action_bonus(
            action=action,
            money=float(money[index]),
            product_quality=float(product_quality[index]),
            burn_rate=float(burn_rate[index]),
            recent_user_growth=float(recent_user_growth[index]),
            last_3_growth=[float(value) for value in last_3_growth[index]],
            trend_direction=str(trend_direction[index]),
            ad_performance=str(ad_performance[index]),
            runway_hint=float(runway_hint[index]),
            crisis_level=str(crisis_level[index]),
            recent_actions=list(recent_actions[index]),
        )
        rewards.append(round(reward, 4))

    _log_batch(log_extra, log_metric, actions, "proxy_action", "proxy_valid_rate")
    return rewards


def proposal_alignment_reward(
    completions: List[Any],
    proposal_actions: List[Dict[str, str]],
    crisis_level: List[str],
    runway_hint: List[float],
    log_metric=None,
    **_: Any,
) -> List[float]:
    rewards: List[float] = []
    aligned = 0

    for index, completion in enumerate(completions):
        action = parse_action(_completion_text(completion), VALID_ACTIONS)
        if action is None:
            rewards.append(-0.6)
            continue

        proposals = proposal_actions[index] or {}
        vote_counts = Counter(proposals.values())
        reward = 0.0
        if action in vote_counts:
            aligned += 1
            reward += 0.15 + 0.15 * vote_counts[action]
        else:
            reward -= 0.25

        if (
            crisis_level[index] == "crisis" or float(runway_hint[index]) < 2.0
        ) and action in CRISIS_DISALLOWED:
            reward -= 1.0

        rewards.append(round(reward, 4))

    if log_metric and rewards:
        log_metric("proposal_alignment_rate", aligned / len(rewards))
    return rewards


def _state_action_bonus(
    action: str,
    money: float,
    product_quality: float,
    burn_rate: float,
    recent_user_growth: float,
    last_3_growth: List[float],
    trend_direction: str,
    ad_performance: str,
    runway_hint: float,
    crisis_level: str,
    recent_actions: List[str],
) -> float:
    reward = 0.0
    cash_stress = crisis_level == "crisis" or runway_hint < 2.0 or money < burn_rate * 2.0
    repeated = recent_actions[-2:].count(action) >= 2
    average_recent_growth = mean(last_3_growth) if last_3_growth else recent_user_growth

    if cash_stress:
        if action == "fire_employee":
            reward += 0.9
        if action in CRISIS_DISALLOWED:
            reward -= 1.2
        if action == "do_nothing":
            reward -= 0.35

    if product_quality < 0.62 and action == "invest_in_product":
        reward += 0.45
    if product_quality < 0.55 and action == "run_marketing_campaign":
        reward -= 0.35

    if (
        action == "run_marketing_campaign"
        and not cash_stress
        and product_quality >= 0.62
        and (trend_direction == "improving" or ad_performance == "good" or average_recent_growth > 8)
    ):
        reward += 0.45

    if trend_direction == "declining":
        if action in {"invest_in_product", "pivot_strategy"}:
            reward += 0.25
        if action == "do_nothing":
            reward -= 0.3

    if action == "hire_employee":
        if runway_hint > 6.0 and average_recent_growth > 12 and product_quality > 0.68:
            reward += 0.35
        else:
            reward -= 0.4

    if repeated:
        reward -= 0.45 if action != "fire_employee" else 0.2
    return reward


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def _log_batch(
    log_extra,
    log_metric,
    actions: Iterable[str],
    column_name: str,
    metric_name: str,
) -> None:
    actions = list(actions)
    if log_extra:
        log_extra(column_name, actions)
    if log_metric and actions:
        valid_count = sum(action != "invalid" for action in actions)
        log_metric(metric_name, valid_count / len(actions))


if __name__ == "__main__":
    main()
