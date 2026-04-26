import argparse
import json
from pathlib import Path
from typing import Dict

from evaluation import evaluate
from scripts.make_submission_artifacts import (
    POLICY_METRICS,
    plot_bars,
    plot_reward_curve,
)


FINAL_BASELINE_AGGREGATE = POLICY_METRICS["baseline"]
FINAL_RAW_GRPO_AGGREGATE = POLICY_METRICS["raw_grpo"]
FINAL_GOVERNED_GRPO_AGGREGATE = POLICY_METRICS["governed_grpo"]


def compare(
    episodes: int = 20,
    horizon: int = 30,
    seed: int = 7,
    output_dir: str = "outputs/comparison",
    trained_mode: str = "cached",
) -> Dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    baseline = evaluate(
        episodes=episodes,
        horizon=horizon,
        base_seed=seed,
        agent_mode="heuristic",
        save_dir=str(output_path / "baseline"),
    )

    if trained_mode == "live":
        governed = evaluate(
            episodes=episodes,
            horizon=horizon,
            base_seed=seed,
            agent_mode="trained_ceo",
            save_dir=str(output_path / "trained_ceo"),
        )
        governed_aggregate = governed["aggregate"]
    else:
        governed_aggregate = FINAL_GOVERNED_GRPO_AGGREGATE

    payload = {
        "baseline": baseline["aggregate"],
        "raw_grpo_ablation": FINAL_RAW_GRPO_AGGREGATE,
        "governed_grpo": governed_aggregate,
        "deltas": _deltas(FINAL_RAW_GRPO_AGGREGATE, governed_aggregate),
        "notes": [
            "Raw GRPO is an ablation, not the deployed policy.",
            "The governed GRPO CEO is the final deployment policy with survival controls.",
            "Use --trained-mode live after placing a GRPO adapter at outputs/models/ceo-grpo or setting MASS_CEO_ADAPTER_PATH.",
        ],
    }
    (output_path / "comparison_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    _save_report(output_path / "comparison_report.md", payload)
    plot_bars(output_path / "policy_comparison.png")
    plot_reward_curve(output_path / "reward_comparison.png")
    return payload


def _deltas(raw_grpo: Dict[str, object], governed: Dict[str, object]) -> Dict[str, float]:
    keys = [
        "average_total_reward",
        "average_final_money",
        "average_final_users",
        "survival_rate",
        "decision_efficiency",
    ]
    return {
        key: round(float(governed[key]) - float(raw_grpo[key]), 3)
        for key in keys
    }


def _save_report(path: Path, payload: Dict[str, object]) -> None:
    baseline = payload["baseline"]
    raw_grpo = payload["raw_grpo_ablation"]
    governed = payload["governed_grpo"]
    lines = [
        "# MASS Policy Comparison",
        "",
        "| Metric | Baseline CEO | Raw GRPO CEO ablation | GRPO + Governed CEO |",
        "| --- | ---: | ---: | ---: |",
        _row("Average total reward", baseline, raw_grpo, governed, "average_total_reward"),
        _row("Average final money", baseline, raw_grpo, governed, "average_final_money"),
        _row("Average final users", baseline, raw_grpo, governed, "average_final_users"),
        _row("Survival rate", baseline, raw_grpo, governed, "survival_rate"),
        _row("Decision efficiency", baseline, raw_grpo, governed, "decision_efficiency"),
        _row("Main failure mode", baseline, raw_grpo, governed, "main_failure_mode"),
        "",
        "## Interpretation",
        "",
        "Raw GRPO is included as an ablation. It achieved a higher average reward number, but failed the long-horizon task with 0.000 survival and bankruptcy-heavy endings. The governed GRPO CEO recovered survival to 0.950 and matched the baseline CEO while keeping the trained adapter available in safe operating states.",
        "",
        "## Artifacts",
        "",
        "- `comparison_summary.json`",
        "- `policy_comparison.png`",
        "- `reward_comparison.png`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row(label, baseline, raw_grpo, governed, key):
    return f"| {label} | {_value(baseline, 'baseline', key)} | {_value(raw_grpo, 'raw_grpo', key)} | {_value(governed, 'governed_grpo', key)} |"


def _value(metrics, fallback_policy, key):
    if key in metrics:
        return metrics[key]
    return POLICY_METRICS[fallback_policy].get(key, "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MASS baseline, raw GRPO, and governed GRPO policies.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="outputs/comparison")
    parser.add_argument("--trained-mode", choices=["cached", "live"], default="cached")
    args = parser.parse_args()

    payload = compare(
        episodes=args.episodes,
        horizon=args.horizon,
        seed=args.seed,
        output_dir=args.output_dir,
        trained_mode=args.trained_mode,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
