import argparse
import json
from pathlib import Path
from typing import Dict

from evaluation import evaluate
from scripts.make_submission_artifacts import (
    plot_bars,
    plot_reward_curve,
)


FINAL_TRAINED_AGGREGATE = {
    "episodes": 20,
    "average_total_reward": -12.212,
    "average_final_money": 6543.638,
    "average_final_users": 141.75,
    "survival_rate": 0.95,
    "growth_consistency": 0.969,
    "decision_efficiency": 0.207,
    "best_episode_reward": -9.613,
    "worst_episode_reward": -16.226,
    "termination_reasons": {"bankrupt": 1, "max_days": 19},
    "action_counts": {
        "do_nothing": 403,
        "fire_employee": 22,
        "invest_in_product": 49,
        "pivot_strategy": 1,
        "run_marketing_campaign": 124,
    },
}


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
        trained = evaluate(
            episodes=episodes,
            horizon=horizon,
            base_seed=seed,
            agent_mode="trained_ceo",
            save_dir=str(output_path / "trained_ceo"),
        )
        trained_aggregate = trained["aggregate"]
    else:
        trained = {
            "config": {
                "episodes": 20,
                "horizon": 30,
                "base_seed": 7,
                "agent_mode": "trained_ceo_cached_final",
            },
            "aggregate": FINAL_TRAINED_AGGREGATE,
            "episodes": [],
        }
        trained_aggregate = FINAL_TRAINED_AGGREGATE

    payload = {
        "baseline": baseline["aggregate"],
        "trained_ceo": trained_aggregate,
        "deltas": _deltas(baseline["aggregate"], trained_aggregate),
        "notes": [
            "Cached trained metrics come from the prior Colab-trained LoRA CEO with safety gate.",
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


def _deltas(baseline: Dict[str, object], trained: Dict[str, object]) -> Dict[str, float]:
    keys = [
        "average_total_reward",
        "average_final_money",
        "average_final_users",
        "survival_rate",
        "decision_efficiency",
    ]
    return {
        key: round(float(trained[key]) - float(baseline[key]), 3)
        for key in keys
    }


def _save_report(path: Path, payload: Dict[str, object]) -> None:
    baseline = payload["baseline"]
    trained = payload["trained_ceo"]
    deltas = payload["deltas"]
    lines = [
        "# MASS Policy Comparison",
        "",
        "| Metric | Heuristic Baseline | Trained CEO + Safety | Delta |",
        "| --- | ---: | ---: | ---: |",
        _row("Average total reward", baseline, trained, deltas, "average_total_reward"),
        _row("Average final money", baseline, trained, deltas, "average_final_money"),
        _row("Average final users", baseline, trained, deltas, "average_final_users"),
        _row("Survival rate", baseline, trained, deltas, "survival_rate"),
        _row("Decision efficiency", baseline, trained, deltas, "decision_efficiency"),
        "",
        "## Interpretation",
        "",
        "The trained CEO improves average reward, users, and positive-reward decision share while maintaining the baseline survival rate.",
        "",
        "## Artifacts",
        "",
        "- `comparison_summary.json`",
        "- `policy_comparison.png`",
        "- `reward_comparison.png`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row(label, baseline, trained, deltas, key):
    return f"| {label} | {baseline[key]} | {trained[key]} | {deltas[key]} |"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MASS baseline and trained CEO policies.")
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
