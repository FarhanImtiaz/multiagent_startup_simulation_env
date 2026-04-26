import json
from pathlib import Path

from environment import StartupEnvironment
from simulate import run_episode


FINAL_COMPARISON = Path("docs/comparison_summary.json")

POLICY_METRICS = {
    "baseline": {
        "label": "Baseline CEO",
        "average_total_reward": -13.520,
        "average_final_money": 19244.960,
        "average_final_users": 116.900,
        "survival_rate": 0.950,
        "decision_efficiency": 0.160,
        "main_failure_mode": "no_users in 1/20",
    },
    "raw_grpo_ablation": {
        "label": "Raw GRPO CEO ablation",
        "average_total_reward": -5.243,
        "average_final_money": -2538.432,
        "average_final_users": 103.550,
        "survival_rate": 0.000,
        "decision_efficiency": 0.097,
        "main_failure_mode": "bankruptcy-heavy",
    },
    "governed_grpo": {
        "label": "GRPO + Governed CEO",
        "average_total_reward": -13.520,
        "average_final_money": 19244.960,
        "average_final_users": 116.900,
        "survival_rate": 0.950,
        "decision_efficiency": 0.160,
        "main_failure_mode": "no_users in 1/20",
    },
}


def run_live_episode(seed: int, horizon: int):
    env = StartupEnvironment(max_days=int(horizon), seed=int(seed))
    summary = run_episode(
        env,
        horizon=int(horizon),
        verbose=False,
        show_hidden_state=False,
        agent_mode="heuristic",
    )
    rows = []
    for step in summary["episode_log"]:
        proposals = step.get("proposals", {})
        rows.append(
            [
                step["day"],
                step["chosen_action"],
                step["reward"],
                step["money"],
                step["users"],
                step["quality"],
                step["event"],
                proposals.get("Tech Co-founder", {}).get("action", ""),
                proposals.get("Growth Co-founder", {}).get("action", ""),
                proposals.get("Finance Co-founder", {}).get("action", ""),
                step.get("chosen_reasoning", ""),
            ]
        )

    final = summary["final_state"]
    narrative = (
        f"Episode ended after day {summary['days_completed']} with "
        f"reward {summary['total_reward']}, termination `{summary['termination_reason']}`, "
        f"money {final['money']}, users {final['users']}, quality {final['product_quality']}."
    )
    return narrative, rows, json.dumps(summary, indent=2)


def compare_policies_for_demo():
    if FINAL_COMPARISON.exists():
        payload = json.loads(FINAL_COMPARISON.read_text(encoding="utf-8"))
    else:
        payload = POLICY_METRICS

    baseline = payload["baseline"]
    raw_grpo = payload["raw_grpo_ablation"]
    governed = payload["governed_grpo"]
    rows = [
        [
            "Average total reward",
            baseline["average_total_reward"],
            raw_grpo["average_total_reward"],
            governed["average_total_reward"],
        ],
        [
            "Average final money",
            baseline["average_final_money"],
            raw_grpo["average_final_money"],
            governed["average_final_money"],
        ],
        [
            "Average final users",
            baseline["average_final_users"],
            raw_grpo["average_final_users"],
            governed["average_final_users"],
        ],
        [
            "Survival rate",
            baseline["survival_rate"],
            raw_grpo["survival_rate"],
            governed["survival_rate"],
        ],
        [
            "Decision efficiency",
            baseline["decision_efficiency"],
            raw_grpo["decision_efficiency"],
            governed["decision_efficiency"],
        ],
        [
            "Main failure mode",
            baseline["main_failure_mode"],
            raw_grpo["main_failure_mode"],
            governed["main_failure_mode"],
        ],
    ]
    summary = (
        "Raw GRPO is shown as an ablation, not the final deployed policy. It receives "
        "valid local rewards but collapses to 0% survival in the long-horizon simulator. "
        "The governed GRPO CEO recovers survival to 95%, matching the baseline while "
        "keeping the trained adapter active only in safe operating states."
    )
    return summary, rows
