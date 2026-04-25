import json
from pathlib import Path

from compare_policies import FINAL_TRAINED_AGGREGATE, compare
from environment import StartupEnvironment
from simulate import run_episode


BASELINE_CACHE = Path("outputs/comparison/comparison_summary.json")


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
    if BASELINE_CACHE.exists():
        payload = json.loads(BASELINE_CACHE.read_text(encoding="utf-8"))
    else:
        payload = compare(output_dir="outputs/comparison", trained_mode="cached")

    baseline = payload["baseline"]
    trained = payload.get("trained_ceo", FINAL_TRAINED_AGGREGATE)
    rows = [
        ["Average total reward", baseline["average_total_reward"], trained["average_total_reward"]],
        ["Average final users", baseline["average_final_users"], trained["average_final_users"]],
        ["Survival rate", baseline["survival_rate"], trained["survival_rate"]],
        ["Decision efficiency", baseline["decision_efficiency"], trained["decision_efficiency"]],
    ]
    summary = (
        "The trained CEO improves reward, users, and decision efficiency while preserving "
        "the baseline 95% survival rate. It spends more aggressively than the heuristic "
        "CEO, so final cash is lower, but the safety gate prevents the high-growth model "
        "from collapsing into all-bankruptcy behavior."
    )
    return summary, rows

