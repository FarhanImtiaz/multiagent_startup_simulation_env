import argparse
import json
from pathlib import Path
from typing import Dict, List

from environment import StartupEnvironment
from simulate import run_episode


def collect_trajectories(
    episodes: int = 20,
    horizon: int = 30,
    base_seed: int = 7,
    agent_mode: str = "heuristic",
) -> List[Dict[str, object]]:
    trajectories: List[Dict[str, object]] = []
    for episode_index in range(episodes):
        env = StartupEnvironment(max_days=horizon, seed=base_seed + episode_index)
        summary = run_episode(
            env,
            horizon=horizon,
            verbose=False,
            show_hidden_state=False,
            agent_mode=agent_mode,
        )

        trajectories.append(
            {
                "episode_index": episode_index,
                "agent_mode": agent_mode,
                "total_reward": summary["total_reward"],
                "days_completed": summary["days_completed"],
                "termination_reason": summary["termination_reason"],
                "final_state": summary["final_state"],
                "steps": summary["episode_log"],
            }
        )

    return trajectories


def save_trajectories(path: str, trajectories: List[Dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trajectories, indent=2), encoding="utf-8")


def save_jsonl(path: str, records: List[Dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_sft_records(
    trajectories: List[Dict[str, object]],
    min_step_reward: float | None = None,
    survivors_only: bool = False,
    min_final_money: float | None = None,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for trajectory in trajectories:
        if not _trajectory_matches_filters(
            trajectory,
            survivors_only=survivors_only,
            min_final_money=min_final_money,
        ):
            continue
        for step in trajectory["steps"]:
            if min_step_reward is not None and step["reward"] < min_step_reward:
                continue
            records.append(
                {
                    "episode_index": trajectory["episode_index"],
                    "day": step["day"],
                    "reward": step["reward"],
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are the CEO in a startup simulator. Choose one valid action "
                                "from co-founder proposals while balancing survival, recovery, and growth."
                            ),
                        },
                        {
                            "role": "user",
                            "content": _format_training_prompt(step),
                        },
                        {
                            "role": "assistant",
                            "content": f"Action: {step['chosen_action']}",
                        },
                    ],
                }
            )
    return records


def build_preference_records(
    trajectories: List[Dict[str, object]],
    min_step_reward: float | None = 0.0,
    survivors_only: bool = False,
    min_final_money: float | None = None,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for trajectory in trajectories:
        if not _trajectory_matches_filters(
            trajectory,
            survivors_only=survivors_only,
            min_final_money=min_final_money,
        ):
            continue
        for step in trajectory["steps"]:
            if min_step_reward is not None and step["reward"] < min_step_reward:
                continue

            rejected_actions = sorted(
                {
                    proposal["action"]
                    for proposal in step.get("proposals", {}).values()
                    if proposal["action"] != step["chosen_action"]
                }
            )
            if not rejected_actions:
                continue

            records.append(
                {
                    "episode_index": trajectory["episode_index"],
                    "day": step["day"],
                    "reward": step["reward"],
                    "prompt": _format_training_prompt(step),
                    "chosen": f"Action: {step['chosen_action']}",
                    "rejected": f"Action: {rejected_actions[0]}",
                }
            )
    return records


def build_grpo_records(
    trajectories: List[Dict[str, object]],
    min_reference_reward: float | None = None,
    survivors_only: bool = False,
    min_final_money: float | None = None,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for trajectory in trajectories:
        if not _trajectory_matches_filters(
            trajectory,
            survivors_only=survivors_only,
            min_final_money=min_final_money,
        ):
            continue
        for step in trajectory["steps"]:
            if min_reference_reward is not None and step["reward"] < min_reference_reward:
                continue

            observation = step["observation"]
            proposals = {
                name: proposal["action"]
                for name, proposal in step.get("proposals", {}).items()
            }
            records.append(
                {
                    "episode_index": trajectory["episode_index"],
                    "day": step["day"],
                    "prompt": [
                        {
                            "role": "system",
                            "content": (
                                "You are the CEO in a startup simulator. Choose one valid action "
                                "from co-founder proposals while balancing survival, recovery, and growth."
                            ),
                        },
                        {
                            "role": "user",
                            "content": _format_training_prompt(step),
                        },
                    ],
                    "reference_action": step["chosen_action"],
                    "reference_reward": step["reward"],
                    "proposal_actions": proposals,
                    "money": observation["money"],
                    "users": observation["users"],
                    "product_quality": observation["product_quality"],
                    "burn_rate": observation["burn_rate"],
                    "recent_user_growth": observation["recent_user_growth"],
                    "last_3_growth": observation["last_3_growth"],
                    "trend_direction": observation["trend_direction"],
                    "ad_performance": observation["ad_performance"],
                    "runway_hint": observation["runway_hint"],
                    "crisis_level": observation["crisis_level"],
                    "recent_actions": observation["recent_actions"],
                }
            )
    return records


def _trajectory_matches_filters(
    trajectory: Dict[str, object],
    survivors_only: bool = False,
    min_final_money: float | None = None,
) -> bool:
    if survivors_only and trajectory.get("termination_reason") != "max_days":
        return False
    if min_final_money is not None:
        final_state = trajectory.get("final_state", {})
        if float(final_state.get("money", 0.0)) < min_final_money:
            return False
    return True


def _format_training_prompt(step: Dict[str, object]) -> str:
    observation = step["observation"]
    proposal_lines = [
        f"- {name}: {proposal['action']} | {proposal['reasoning']}"
        for name, proposal in step.get("proposals", {}).items()
    ]
    return (
        "Observed startup state:\n"
        f"- Day: {observation['day']}\n"
        f"- Money: {observation['money']}\n"
        f"- Users: {observation['users']}\n"
        f"- Product quality: {observation['product_quality']}\n"
        f"- Team size: {observation['team_size']}\n"
        f"- Burn rate: {observation['burn_rate']}\n"
        f"- Recent user growth: {observation['recent_user_growth']}\n"
        f"- Last 3 growth: {observation['last_3_growth']}\n"
        f"- Trend direction: {observation['trend_direction']}\n"
        f"- Ad performance: {observation['ad_performance']}\n"
        f"- Runway hint: {observation['runway_hint']}\n"
        f"- Crisis level: {observation['crisis_level']}\n"
        f"- Crisis reason: {observation['crisis_reason']}\n"
        f"- Recent events: {observation['recent_events']}\n"
        f"- Recent actions: {observation['recent_actions']}\n\n"
        "Co-founder proposals:\n"
        f"{chr(10).join(proposal_lines)}\n\n"
        "Respond with exactly one line: Action: <action_name>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect startup simulator trajectories and training-ready decision datasets."
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--agent-mode", choices=["heuristic", "prompt_scaffold", "trained_ceo"], default="heuristic")
    parser.add_argument("--output", default="outputs/trajectories.json")
    parser.add_argument("--sft-output", default=None)
    parser.add_argument("--preference-output", default=None)
    parser.add_argument("--grpo-output", default=None)
    parser.add_argument(
        "--min-sft-reward",
        type=float,
        default=None,
        help="Only include SFT steps with reward at or above this value. Defaults to all steps.",
    )
    parser.add_argument(
        "--min-preference-reward",
        type=float,
        default=0.0,
        help="Only include preference steps with reward at or above this value.",
    )
    parser.add_argument(
        "--min-grpo-reference-reward",
        type=float,
        default=None,
        help="Only include GRPO prompts whose historical chosen action reached this reward.",
    )
    parser.add_argument(
        "--survivors-only",
        action="store_true",
        help="Only export examples from episodes that survive to the max horizon.",
    )
    parser.add_argument(
        "--min-final-money",
        type=float,
        default=None,
        help="Only export examples from episodes ending with at least this much money.",
    )
    args = parser.parse_args()

    trajectories = collect_trajectories(
        episodes=args.episodes,
        horizon=args.horizon,
        base_seed=args.seed,
        agent_mode=args.agent_mode,
    )
    save_trajectories(args.output, trajectories)

    print(
        "Collected trajectories"
        f" episodes={len(trajectories)}"
        f" output={args.output}"
    )
    if args.sft_output:
        sft_records = build_sft_records(
            trajectories,
            min_step_reward=args.min_sft_reward,
            survivors_only=args.survivors_only,
            min_final_money=args.min_final_money,
        )
        save_jsonl(args.sft_output, sft_records)
        print(f"Saved SFT records count={len(sft_records)} output={args.sft_output}")

    if args.preference_output:
        preference_records = build_preference_records(
            trajectories,
            min_step_reward=args.min_preference_reward,
            survivors_only=args.survivors_only,
            min_final_money=args.min_final_money,
        )
        save_jsonl(args.preference_output, preference_records)
        print(
            "Saved preference records"
            f" count={len(preference_records)}"
            f" output={args.preference_output}"
        )

    if args.grpo_output:
        grpo_records = build_grpo_records(
            trajectories,
            min_reference_reward=args.min_grpo_reference_reward,
            survivors_only=args.survivors_only,
            min_final_money=args.min_final_money,
        )
        save_jsonl(args.grpo_output, grpo_records)
        print(f"Saved GRPO prompts count={len(grpo_records)} output={args.grpo_output}")

    print("Optimizer/fine-tuning execution is external; this script prepares the data for it.")


if __name__ == "__main__":
    main()
