import argparse
import csv
import json
from pathlib import Path
from collections import Counter
from statistics import mean
from typing import Dict, List, Optional

from simulate import run_episode
from environment import StartupEnvironment


def evaluate(
    episodes: int = 10,
    horizon: int = 30,
    base_seed: int = 7,
    agent_mode: str = "heuristic",
    save_dir: Optional[str] = "outputs",
) -> Dict[str, object]:
    episode_summaries: List[Dict[str, object]] = []

    for episode_index in range(episodes):
        env = StartupEnvironment(max_days=horizon, seed=base_seed + episode_index)
        summary = run_episode(
            env,
            horizon=horizon,
            verbose=False,
            show_hidden_state=False,
            agent_mode=agent_mode,
        )
        summary["episode_index"] = episode_index
        episode_summaries.append(summary)

    aggregate = _build_aggregate_metrics(episode_summaries)
    payload = {
        "config": {
            "episodes": episodes,
            "horizon": horizon,
            "base_seed": base_seed,
            "agent_mode": agent_mode,
        },
        "aggregate": aggregate,
        "episodes": episode_summaries,
    }

    if save_dir is not None:
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_json(output_dir / "evaluation_summary.json", payload)
        _save_episode_csv(output_dir / "episode_metrics.csv", episode_summaries)
        _save_step_csv(output_dir / "step_metrics.csv", episode_summaries)
        _save_action_distribution_csv(output_dir / "action_distribution.csv", episode_summaries)
        _save_report(output_dir / "baseline_report.md", payload)
        _save_plots(output_dir, episode_summaries)

    return payload


def _build_aggregate_metrics(episode_summaries: List[Dict[str, object]]) -> Dict[str, object]:
    if not episode_summaries:
        return {
            "episodes": 0,
            "average_total_reward": 0.0,
            "average_final_money": 0.0,
            "average_final_users": 0.0,
            "survival_rate": 0.0,
            "growth_consistency": 0.0,
            "decision_efficiency": 0.0,
            "best_episode_reward": 0.0,
            "worst_episode_reward": 0.0,
            "termination_reasons": {},
            "action_counts": {},
        }

    total_rewards = [summary["total_reward"] for summary in episode_summaries]
    final_money = [summary["final_state"]["money"] for summary in episode_summaries]
    final_users = [summary["final_state"]["users"] for summary in episode_summaries]
    survival_flags = [summary["final_state"]["money"] >= 0 and summary["final_state"]["users"] > 0 for summary in episode_summaries]
    decision_efficiency = [
        _positive_reward_ratio(summary["episode_log"])
        for summary in episode_summaries
    ]
    growth_consistency = [
        _growth_consistency(summary["episode_log"])
        for summary in episode_summaries
    ]
    termination_reasons = Counter(summary["termination_reason"] for summary in episode_summaries)
    action_counts = Counter(
        row["chosen_action"]
        for summary in episode_summaries
        for row in summary["episode_log"]
    )

    return {
        "episodes": len(episode_summaries),
        "average_total_reward": round(mean(total_rewards), 3),
        "average_final_money": round(mean(final_money), 3),
        "average_final_users": round(mean(final_users), 3),
        "survival_rate": round(sum(1 for flag in survival_flags if flag) / max(1, len(survival_flags)), 3),
        "growth_consistency": round(mean(growth_consistency), 3),
        "decision_efficiency": round(mean(decision_efficiency), 3),
        "best_episode_reward": round(max(total_rewards), 3),
        "worst_episode_reward": round(min(total_rewards), 3),
        "termination_reasons": dict(sorted(termination_reasons.items())),
        "action_counts": dict(sorted(action_counts.items())),
    }


def _positive_reward_ratio(episode_log: List[Dict[str, object]]) -> float:
    if not episode_log:
        return 0.0
    positive_steps = sum(1 for row in episode_log if row["reward"] > 0)
    return positive_steps / len(episode_log)


def _growth_consistency(episode_log: List[Dict[str, object]]) -> float:
    if len(episode_log) < 2:
        return 0.0
    non_crash_steps = 0
    for previous, current in zip(episode_log, episode_log[1:]):
        if current["users"] >= previous["users"] * 0.85:
            non_crash_steps += 1
    return non_crash_steps / (len(episode_log) - 1)


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _save_episode_csv(path: Path, episode_summaries: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode_index",
                "days_completed",
                "total_reward",
                "final_money",
                "final_users",
                "final_quality",
            ],
        )
        writer.writeheader()
        for summary in episode_summaries:
            writer.writerow(
                {
                    "episode_index": summary["episode_index"],
                    "days_completed": summary["days_completed"],
                    "total_reward": summary["total_reward"],
                    "final_money": summary["final_state"]["money"],
                    "final_users": summary["final_state"]["users"],
                    "final_quality": summary["final_state"]["product_quality"],
                }
            )


def _save_step_csv(path: Path, episode_summaries: List[Dict[str, object]]) -> None:
    fieldnames = [
        "episode_index",
        "day",
        "chosen_action",
        "reward",
        "raw_reward",
        "event",
        "termination_reason",
        "money",
        "users",
        "quality",
        "obs_money",
        "obs_users",
        "obs_quality",
        "obs_growth",
        "obs_last_3_growth",
        "obs_trend",
        "obs_runway_hint",
        "obs_crisis_level",
        "tech_action",
        "growth_action",
        "finance_action",
        "chosen_reasoning",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in episode_summaries:
            for row in summary["episode_log"]:
                observation = row.get("observation", {})
                proposals = row.get("proposals", {})
                writer.writerow(
                    {
                        "episode_index": summary["episode_index"],
                        "day": row.get("day"),
                        "chosen_action": row.get("chosen_action"),
                        "reward": row.get("reward"),
                        "raw_reward": row.get("raw_reward"),
                        "event": row.get("event"),
                        "termination_reason": row.get("termination_reason"),
                        "money": row.get("money"),
                        "users": row.get("users"),
                        "quality": row.get("quality"),
                        "obs_money": observation.get("money"),
                        "obs_users": observation.get("users"),
                        "obs_quality": observation.get("product_quality"),
                        "obs_growth": observation.get("recent_user_growth"),
                        "obs_last_3_growth": json.dumps(observation.get("last_3_growth", [])),
                        "obs_trend": observation.get("trend_direction"),
                        "obs_runway_hint": observation.get("runway_hint"),
                        "obs_crisis_level": observation.get("crisis_level"),
                        "tech_action": _proposal_action(proposals, "Tech Co-founder"),
                        "growth_action": _proposal_action(proposals, "Growth Co-founder"),
                        "finance_action": _proposal_action(proposals, "Finance Co-founder"),
                        "chosen_reasoning": _compact_text(row.get("chosen_reasoning", ""), limit=260),
                    }
                )


def _proposal_action(proposals: Dict[str, object], name: str) -> str:
    proposal = proposals.get(name, {})
    if isinstance(proposal, dict):
        return str(proposal.get("action", ""))
    return ""


def _compact_text(text: object, limit: int = 160) -> str:
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _save_action_distribution_csv(path: Path, episode_summaries: List[Dict[str, object]]) -> None:
    action_counts = Counter(
        row["chosen_action"]
        for summary in episode_summaries
        for row in summary["episode_log"]
    )
    total = sum(action_counts.values())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["action", "count", "share"])
        writer.writeheader()
        for action, count in sorted(action_counts.items()):
            writer.writerow(
                {
                    "action": action,
                    "count": count,
                    "share": round(count / max(1, total), 3),
                }
            )


def _save_report(path: Path, payload: Dict[str, object]) -> None:
    aggregate = payload["aggregate"]
    config = payload["config"]
    lines = [
        "# MASS Baseline Evaluation Report",
        "",
        "## Config",
        "",
        f"- Episodes: {config['episodes']}",
        f"- Horizon: {config['horizon']}",
        f"- Base seed: {config['base_seed']}",
        f"- Agent mode: {config['agent_mode']}",
        "",
        "## Aggregate Metrics",
        "",
        f"- Average total reward: {aggregate['average_total_reward']}",
        f"- Best episode reward: {aggregate['best_episode_reward']}",
        f"- Worst episode reward: {aggregate['worst_episode_reward']}",
        f"- Average final money: {aggregate['average_final_money']}",
        f"- Average final users: {aggregate['average_final_users']}",
        f"- Survival rate: {aggregate['survival_rate']}",
        f"- Growth consistency: {aggregate['growth_consistency']}",
        f"- Decision efficiency: {aggregate['decision_efficiency']}",
        "",
        "## Termination Reasons",
        "",
    ]
    for reason, count in aggregate["termination_reasons"].items():
        lines.append(f"- {reason}: {count}")

    lines.extend(["", "## Action Counts", ""])
    for action, count in aggregate["action_counts"].items():
        lines.append(f"- {action}: {count}")

    lines.extend(
        [
            "",
            "## Generated Artifacts",
            "",
            "- `evaluation_summary.json`",
            "- `episode_metrics.csv`",
            "- `step_metrics.csv`",
            "- `action_distribution.csv`",
            "- `reward_curve.svg`",
            "- `outcome_curve.svg`",
            "- `action_distribution.svg`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_plots(output_dir: Path, episode_summaries: List[Dict[str, object]]) -> None:
    rewards = [float(summary["total_reward"]) for summary in episode_summaries]
    money = [float(summary["final_state"]["money"]) for summary in episode_summaries]
    users = [float(summary["final_state"]["users"]) for summary in episode_summaries]
    action_counts = Counter(
        row["chosen_action"]
        for summary in episode_summaries
        for row in summary["episode_log"]
    )

    _write_line_svg(
        output_dir / "reward_curve.svg",
        title="Total Reward by Episode",
        series=[("reward", rewards)],
    )
    _write_line_svg(
        output_dir / "outcome_curve.svg",
        title="Final Outcomes by Episode",
        series=[("money", money), ("users", users)],
    )
    _write_bar_svg(
        output_dir / "action_distribution.svg",
        title="Chosen Action Distribution",
        values=dict(sorted(action_counts.items())),
    )


def _write_line_svg(path: Path, title: str, series: List[tuple[str, List[float]]]) -> None:
    width = 840
    height = 360
    margin = 48
    all_values = [value for _, values in series for value in values]
    if not all_values:
        all_values = [0.0]
    min_value = min(all_values)
    max_value = max(all_values)
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0

    max_len = max((len(values) for _, values in series), default=1)

    def point(index: int, value: float) -> tuple[float, float]:
        x_span = width - 2 * margin
        y_span = height - 2 * margin
        x = margin + (index / max(1, max_len - 1)) * x_span
        y = height - margin - ((value - min_value) / (max_value - min_value)) * y_span
        return x, y

    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea"]
    lines = [_svg_header(width, height, title)]
    lines.append(f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#94a3b8"/>')
    lines.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#94a3b8"/>')
    lines.append(f'<text x="{margin}" y="{margin - 14}" font-size="12" fill="#475569">max {max_value:.2f}</text>')
    lines.append(f'<text x="{margin}" y="{height - margin + 22}" font-size="12" fill="#475569">min {min_value:.2f}</text>')

    for series_index, (label, values) in enumerate(series):
        if not values:
            continue
        color = colors[series_index % len(colors)]
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(i, value) for i, value in enumerate(values)))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for i, value in enumerate(values):
            x, y = point(i, value)
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
        lines.append(f'<text x="{width - margin - 120}" y="{margin + 20 + series_index * 18}" font-size="13" fill="{color}">{_escape_xml(label)}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_bar_svg(path: Path, title: str, values: Dict[str, int]) -> None:
    width = 840
    height = 360
    margin = 48
    bar_gap = 12
    max_value = max(values.values(), default=1)
    labels = list(values.keys())
    bar_width = (width - 2 * margin - max(0, len(labels) - 1) * bar_gap) / max(1, len(labels))
    lines = [_svg_header(width, height, title)]
    lines.append(f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#94a3b8"/>')

    for index, label in enumerate(labels):
        value = values[label]
        x = margin + index * (bar_width + bar_gap)
        bar_height = (value / max_value) * (height - 2 * margin)
        y = height - margin - bar_height
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#2563eb"/>')
        lines.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" font-size="12" text-anchor="middle" fill="#0f172a">{value}</text>')
        lines.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{height - margin + 18}" font-size="10" text-anchor="middle" fill="#475569">'
            f'{_escape_xml(label.replace("_", " "))}</text>'
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _svg_header(width: int, height: int, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect width="100%" height="100%" fill="#f8fafc"/>'
        f'<text x="24" y="30" font-size="18" font-family="Arial, sans-serif" font-weight="700" fill="#0f172a">{_escape_xml(title)}</text>'
    )


def _escape_xml(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the startup simulator across many episodes.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--agent-mode", choices=["heuristic", "prompt_scaffold", "trained_ceo"], default="heuristic")
    parser.add_argument("--save-dir", default="outputs")
    args = parser.parse_args()

    payload = evaluate(
        episodes=args.episodes,
        horizon=args.horizon,
        base_seed=args.seed,
        agent_mode=args.agent_mode,
        save_dir=args.save_dir,
    )

    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
