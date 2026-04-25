import argparse
import json
import os
from pathlib import Path

from agents import ActionProposal, build_heuristic_agents
from environment import StartupEnvironment
from llm_agents import build_prompted_agents, build_trained_ceo_agents


def run_episode(
    env: StartupEnvironment,
    horizon: int = 30,
    verbose: bool = True,
    show_hidden_state: bool = False,
    agent_mode: str = "heuristic",
    log_detail: str = "compact",
) -> dict:
    observation = env.reset()
    tech, growth, finance, ceo = _build_agent_stack(agent_mode)

    total_reward = 0.0
    episode_log = []

    for _ in range(horizon):
        proposals = {
            tech.name: tech.propose(observation),
            growth.name: growth.propose(observation),
            finance.name: finance.propose(observation),
        }

        selected = ceo.choose_action(proposals, observation)
        if (
            (observation.get("crisis_level") == "crisis" or observation.get("runway_hint", 999) < 2)
            and selected.action in StartupEnvironment.CRISIS_DISALLOWED_ACTIONS
        ):
            selected = ActionProposal(
                action="fire_employee",
                reasoning=(
                    f"Safety override: {selected.action} is disallowed in crisis, so the CEO selected fire_employee."
                ),
            )
        result = env.step(selected.action, proposals=proposals)
        total_reward += result["reward"]

        log_row = {
            "day": observation["day"],
            "observation": dict(observation),
            "proposals": {
                name: {
                    "action": proposal.action,
                    "reasoning": proposal.reasoning,
                }
                for name, proposal in proposals.items()
            },
            "chosen_action": selected.action,
            "chosen_reasoning": selected.reasoning,
            "reward": result["reward"],
            "raw_reward": result.get("raw_reward"),
            "reward_details": result.get("reward_details"),
            "event": result["event"],
            "termination_reason": result.get("termination_reason"),
            "money": result["state"]["money"],
            "users": result["state"]["users"],
            "quality": result["state"]["product_quality"],
        }
        prompt_debug = _collect_prompt_debug(tech, growth, finance, ceo)
        if prompt_debug:
            log_row["prompt_debug"] = prompt_debug
        episode_log.append(log_row)

        if verbose:
            if log_detail == "full":
                _print_full_step(observation, proposals, selected, result)
            else:
                _print_compact_step(observation, proposals, selected, result)
            if show_hidden_state:
                print(f"  debug_hidden={result['debug_state']['hidden_state']}")
                print(f"  pending_effects={result['debug_state']['pending_effects']}")
            print("---")

        observation = result["state"]
        if result["done"]:
            break

    summary = {
        "total_reward": round(total_reward, 3),
        "days_completed": len(episode_log),
        "termination_reason": episode_log[-1]["termination_reason"] if episode_log else "not_started",
        "final_state": env.get_debug_state()["public_state"],
        "episode_log": episode_log,
    }

    if verbose:
        print(
            "Episode finished"
            f" day={summary['days_completed']},"
            f" total_reward={summary['total_reward']},"
            f" termination_reason={summary['termination_reason']},"
            f" final_money={summary['final_state']['money']},"
            f" final_users={summary['final_state']['users']}"
        )

    return summary


def _print_compact_step(
    observation: dict,
    proposals: dict,
    selected: ActionProposal,
    result: dict,
) -> None:
    proposal_summary = ", ".join(
        f"{_short_role_name(name)}={proposal.action}"
        for name, proposal in proposals.items()
    )
    print(
        f"Day {observation['day']:>2} | "
        f"CEO={selected.action:<22} | "
        f"reward={result['reward']:>6} | "
        f"money={result['state']['money']:>9} | "
        f"users={result['state']['users']:>4} | "
        f"quality={result['state']['product_quality']}"
    )
    print(
        "  signal: "
        f"growth={observation['recent_user_growth']}, "
        f"last3={observation['last_3_growth']}, "
        f"trend={observation['trend_direction']}, "
        f"runway={observation['runway_hint']}, "
        f"crisis={observation['crisis_level']}, "
        f"event={result['event']}"
    )
    print(f"  proposals: {proposal_summary}")
    print(f"  CEO: {_shorten(selected.reasoning, 220)}")


def _print_full_step(
    observation: dict,
    proposals: dict,
    selected: ActionProposal,
    result: dict,
) -> None:
    print(f"Day {observation['day']}: CEO chose {selected.action}")
    for name, proposal in proposals.items():
        print(f"  {name}: {proposal.action} | {proposal.reasoning}")
    print(
        "  obs:"
        f" growth={observation['recent_user_growth']},"
        f" last_3_growth={observation['last_3_growth']},"
        f" trend={observation['trend_direction']},"
        f" ad={observation['ad_performance']},"
        f" runway_hint={observation['runway_hint']},"
        f" crisis={observation['is_crisis']},"
        f" crisis_level={observation['crisis_level']},"
        f" last_action={observation['last_action']},"
        f" streak={observation['consecutive_action_streak']},"
        f" events={observation['recent_events']}"
    )
    print(f"  crisis_reason={observation['crisis_reason']}")
    print(f"  CEO reasoning: {selected.reasoning}")
    print(
        "  result:"
        f" reward={result['reward']},"
        f" raw_reward={result.get('raw_reward')},"
        f" event={result['event']},"
        f" termination_reason={result.get('termination_reason')},"
        f" money={result['state']['money']},"
        f" users={result['state']['users']},"
        f" quality={result['state']['product_quality']}"
    )
    print(f"  reward_details={result.get('reward_details')}")
    print(f"  agent_rewards={result['agent_rewards']}")


def _short_role_name(name: str) -> str:
    if name.startswith("Tech"):
        return "Tech"
    if name.startswith("Growth"):
        return "Growth"
    if name.startswith("Finance"):
        return "Finance"
    return name


def _shorten(text: object, limit: int = 180) -> str:
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _build_agent_stack(agent_mode: str):
    if agent_mode == "prompt_scaffold":
        return build_prompted_agents()
    if agent_mode == "trained_ceo":
        return build_trained_ceo_agents(
            adapter_path=os.getenv("MASS_CEO_ADAPTER_PATH", "outputs/models/ceo-grpo"),
            base_model=os.getenv("MASS_CEO_BASE_MODEL", "Qwen/Qwen3-0.6B"),
        )
    return build_heuristic_agents()


def _collect_prompt_debug(tech, growth, finance, ceo):
    prompt_debug = {}
    for agent in (tech, growth, finance, ceo):
        prompt = getattr(agent, "last_prompt", None)
        raw_response = getattr(agent, "last_raw_response", None)
        if prompt is None and raw_response is None:
            continue
        prompt_debug[agent.name] = {
            "system_prompt": None if prompt is None else prompt.system_prompt,
            "user_prompt": None if prompt is None else prompt.user_prompt,
            "raw_response": raw_response,
        }
    return prompt_debug


def main() -> None:
    parser = argparse.ArgumentParser(description="Run startup simulator episodes.")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--show-hidden-state", action="store_true")
    parser.add_argument("--agent-mode", choices=["heuristic", "prompt_scaffold", "trained_ceo"], default="heuristic")
    parser.add_argument("--log-detail", choices=["compact", "full"], default="compact")
    parser.add_argument("--save-summary", type=str, default=None)
    args = parser.parse_args()

    env = StartupEnvironment(max_days=args.horizon, seed=args.seed)
    summary = run_episode(
        env,
        horizon=args.horizon,
        verbose=not args.quiet,
        show_hidden_state=args.show_hidden_state,
        agent_mode=args.agent_mode,
        log_detail=args.log_detail,
    )

    if args.save_summary:
        path = Path(args.save_summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary to {path}")


if __name__ == "__main__":
    main()
