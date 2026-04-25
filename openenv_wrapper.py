from typing import Dict, Optional

from environment import StartupEnvironment


class OpenEnvStartupWrapper:
    """
    Minimal training-friendly wrapper around StartupEnvironment.

    This is not tied to a specific package, but it exposes the conventional
    reset/step interface expected by downstream evaluation and training code.
    """

    def __init__(self, max_days: int = 30, seed: Optional[int] = None):
        self.env = StartupEnvironment(max_days=max_days, seed=seed)

    @property
    def action_space(self):
        return tuple(self.env.ACTIONS)

    def reset(self) -> Dict[str, object]:
        return self.env.reset()

    def step(self, action: str) -> Dict[str, object]:
        result = self.env.step(action)
        return {
            "observation": result["state"],
            "reward": result["reward"],
            "done": result["done"],
            "info": {
                "action": result["action"],
                "event": result["event"],
                "agent_rewards": result["agent_rewards"],
                "debug_state": result["debug_state"],
            },
        }

    def render(self) -> Dict[str, object]:
        return self.env.get_debug_state()

    def observation_schema(self) -> Dict[str, str]:
        return {
            "day": "int",
            "money": "float",
            "users": "int",
            "product_quality": "float",
            "team_size": "int",
            "burn_rate": "float",
            "recent_user_growth": "float",
            "ad_performance": "str",
            "recent_actions": "list[str]",
            "recent_events": "list[str]",
            "runway_hint": "float",
        }
