from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from environment import StartupEnvironment
from mass_startup_env.models import StartupAction, StartupObservation, StartupState

try:
    from openenv.core.env_server import Environment
except ImportError:
    class Environment:
        pass


class StartupOpenEnv(Environment):
    """OpenEnv-compatible wrapper for the MASS startup simulator."""

    def __init__(self, max_days: int = 30, seed: Optional[int] = None):
        self.max_days = max_days
        self.seed = seed
        self.env = StartupEnvironment(max_days=max_days, seed=seed)
        self._state = StartupState(
            episode_id=str(uuid4()),
            step_count=0,
            max_days=max_days,
            seed=seed,
            public_state={},
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> StartupObservation:
        if seed is not None:
            self.seed = seed
            self.env = StartupEnvironment(max_days=self.max_days, seed=seed)

        observation = self.env.reset()
        self._state = StartupState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            max_days=self.max_days,
            seed=self.seed,
            public_state=self.env.get_debug_state()["public_state"],
        )
        return self._to_observation(observation, reward=0.0, done=False, metadata={"status": "reset"})

    def step(
        self,
        action: StartupAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> StartupObservation:
        action_name = action.action
        if action_name not in StartupEnvironment.ACTIONS:
            return StartupObservation(
                done=False,
                reward=-1.0,
                metadata={
                    "error": f"Invalid action: {action_name}",
                    "valid_actions": list(StartupEnvironment.ACTIONS),
                },
            )

        result = self.env.step(action_name)
        self._state.step_count += 1
        self._state.public_state = self.env.get_debug_state()["public_state"]
        return self._to_observation(
            result["state"],
            reward=result["reward"],
            done=result["done"],
            metadata={
                "action": result["action"],
                "event": result["event"],
                "raw_reward": result.get("raw_reward"),
                "termination_reason": result.get("termination_reason"),
                "agent_rewards": result.get("agent_rewards", {}),
            },
            event=result.get("event"),
            reward_details=result.get("reward_details", {}),
        )

    @property
    def state(self) -> StartupState:
        return self._state

    @staticmethod
    def _to_observation(
        observation: dict,
        reward: float,
        done: bool,
        metadata: dict,
        event: Optional[str] = None,
        reward_details: Optional[dict] = None,
    ) -> StartupObservation:
        return StartupObservation(
            done=done,
            reward=reward,
            metadata=metadata,
            day=observation["day"],
            money=observation["money"],
            users=observation["users"],
            product_quality=observation["product_quality"],
            team_size=observation["team_size"],
            burn_rate=observation["burn_rate"],
            recent_user_growth=observation["recent_user_growth"],
            last_3_growth=observation["last_3_growth"],
            trend_direction=observation["trend_direction"],
            ad_performance=observation["ad_performance"],
            runway_hint=observation["runway_hint"],
            crisis_level=observation["crisis_level"],
            crisis_reason=observation["crisis_reason"],
            recent_events=observation["recent_events"],
            recent_actions=observation["recent_actions"],
            event=event,
            reward_details=reward_details or {},
        )

