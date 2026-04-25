from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from openenv.core.env_server import Action, Observation, State
except ImportError:
    class _FactoryDefault:
        def __init__(self, factory):
            self.factory = factory

    def Field(default=None, default_factory=None, description=None):
        if default_factory is not None:
            return _FactoryDefault(default_factory)
        return default

    class _Model:
        def __init__(self, **kwargs):
            for cls in reversed(type(self).mro()):
                annotations = getattr(cls, "__annotations__", {})
                for name in annotations:
                    if name in kwargs:
                        value = kwargs[name]
                    elif hasattr(cls, name):
                        default = getattr(cls, name)
                        value = default.factory() if isinstance(default, _FactoryDefault) else default
                    else:
                        value = None
                    setattr(self, name, value)

    class Action(_Model):
        pass

    class Observation(_Model):
        done: bool = False
        reward: float = 0.0
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(_Model):
        episode_id: str
        step_count: int = 0
else:
    from pydantic import Field


class StartupAction(Action):
    action: str = Field(
        default="do_nothing",
        description="One MASS action: hire_employee, fire_employee, invest_in_product, run_marketing_campaign, do_nothing, or pivot_strategy.",
    )


class StartupObservation(Observation):
    day: int = 0
    money: float = 0.0
    users: int = 0
    product_quality: float = 0.0
    team_size: int = 0
    burn_rate: float = 0.0
    recent_user_growth: float = 0.0
    last_3_growth: List[float] = Field(default_factory=list)
    trend_direction: str = "stable"
    ad_performance: str = "unknown"
    runway_hint: float = 0.0
    crisis_level: str = "normal"
    crisis_reason: str = ""
    recent_events: List[str] = Field(default_factory=list)
    recent_actions: List[str] = Field(default_factory=list)
    event: Optional[str] = None
    reward_details: Dict[str, Any] = Field(default_factory=dict)


class StartupState(State):
    max_days: int = 30
    seed: Optional[int] = None
    public_state: Dict[str, Any] = Field(default_factory=dict)
