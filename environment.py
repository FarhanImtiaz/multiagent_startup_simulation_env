import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class HiddenState:
    market_demand: float = 0.7
    competition_level: float = 0.35
    economic_condition: float = 0.75


@dataclass
class PendingEffect:
    name: str
    eta: int
    payload: Dict[str, float]


@dataclass
class StartupState:
    day: int = 1
    money: float = 100000.0
    users: int = 150
    product_quality: float = 0.5
    team_size: int = 2
    burn_rate: float = 12000.0
    recent_actions: List[str] = field(default_factory=list)
    last_users: int = 150
    last_money: float = 100000.0
    last_event: str = "none"
    last_ad_performance: str = "average"
    recent_events: List[str] = field(default_factory=list)
    ignored_negative_trends: int = 0
    growth_history: List[int] = field(default_factory=lambda: [0])


class StartupEnvironment:
    TREND_WINDOW = 3
    MAX_REPEAT_WITHOUT_PENALTY = 2
    REWARD_SCALE = 18.0
    CRISIS_RUNWAY_THRESHOLD = 1.0
    CRISIS_CASH_THRESHOLD = 20000.0
    BANKRUPTCY_PENALTY = 15.0
    BANKRUPTCY_TERMINAL_MULTIPLIER = 2.0
    BANKRUPTCY_USER_PENALTY_SCALE = 0.035
    CRISIS_DISALLOWED_ACTIONS = {"run_marketing_campaign", "hire_employee"}
    UNSAFE_CRISIS_RECOMMENDATION_PENALTY = 1.0

    ACTIONS = [
        "hire_employee",
        "fire_employee",
        "invest_in_product",
        "run_marketing_campaign",
        "do_nothing",
        "pivot_strategy",
    ]

    EVENT_TYPES = [
        "competitor_launch",
        "market_crash",
        "viral_growth",
        "tech_failure",
    ]

    def __init__(self, max_days: int = 30, seed: Optional[int] = None):
        self.max_days = max_days
        self.random = random.Random(seed)
        self.state = StartupState()
        self.hidden_state = HiddenState()
        self.pending_effects: List[PendingEffect] = []
        self.history: List[Dict[str, object]] = []
        self.reset()

    def reset(self) -> Dict[str, object]:
        self.state = StartupState()
        self.hidden_state = HiddenState()
        self.pending_effects = []
        self.history = []
        return self.get_observation()

    def get_observation(self) -> Dict[str, object]:
        recent_user_growth = self.state.users - self.state.last_users
        noisy_growth = recent_user_growth + self.random.randint(-6, 6)
        noisy_runway = self.state.money / max(self.state.burn_rate, 1)
        growth_window = self._get_growth_window()
        noisy_growth_window = [value + self.random.randint(-4, 4) for value in growth_window]
        trend_direction = self._infer_trend_direction(growth_window)
        last_action = self.state.recent_actions[-1] if self.state.recent_actions else "none"
        consecutive_action_streak = self._count_consecutive_actions(last_action)
        runway_hint = round(noisy_runway + self.random.uniform(-0.5, 0.5), 2)
        crisis_level, crisis_reason = self._get_crisis_status(
            runway_hint=runway_hint,
            money=self.state.money,
        )

        return {
            "day": self.state.day,
            "money": round(self.state.money, 2),
            "users": self.state.users,
            "product_quality": round(self.state.product_quality, 3),
            "team_size": self.state.team_size,
            "burn_rate": round(self.state.burn_rate, 2),
            "recent_user_growth": noisy_growth,
            "last_3_growth": noisy_growth_window,
            "trend_direction": trend_direction,
            "ad_performance": self.state.last_ad_performance,
            "recent_actions": list(self.state.recent_actions[-5:]),
            "last_action": last_action,
            "consecutive_action_streak": consecutive_action_streak,
            "recent_events": list(self.state.recent_events[-3:]),
            "runway_hint": runway_hint,
            "is_crisis": crisis_level == "crisis",
            "crisis_level": crisis_level,
            "crisis_reason": crisis_reason,
        }

    def get_debug_state(self) -> Dict[str, object]:
        return {
            "public_state": {
                "day": self.state.day,
                "money": round(self.state.money, 2),
                "users": self.state.users,
                "product_quality": round(self.state.product_quality, 3),
                "team_size": self.state.team_size,
                "burn_rate": round(self.state.burn_rate, 2),
                "recent_actions": list(self.state.recent_actions[-5:]),
                "recent_events": list(self.state.recent_events[-3:]),
                "growth_history": self._get_growth_window(),
                "trend_direction": self._infer_trend_direction(self._get_growth_window()),
                "crisis_level": self._get_crisis_status(
                    runway_hint=self.state.money / max(self.state.burn_rate, 1),
                    money=self.state.money,
                )[0],
            },
            "hidden_state": {
                "market_demand": round(self.hidden_state.market_demand, 3),
                "competition_level": round(self.hidden_state.competition_level, 3),
                "economic_condition": round(self.hidden_state.economic_condition, 3),
            },
            "pending_effects": [
                {"name": effect.name, "eta": effect.eta, "payload": effect.payload}
                for effect in self.pending_effects
            ],
        }

    def step(self, action: str, proposals: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        pre_action_crisis_level, _ = self._get_crisis_status(
            runway_hint=self.state.money / max(self.state.burn_rate, 1),
            money=self.state.money,
        )
        self.state.last_users = self.state.users
        self.state.last_money = self.state.money
        self.state.last_event = "none"

        self._apply_pending_effects()
        self._apply_action(action)
        self._apply_environment_dynamics(action)
        event = self._sample_event()
        current_growth = self.state.users - self.state.last_users
        self._record_growth(current_growth)
        reward, reward_details = self._compute_reward(action)
        agent_rewards = self._compute_agent_rewards(
            action,
            reward,
            proposals or {},
            pre_action_crisis_level,
        )
        termination_reason = self._get_termination_reason()
        done = termination_reason != "running"

        self.state.day += 1
        self.state.recent_actions.append(action)
        if len(self.state.recent_actions) > 10:
            self.state.recent_actions = self.state.recent_actions[-10:]

        observation = self.get_observation()
        debug_state = self.get_debug_state()

        step_record = {
            "day": self.state.day - 1,
            "action": action,
            "reward": reward,
            "raw_reward": reward_details["raw_reward"],
            "event": event,
            "money": observation["money"],
            "users": observation["users"],
            "quality": observation["product_quality"],
            "trend_direction": observation["trend_direction"],
            "crisis_level": observation["crisis_level"],
            "termination_reason": termination_reason,
        }
        self.history.append(step_record)

        return {
            "state": observation,
            "reward": reward,
            "raw_reward": reward_details["raw_reward"],
            "reward_details": reward_details,
            "done": done,
            "termination_reason": termination_reason,
            "action": action,
            "event": event,
            "agent_rewards": agent_rewards,
            "debug_state": debug_state,
        }

    def _record_growth(self, growth: int) -> None:
        self.state.growth_history.append(growth)
        max_history = self.TREND_WINDOW + 3
        if len(self.state.growth_history) > max_history:
            self.state.growth_history = self.state.growth_history[-max_history:]

    def _get_growth_window(self) -> List[int]:
        growth_window = list(self.state.growth_history[-self.TREND_WINDOW :])
        if len(growth_window) < self.TREND_WINDOW:
            growth_window = [0] * (self.TREND_WINDOW - len(growth_window)) + growth_window
        return growth_window

    @staticmethod
    def _infer_trend_direction(growth_window: List[int]) -> str:
        if len(growth_window) < 2:
            return "stable"

        prior_average = sum(growth_window[:-1]) / max(1, len(growth_window) - 1)
        latest = growth_window[-1]
        delta = latest - prior_average

        if delta > 12:
            return "improving"
        if delta < -12:
            return "declining"
        return "stable"

    def _count_consecutive_actions(self, action: str) -> int:
        if action == "none":
            return 0

        streak = 0
        for recorded_action in reversed(self.state.recent_actions):
            if recorded_action != action:
                break
            streak += 1
        return streak

    def _get_crisis_status(self, runway_hint: float, money: float) -> tuple[str, str]:
        if runway_hint < self.CRISIS_RUNWAY_THRESHOLD or money < self.CRISIS_CASH_THRESHOLD:
            reasons = []
            if runway_hint < self.CRISIS_RUNWAY_THRESHOLD:
                reasons.append("runway is below the crisis threshold")
            if money < self.CRISIS_CASH_THRESHOLD:
                reasons.append("cash is below the critical threshold")
            return "crisis", " and ".join(reasons)

        if runway_hint < 1.8 or money < self.CRISIS_CASH_THRESHOLD * 1.6:
            reasons = []
            if runway_hint < 1.8:
                reasons.append("runway is getting short")
            if money < self.CRISIS_CASH_THRESHOLD * 1.6:
                reasons.append("cash is getting tight")
            return "warning", " and ".join(reasons)

        return "normal", "company has enough room to make measured decisions"

    def _apply_pending_effects(self) -> None:
        remaining_effects: List[PendingEffect] = []
        for effect in self.pending_effects:
            effect.eta -= 1
            if effect.eta > 0:
                remaining_effects.append(effect)
                continue

            if effect.name == "hire_productivity_boost":
                self.state.product_quality = min(
                    1.0,
                    self.state.product_quality + effect.payload.get("quality_gain", 0.0),
                )
            elif effect.name == "product_quality_boost":
                self.state.product_quality = min(
                    1.0,
                    self.state.product_quality + effect.payload.get("quality_gain", 0.0),
                )
            elif effect.name == "marketing_decay":
                decay = int(effect.payload.get("user_decay", 0))
                self.state.users = max(0, self.state.users - decay)
            elif effect.name == "pivot_demand_shift":
                self.hidden_state.market_demand = min(
                    1.0,
                    max(0.0, self.hidden_state.market_demand + effect.payload.get("demand_gain", 0.0)),
                )

        self.pending_effects = remaining_effects

    def _apply_action(self, action: str) -> None:
        if action == "hire_employee":
            self.state.money -= 12000
            self.state.team_size += 1
            self.state.burn_rate += 2500
            self.pending_effects.append(
                PendingEffect(
                    name="hire_productivity_boost",
                    eta=2,
                    payload={"quality_gain": 0.04},
                )
            )

        elif action == "fire_employee":
            if self.state.team_size > 1:
                self.state.team_size -= 1
                self.state.money += 4000
                self.state.burn_rate = max(6000, self.state.burn_rate - 2000)
                self.state.product_quality = max(0.2, self.state.product_quality - 0.02)

        elif action == "invest_in_product":
            self.state.money -= 9000
            self.state.product_quality = min(1.0, self.state.product_quality + 0.03)
            self.pending_effects.extend(
                [
                    PendingEffect("product_quality_boost", 1, {"quality_gain": 0.03}),
                    PendingEffect("product_quality_boost", 3, {"quality_gain": 0.04}),
                ]
            )

        elif action == "run_marketing_campaign":
            self.state.money -= 7000
            success_prob = (
                0.2
                + 0.5 * self.hidden_state.market_demand
                + 0.2 * self.hidden_state.economic_condition
                + 0.15 * self.state.product_quality
                - 0.35 * self.hidden_state.competition_level
            )
            success_prob = max(0.05, min(0.95, success_prob))

            if self.random.random() < success_prob:
                gain = self.random.randint(90, 180)
                self.state.last_ad_performance = "good"
            else:
                gain = self.random.randint(5, 60)
                self.state.last_ad_performance = "poor" if gain < 20 else "average"

            self.state.users += gain
            self.pending_effects.append(
                PendingEffect(
                    name="marketing_decay",
                    eta=2,
                    payload={"user_decay": max(5, int(gain * 0.18))},
                )
            )

        elif action == "do_nothing":
            self.state.last_ad_performance = "average"

        elif action == "pivot_strategy":
            self.state.money -= 11000
            self.state.product_quality = max(0.25, self.state.product_quality - 0.05)
            self.pending_effects.append(
                PendingEffect(
                    name="pivot_demand_shift",
                    eta=2,
                    payload={"demand_gain": 0.14},
                )
            )

    def _apply_environment_dynamics(self, action: str) -> None:
        recurring_revenue = self.state.users * (0.22 + 0.35 * self.state.product_quality)
        recurring_revenue *= 0.7 + 0.3 * self.hidden_state.economic_condition
        self.state.money += recurring_revenue

        self.state.burn_rate = max(7000, 7500 + self.state.team_size * 2200)
        self.state.money -= self.state.burn_rate * 0.09

        churn_rate = (
            0.09
            - 0.05 * self.state.product_quality
            + 0.04 * self.hidden_state.competition_level
            - 0.025 * self.hidden_state.economic_condition
        )
        churn = int(self.state.users * max(0.01, churn_rate))
        self.state.users = max(0, self.state.users - churn)

        self.hidden_state.market_demand = min(
            1.0,
            max(0.05, self.hidden_state.market_demand + self.random.uniform(-0.04, 0.05)),
        )
        self.hidden_state.competition_level = min(
            1.0,
            max(0.05, self.hidden_state.competition_level + self.random.uniform(-0.03, 0.04)),
        )
        self.hidden_state.economic_condition = min(
            1.0,
            max(0.2, self.hidden_state.economic_condition + self.random.uniform(-0.03, 0.03)),
        )

        recent_growth = self.state.users - self.state.last_users
        if recent_growth < 0 and action in {"do_nothing", "hire_employee"}:
            self.state.ignored_negative_trends += 1
        else:
            self.state.ignored_negative_trends = max(0, self.state.ignored_negative_trends - 1)

    def _sample_event(self) -> str:
        event_probability = 0.22
        if self.random.random() > event_probability:
            self.state.recent_events.append("none")
            self.state.recent_events = self.state.recent_events[-5:]
            return "none"

        event = self.random.choice(self.EVENT_TYPES)
        self.state.last_event = event

        if event == "competitor_launch":
            loss = self.random.randint(15, 50)
            self.state.users = max(0, self.state.users - loss)
            self.hidden_state.competition_level = min(1.0, self.hidden_state.competition_level + 0.12)
        elif event == "market_crash":
            self.hidden_state.market_demand = max(0.05, self.hidden_state.market_demand - 0.18)
            self.hidden_state.economic_condition = max(0.2, self.hidden_state.economic_condition - 0.15)
            self.state.money -= 2500
        elif event == "viral_growth":
            gain = self.random.randint(50, 120)
            self.state.users += gain
            self.hidden_state.market_demand = min(1.0, self.hidden_state.market_demand + 0.08)
        elif event == "tech_failure":
            self.state.product_quality = max(0.15, self.state.product_quality - 0.12)
            self.state.money -= 1800

        self.state.recent_events.append(event)
        self.state.recent_events = self.state.recent_events[-5:]
        return event

    def _compute_reward(self, action: str) -> tuple[float, Dict[str, float]]:
        long_term_user_growth = self.state.users - self.state.last_users
        profit = self.state.money - self.state.last_money
        product_quality = self.state.product_quality
        burn = self.state.burn_rate / 1000.0
        runway = self.state.money / max(self.state.burn_rate, 1)
        repeated_count = 1 + self._count_consecutive_actions(action)
        crisis_level, _ = self._get_crisis_status(
            runway_hint=runway,
            money=self.state.money,
        )
        is_crisis = crisis_level == "crisis"
        growth_weight = 0.22
        profit_weight = 0.24
        burn_weight = 0.14

        if self.state.money < self.CRISIS_CASH_THRESHOLD:
            profit_weight *= 2.5
            growth_weight *= 0.5
            burn_weight *= 1.8

        if runway < 1.0:
            growth_weight *= 0.3

        growth_component = growth_weight * long_term_user_growth
        profit_component = profit_weight * (profit / 1000.0)
        quality_component = 7.0 * product_quality
        burn_component = -burn_weight * burn

        repeat_penalty_multiplier = 1.8 if is_crisis and action == "fire_employee" else 2.5
        repeat_penalty = max(0, repeated_count - self.MAX_REPEAT_WITHOUT_PENALTY) * repeat_penalty_multiplier
        if action == "pivot_strategy" and repeated_count > 1:
            repeat_penalty += 3.0 * (repeated_count - 1)
        negative_trend_penalty = 1.0 * self.state.ignored_negative_trends
        bankruptcy_penalty = 0.0
        if self.state.money < 0:
            bankruptcy_penalty = self.BANKRUPTCY_PENALTY + self.state.users * self.BANKRUPTCY_USER_PENALTY_SCALE
        terminal_bankruptcy_penalty = (
            self.BANKRUPTCY_PENALTY * self.BANKRUPTCY_TERMINAL_MULTIPLIER
            if self.state.money < 0
            else 0.0
        )
        crisis_growth_burn_penalty = 0.0
        if is_crisis and action == "run_marketing_campaign":
            crisis_growth_burn_penalty = 2.0 + burn * 0.25
        survival_action_bonus = 1.2 if is_crisis and action == "fire_employee" else 0.0
        recovery_bonus = self._compute_recovery_bonus(long_term_user_growth, profit)
        crisis_response_bonus, crisis_freeze_penalty = self._compute_crisis_response_reward(
            action=action,
            current_growth=long_term_user_growth,
            profit=profit,
            is_crisis=is_crisis,
        )

        raw_reward = (
            growth_component
            + profit_component
            + quality_component
            + burn_component
            + recovery_bonus
            + crisis_response_bonus
            + survival_action_bonus
            - repeat_penalty
            - negative_trend_penalty
            - bankruptcy_penalty
            - terminal_bankruptcy_penalty
            - crisis_freeze_penalty
            - crisis_growth_burn_penalty
        )
        normalized_reward = math.tanh(raw_reward / self.REWARD_SCALE)
        reward_details = {
            "raw_reward": round(raw_reward, 3),
            "normalized_reward": round(normalized_reward, 3),
            "growth_component": round(growth_component, 3),
            "profit_component": round(profit_component, 3),
            "quality_component": round(quality_component, 3),
            "burn_component": round(burn_component, 3),
            "growth_weight": round(growth_weight, 3),
            "profit_weight": round(profit_weight, 3),
            "burn_weight": round(burn_weight, 3),
            "repeat_penalty": round(repeat_penalty, 3),
            "negative_trend_penalty": round(negative_trend_penalty, 3),
            "recovery_bonus": round(recovery_bonus, 3),
            "crisis_response_bonus": round(crisis_response_bonus, 3),
            "crisis_freeze_penalty": round(crisis_freeze_penalty, 3),
            "crisis_growth_burn_penalty": round(crisis_growth_burn_penalty, 3),
            "survival_action_bonus": round(survival_action_bonus, 3),
            "bankruptcy_penalty": round(bankruptcy_penalty, 3),
            "terminal_bankruptcy_penalty": round(terminal_bankruptcy_penalty, 3),
        }
        return round(normalized_reward, 3), reward_details

    def _compute_recovery_bonus(self, current_growth: int, profit: float) -> float:
        growth_window = self._get_growth_window()
        previous_growth = growth_window[-2] if len(growth_window) >= 2 else 0
        earlier_growth = growth_window[-3] if len(growth_window) >= 3 else previous_growth

        recovery_bonus = 0.0
        if previous_growth < -10 and current_growth > previous_growth + 15:
            recovery_bonus += 1.8

        if earlier_growth < -10 and previous_growth < 0 and current_growth > 0:
            recovery_bonus += 1.4

        adverse_events = {"competitor_launch", "market_crash", "tech_failure"}
        if any(event in adverse_events for event in self.state.recent_events[-3:]):
            if current_growth >= -5 and profit > 0:
                recovery_bonus += 1.2

        return recovery_bonus

    def _compute_crisis_response_reward(
        self,
        action: str,
        current_growth: int,
        profit: float,
        is_crisis: bool,
    ) -> tuple[float, float]:
        if not is_crisis:
            return 0.0, 0.0

        if action == "do_nothing":
            return 0.0, 3.0

        response_bonus = 1.0
        if action == "fire_employee" and self.state.team_size >= 1:
            response_bonus += 0.6
        if action == "pivot_strategy" and current_growth > -20:
            response_bonus += 0.8
        if action == "run_marketing_campaign" and current_growth >= -5:
            response_bonus += 0.8
        if action == "invest_in_product" and self.state.product_quality < 0.6:
            response_bonus += 0.6
        if profit > -2000:
            response_bonus += 0.7
        if current_growth > 0:
            response_bonus += 0.9

        return response_bonus, 0.0

    def _compute_agent_rewards(
        self,
        chosen_action: str,
        reward: float,
        proposals: Dict[str, object],
        crisis_level: str = "normal",
    ) -> Dict[str, float]:
        agent_rewards = {"CEO": reward}
        for agent_name, proposal in proposals.items():
            proposed_action = getattr(proposal, "action", None)
            aligned = proposed_action == chosen_action
            agent_reward = reward if aligned else reward * 0.2
            if crisis_level == "crisis" and proposed_action in self.CRISIS_DISALLOWED_ACTIONS:
                agent_reward -= self.UNSAFE_CRISIS_RECOMMENDATION_PENALTY
            agent_rewards[agent_name] = round(agent_reward, 3)
        return agent_rewards

    def _is_done(self) -> bool:
        return self._get_termination_reason() != "running"

    def _get_termination_reason(self) -> str:
        if self.state.day >= self.max_days:
            return "max_days"
        if self.state.money < 0:
            return "bankrupt"
        if self.state.users <= 0:
            return "no_users"
        return "running"
