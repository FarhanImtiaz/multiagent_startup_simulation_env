from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple


@dataclass
class ActionProposal:
    action: str
    reasoning: str


class BaseCoFounder:
    def __init__(self, name: str):
        self.name = name

    def propose(self, observation: Dict[str, object]) -> ActionProposal:
        raise NotImplementedError

    def _choose(
        self,
        action: str,
        reasoning: str,
        observation: Dict[str, object],
        fallback_action: str = "do_nothing",
        fallback_reason: str = "Avoiding a repeated move while we wait for a cleaner signal.",
    ) -> ActionProposal:
        if _was_repeated(observation, action):
            return ActionProposal(fallback_action, fallback_reason)
        return ActionProposal(action, reasoning)


ACTION_COSTS = {
    "hire_employee": 12000,
    "fire_employee": -4000,
    "invest_in_product": 9000,
    "run_marketing_campaign": 7000,
    "pivot_strategy": 11000,
    "do_nothing": 0,
}

CRISIS_UNSAFE_ACTIONS = {"hire_employee", "run_marketing_campaign"}


def _growth_window(observation: Dict[str, object]) -> Tuple[float, ...]:
    return tuple(observation.get("last_3_growth", ()))


def _average_growth(observation: Dict[str, object]) -> float:
    window = _growth_window(observation)
    if not window:
        return float(observation.get("recent_user_growth", 0))
    return sum(window) / len(window)


def _positive_growth(observation: Dict[str, object]) -> bool:
    window = _growth_window(observation)
    return len(window) >= 3 and all(value > 0 for value in window)


def _mostly_negative_growth(observation: Dict[str, object]) -> bool:
    window = _growth_window(observation)
    if not window:
        return float(observation.get("recent_user_growth", 0)) < 0
    return sum(1 for value in window if value < 0) >= 2


def _strong_decline(observation: Dict[str, object]) -> bool:
    return _mostly_negative_growth(observation) and _average_growth(observation) < -8


def _has_event(observation: Dict[str, object], event_name: str) -> bool:
    return event_name in observation.get("recent_events", [])


def _is_crisis(observation: Dict[str, object]) -> bool:
    return bool(observation.get("is_crisis")) or float(observation.get("runway_hint", 999)) < 1.5


def _recently_did(observation: Dict[str, object], action: str, lookback: int = 2) -> bool:
    return action in observation.get("recent_actions", [])[-lookback:]


def _recent_count(observation: Dict[str, object], action: str, lookback: int = 5) -> int:
    return sum(1 for recent_action in observation.get("recent_actions", [])[-lookback:] if recent_action == action)


def _was_repeated(observation: Dict[str, object], action: str, limit: int = 2) -> bool:
    return observation.get("last_action") == action and observation.get("consecutive_action_streak", 0) >= limit


def _can_afford(action: str, observation: Dict[str, object]) -> bool:
    cost = ACTION_COSTS.get(action, 0)
    if cost <= 0:
        return True

    money = float(observation.get("money", 0))
    burn_rate = float(observation.get("burn_rate", 12000))
    if action in {"hire_employee", "pivot_strategy"}:
        buffer = burn_rate * 0.65
    elif action == "invest_in_product":
        buffer = burn_rate * 0.35
    else:
        buffer = burn_rate * 0.1
    return money >= cost + buffer


def _first_affordable(actions: Iterable[str], observation: Dict[str, object]) -> str:
    for action in actions:
        if _can_afford(action, observation):
            return action
    return "do_nothing"


class TechCoFounder(BaseCoFounder):
    def __init__(self):
        super().__init__("Tech Co-founder")

    def propose(self, observation: Dict[str, object]) -> ActionProposal:
        quality = float(observation["product_quality"])
        runway = float(observation["runway_hint"])

        if _is_crisis(observation):
            if (_has_event(observation, "tech_failure") or quality < 0.52) and _can_afford("invest_in_product", observation):
                return ActionProposal(
                    "invest_in_product",
                    "The company is under pressure and product risk is visible, so repair quality before asking growth to carry us.",
                )
            if _strong_decline(observation) and _can_afford("pivot_strategy", observation):
                return ActionProposal(
                    "pivot_strategy",
                    "The product signal alone may not explain the decline, so a strategic reset is worth considering.",
                )
            return ActionProposal(
                "do_nothing",
                "Cash is tight and the product signal is not bad enough to justify a spend-heavy technical move.",
            )

        if _has_event(observation, "tech_failure") or quality < 0.56:
            return self._choose(
                "invest_in_product",
                "Product quality is the main constraint; improving it should reduce churn and support future growth.",
                observation,
                fallback_action="pivot_strategy",
                fallback_reason="Product work has repeated, so test whether positioning is the real problem.",
            )

        if _strong_decline(observation) and runway >= 3.0:
            return ActionProposal(
                "invest_in_product",
                "Growth is weakening and product quality is not strong enough to rule out retention problems.",
            )

        if observation["trend_direction"] == "improving" and quality >= 0.72 and runway > 6.0:
            return ActionProposal(
                "hire_employee",
                "Quality and runway are both strong, so extra capacity can compound the current momentum.",
            )

        return ActionProposal(
            "invest_in_product",
            "A measured product improvement is the safest technical contribution while signals remain mixed.",
        )


class GrowthCoFounder(BaseCoFounder):
    def __init__(self):
        super().__init__("Growth Co-founder")

    def propose(self, observation: Dict[str, object]) -> ActionProposal:
        ad_performance = observation["ad_performance"]
        quality = float(observation["product_quality"])
        runway = float(observation["runway_hint"])

        if _is_crisis(observation):
            if _strong_decline(observation) and not _recently_did(observation, "pivot_strategy") and _can_afford("pivot_strategy", observation):
                return ActionProposal(
                    "pivot_strategy",
                    "Acquisition is unsafe in crisis, so the growth move is to change positioning instead of buying traffic.",
                )
            return ActionProposal(
                "do_nothing",
                "Growth spend is unsafe while runway is short; preserve cash until the company exits crisis.",
            )

        if _has_event(observation, "viral_growth") and runway > 2.5 and quality >= 0.55:
            return self._choose(
                "run_marketing_campaign",
                "A viral signal is active, so marketing can convert attention into a larger retained user base.",
                observation,
                fallback_action="invest_in_product",
                fallback_reason="Marketing has repeated, so improve retention before spending again.",
            )

        if _strong_decline(observation) or ad_performance == "poor":
            if _can_afford("pivot_strategy", observation) and not _recently_did(observation, "pivot_strategy"):
                return ActionProposal(
                    "pivot_strategy",
                    "Growth signals are weak enough that positioning may be the constraint.",
                )
            return ActionProposal(
                "invest_in_product",
                "Growth is weak, but a pivot is blocked or unaffordable, so improve product quality first.",
            )

        if _can_afford("run_marketing_campaign", observation) and quality >= 0.58:
            return self._choose(
                "run_marketing_campaign",
                "Demand is usable and product quality is high enough to support acquisition.",
                observation,
                fallback_action="do_nothing",
                fallback_reason="Marketing has repeated, so pause one turn and let the last campaign settle.",
            )

        return ActionProposal(
            "invest_in_product",
            "The product is not ready enough for efficient growth spend, so improve retention first.",
        )


class FinanceCoFounder(BaseCoFounder):
    def __init__(self):
        super().__init__("Finance Co-founder")

    def propose(self, observation: Dict[str, object]) -> ActionProposal:
        money = float(observation["money"])
        runway = float(observation["runway_hint"])
        team_size = int(observation["team_size"])
        burn_rate = float(observation["burn_rate"])

        if runway < 5.0 or money < 55000:
            if team_size > 1 and not _was_repeated(observation, "fire_employee"):
                return ActionProposal(
                    "fire_employee",
                    "Runway is tight, so reducing burn is the clearest survival move.",
                )
            return ActionProposal(
                "do_nothing",
                "The company is already lean or recently cut; preserve cash instead of forcing another expensive move.",
            )

        if burn_rate > 18000 and observation["trend_direction"] != "improving":
            return ActionProposal(
                "do_nothing",
                "Burn is high and growth is not clearly improving, so avoid new spend this turn.",
            )

        if runway > 6.0 and _positive_growth(observation):
            return ActionProposal(
                "hire_employee",
                "Runway and growth are both healthy enough to support careful hiring.",
            )

        return ActionProposal(
            "do_nothing",
            "The financial position is acceptable but not strong enough to force a spend-heavy decision.",
        )


class CEO:
    def __init__(self):
        self.name = "CEO"

    def choose_action(self, proposals: Dict[str, ActionProposal], observation: Dict[str, object]) -> ActionProposal:
        focus, focus_reason = self._determine_focus(observation)
        action, policy_reason = self._policy_action(observation)
        matched = self._matching_proposal(proposals, action)

        if matched is None:
            proposal_text = "No proposal matched exactly; CEO synthesized the final action."
        else:
            proposal_text = f"Aligned with proposal: {matched.reasoning}"

        return ActionProposal(
            action,
            (
                f"Focus={focus}. {focus_reason} "
                f"Selected {action}: {policy_reason} "
                f"{proposal_text}"
            ),
        )

    def _determine_focus(self, observation: Dict[str, object]) -> Tuple[str, str]:
        runway = float(observation["runway_hint"])
        money = float(observation["money"])
        quality = float(observation["product_quality"])
        trend = observation["trend_direction"]

        if _is_crisis(observation):
            return "survival", "Runway or cash is in the danger zone, so survival beats growth."
        if runway < 5.0 or money < 50000:
            return "cash_control", "Runway is short, so reduce burn and avoid large commitments."
        if _strong_decline(observation):
            return "recovery", "Recent growth is meaningfully negative, so intervene instead of drifting."
        if trend == "improving" and _average_growth(observation) > 10:
            return "growth_capture", "Growth is improving, so capture momentum while runway allows it."
        if quality < 0.62 or _has_event(observation, "tech_failure"):
            return "product_stability", "Product risk is visible, so retention needs attention."
        return "balanced", "Signals are mixed, so choose the best risk-adjusted action."

    def _policy_action(self, observation: Dict[str, object]) -> Tuple[str, str]:
        runway = float(observation["runway_hint"])
        money = float(observation["money"])
        quality = float(observation["product_quality"])
        trend = observation["trend_direction"]
        recent_growth = float(observation.get("recent_user_growth", 0))
        team_size = int(observation["team_size"])

        if _is_crisis(observation):
            if team_size > 1 and not _was_repeated(observation, "fire_employee"):
                return "fire_employee", "cut burn immediately while there is still team size to reduce."
            if _strong_decline(observation):
                action = _first_affordable(("invest_in_product", "pivot_strategy", "do_nothing"), observation)
                return action, "growth is falling in crisis, so use the cheapest active recovery move."
            return "do_nothing", "the company is lean and no active recovery move is affordable enough."

        if runway < 5.0 or money < 50000:
            if team_size > 1 and not _was_repeated(observation, "fire_employee"):
                return "fire_employee", "warning-zone runway needs burn reduction before new spending."
            if (
                money > 35000
                and quality >= 0.62
                and (recent_growth > 20 or _mostly_negative_growth(observation))
                and _recent_count(observation, "run_marketing_campaign") < 3
                and _can_afford("run_marketing_campaign", observation)
                and not _was_repeated(observation, "run_marketing_campaign")
            ):
                return "run_marketing_campaign", "the team is already lean, so use limited variable growth spend above the cash floor."
            if (
                money > 60000
                and quality < 0.56
                and _recent_count(observation, "invest_in_product") < 2
                and _can_afford("invest_in_product", observation)
                and not _was_repeated(observation, "invest_in_product")
            ):
                return "invest_in_product", "the company is lean, and product quality is still weak enough to justify one repair move."
            return "do_nothing", "runway is short and no high-confidence spend is justified."

        if _strong_decline(observation):
            if money > 60000 and quality < 0.72 and _recent_count(observation, "invest_in_product") < 2 and _can_afford("invest_in_product", observation) and not _was_repeated(observation, "invest_in_product"):
                return "invest_in_product", "decline usually gets one product recovery attempt before a pivot."
            if money > 65000 and not _recently_did(observation, "pivot_strategy") and _can_afford("pivot_strategy", observation):
                return "pivot_strategy", "decline continued after product work or with adequate quality, so reset positioning."
            return "do_nothing", "recovery spending is blocked by budget discipline, so preserve cash."

        if trend == "improving" and recent_growth > 15:
            if money > 45000 and _recent_count(observation, "run_marketing_campaign") < 3 and _can_afford("run_marketing_campaign", observation) and not _was_repeated(observation, "run_marketing_campaign"):
                return "run_marketing_campaign", "positive momentum is best captured with variable growth spend."
            if money > 75000 and runway > 6.0 and _positive_growth(observation) and _can_afford("hire_employee", observation):
                return "hire_employee", "growth is consistent and runway is healthy enough for capacity."
            if money > 65000 and quality < 0.75 and _recent_count(observation, "invest_in_product") < 2 and _can_afford("invest_in_product", observation):
                return "invest_in_product", "marketing has repeated, so improve retention before another push."
            return "do_nothing", "momentum exists, but the next spend is not clearly efficient."

        if _recently_did(observation, "run_marketing_campaign", lookback=1) and recent_growth >= -5:
            return "do_nothing", "pause after marketing to let users generate revenue before another spend."

        if money > 65000 and quality < 0.56 and _recent_count(observation, "invest_in_product") < 2 and _can_afford("invest_in_product", observation) and not _was_repeated(observation, "invest_in_product"):
            return "invest_in_product", "quality is below the safe scaling threshold."

        if money > 50000 and quality >= 0.56 and _recent_count(observation, "run_marketing_campaign") < 3 and _can_afford("run_marketing_campaign", observation) and not _was_repeated(observation, "run_marketing_campaign"):
            return "run_marketing_campaign", "product is adequate and marketing is the cheapest growth lever."

        if money > 55000 and _can_afford("run_marketing_campaign", observation) and recent_growth > 35:
            return "run_marketing_campaign", "growth is strong enough to justify another marketing push despite repetition risk."

        return "do_nothing", "no strong signal justifies spending this turn."

    @staticmethod
    def _matching_proposal(
        proposals: Dict[str, ActionProposal],
        action: str,
    ) -> Optional[ActionProposal]:
        for proposal in proposals.values():
            if proposal.action == action:
                return proposal
        return None


def build_heuristic_agents() -> Tuple[TechCoFounder, GrowthCoFounder, FinanceCoFounder, CEO]:
    return TechCoFounder(), GrowthCoFounder(), FinanceCoFounder(), CEO()
