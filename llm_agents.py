import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from agents import ActionProposal, CEO, FinanceCoFounder, GrowthCoFounder, TechCoFounder
from environment import StartupEnvironment


ActionGenerator = Callable[[str], str]


@dataclass
class PromptArtifacts:
    system_prompt: str
    user_prompt: str


class PromptedAgentMixin:
    def __init__(
        self,
        name: str,
        fallback_agent,
        allowed_actions,
        generator: Optional[ActionGenerator] = None,
    ):
        self.name = name
        self.fallback_agent = fallback_agent
        self.allowed_actions = tuple(allowed_actions)
        self.generator = generator
        self.last_prompt: Optional[PromptArtifacts] = None
        self.last_raw_response: Optional[str] = None

    def propose(self, observation: Dict[str, object]) -> ActionProposal:
        fallback = self.fallback_agent.propose(observation)
        prompt = self.build_prompt(observation, fallback)
        self.last_prompt = prompt

        if self.generator is None:
            self.last_raw_response = None
            return fallback

        raw_response = self.generator(self._compose_prompt(prompt))
        self.last_raw_response = raw_response
        action = parse_action(raw_response, self.allowed_actions)
        if action is None:
            return fallback
        if (
            (observation.get("crisis_level") == "crisis" or observation.get("runway_hint", 999) < 2)
            and action in StartupEnvironment.CRISIS_DISALLOWED_ACTIONS
        ):
            if fallback.action not in StartupEnvironment.CRISIS_DISALLOWED_ACTIONS:
                return fallback
            return ActionProposal(
                action="do_nothing",
                reasoning=f"Prompt safety fallback: {action} is disallowed in crisis.",
            )

        return ActionProposal(action=action, reasoning=f"LLM-selected action from prompt scaffold: {action}")

    def build_prompt(self, observation: Dict[str, object], fallback: ActionProposal) -> PromptArtifacts:
        raise NotImplementedError

    @staticmethod
    def _compose_prompt(prompt: PromptArtifacts) -> str:
        return f"{prompt.system_prompt}\n\n{prompt.user_prompt}"


class PromptedTechCoFounder(PromptedAgentMixin):
    def __init__(self, generator: Optional[ActionGenerator] = None):
        super().__init__(
            name="Tech Co-founder",
            fallback_agent=TechCoFounder(),
            allowed_actions=StartupEnvironment.ACTIONS,
            generator=generator,
        )

    def build_prompt(self, observation: Dict[str, object], fallback: ActionProposal) -> PromptArtifacts:
        return PromptArtifacts(
            system_prompt=(
                "You are the Tech Co-founder of a startup operating under uncertainty. "
                "Use recent trends, the last three growth signals, runway hints, recent actions, and recent events to infer hidden conditions. "
                "Prioritize product quality, stability, and long-term retention, and avoid mindlessly repeating the same action unless the evidence is still strong. "
                "If the company is in crisis, prefer decisive recovery moves over passive waiting."
            ),
            user_prompt=_format_role_prompt(observation, self.allowed_actions, fallback.action),
        )


class PromptedGrowthCoFounder(PromptedAgentMixin):
    def __init__(self, generator: Optional[ActionGenerator] = None):
        super().__init__(
            name="Growth Co-founder",
            fallback_agent=GrowthCoFounder(),
            allowed_actions=StartupEnvironment.ACTIONS,
            generator=generator,
        )

    def build_prompt(self, observation: Dict[str, object], fallback: ActionProposal) -> PromptArtifacts:
        return PromptArtifacts(
            system_prompt=(
                "You are the Growth Co-founder of a startup operating under uncertainty. "
                "Use recent trends, the last three growth signals, runway hints, recent actions, and recent events to infer hidden conditions. "
                "Prioritize user growth, market capture, and momentum, and avoid repeating stale growth plays when recent evidence weakens. "
                "If the company is in crisis, prefer decisive recovery moves over passive waiting."
            ),
            user_prompt=_format_role_prompt(observation, self.allowed_actions, fallback.action),
        )


class PromptedFinanceCoFounder(PromptedAgentMixin):
    def __init__(self, generator: Optional[ActionGenerator] = None):
        super().__init__(
            name="Finance Co-founder",
            fallback_agent=FinanceCoFounder(),
            allowed_actions=StartupEnvironment.ACTIONS,
            generator=generator,
        )

    def build_prompt(self, observation: Dict[str, object], fallback: ActionProposal) -> PromptArtifacts:
        return PromptArtifacts(
            system_prompt=(
                "You are the Finance Co-founder of a startup operating under uncertainty. "
                "Use recent trends, the last three growth signals, runway hints, recent actions, and recent events to infer hidden conditions. "
                "Prioritize cash preservation, runway, and operational sustainability, but avoid getting stuck in repetitive cost-cutting when the business may need recovery. "
                "If the company is in crisis, prefer decisive survival-and-recovery moves over passive waiting."
            ),
            user_prompt=_format_role_prompt(observation, self.allowed_actions, fallback.action),
        )


class PromptedCEO:
    def __init__(
        self,
        generator: Optional[ActionGenerator] = None,
        fallback_agent: Optional[CEO] = None,
    ):
        self.name = "CEO"
        self.generator = generator
        self.fallback_agent = fallback_agent or CEO()
        self.allowed_actions = tuple(StartupEnvironment.ACTIONS)
        self.last_prompt: Optional[PromptArtifacts] = None
        self.last_raw_response: Optional[str] = None

    def choose_action(self, proposals: Dict[str, ActionProposal], observation: Dict[str, object]) -> ActionProposal:
        fallback = self.fallback_agent.choose_action(proposals, observation)
        prompt = self.build_prompt(proposals, observation, fallback)
        self.last_prompt = prompt

        if self.generator is None:
            self.last_raw_response = None
            return fallback

        if hasattr(self.generator, "generate_from_messages"):
            raw_response = self.generator.generate_from_messages(
                [
                    {"role": "system", "content": prompt.system_prompt},
                    {"role": "user", "content": prompt.user_prompt},
                ]
            )
        else:
            raw_response = self.generator(f"{prompt.system_prompt}\n\n{prompt.user_prompt}")
        self.last_raw_response = raw_response
        action = parse_action(raw_response, self.allowed_actions)
        if action is None:
            return fallback
        original_action = action
        action = self._apply_safety_gate(action, fallback, observation)
        if (
            (observation.get("crisis_level") == "crisis" or observation.get("runway_hint", 999) < 2)
            and action in StartupEnvironment.CRISIS_DISALLOWED_ACTIONS
        ):
            return ActionProposal(
                action="fire_employee",
                reasoning=f"LLM safety override: {action} is disallowed in crisis, so selected fire_employee.",
            )
        if action != original_action:
            return ActionProposal(
                action=action,
                reasoning=(
                    f"LLM safety gate changed {original_action} to {action} "
                    "because runway or cash risk was too high."
                ),
            )
        return ActionProposal(action=action, reasoning=f"LLM-selected final decision: {action}")

    def _apply_safety_gate(
        self,
        action: str,
        fallback: ActionProposal,
        observation: Dict[str, object],
    ) -> str:
        runway = float(observation.get("runway_hint", 999))
        money = float(observation.get("money", 0.0))
        burn_rate = float(observation.get("burn_rate", 1.0))
        recent_actions = list(observation.get("recent_actions", []))
        low_runway = runway < 4
        cash_stress = money < burn_rate * 4
        repeated_growth_spend = recent_actions[-2:].count("run_marketing_campaign") >= 2

        if action in {"run_marketing_campaign", "hire_employee"} and (
            low_runway or cash_stress or repeated_growth_spend
        ):
            if fallback.action not in StartupEnvironment.CRISIS_DISALLOWED_ACTIONS:
                return fallback.action
            return "fire_employee"

        if action == "invest_in_product" and runway < 2:
            if fallback.action not in StartupEnvironment.CRISIS_DISALLOWED_ACTIONS:
                return fallback.action
            return "fire_employee"

        return action

    def build_prompt(
        self,
        proposals: Dict[str, ActionProposal],
        observation: Dict[str, object],
        fallback: ActionProposal,
    ) -> PromptArtifacts:
        proposal_lines = [
            f"- {agent_name}: {proposal.action} | {proposal.reasoning}"
            for agent_name, proposal in proposals.items()
        ]
        user_prompt = (
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
            "Respond with exactly one line in the form: Action: <action_name>"
        )
        return PromptArtifacts(
            system_prompt=(
                "You are the CEO in a startup simulator. Choose one valid action "
                "from co-founder proposals while balancing survival, recovery, and growth."
            ),
            user_prompt=user_prompt,
        )


def build_prompted_agents(
    generator: Optional[ActionGenerator] = None,
) -> Tuple[PromptedTechCoFounder, PromptedGrowthCoFounder, PromptedFinanceCoFounder, PromptedCEO]:
    return (
        PromptedTechCoFounder(generator=generator),
        PromptedGrowthCoFounder(generator=generator),
        PromptedFinanceCoFounder(generator=generator),
        PromptedCEO(generator=generator),
    )


def build_trained_ceo_agents(
    adapter_path: str = "outputs/models/ceo-grpo",
    base_model: str = "Qwen/Qwen3-0.6B",
) -> Tuple[TechCoFounder, GrowthCoFounder, FinanceCoFounder, PromptedCEO]:
    generator = HuggingFaceActionGenerator(base_model=base_model, adapter_path=adapter_path)
    return (
        TechCoFounder(),
        GrowthCoFounder(),
        FinanceCoFounder(),
        PromptedCEO(generator=generator),
    )


class HuggingFaceActionGenerator:
    def __init__(
        self,
        base_model: str,
        adapter_path: str,
        max_new_tokens: int = 16,
    ):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.model = None

    def __call__(self, prompt: str) -> str:
        return self.generate_from_messages([{"role": "user", "content": prompt}])

    def generate_from_messages(self, messages) -> str:
        self._load()
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n\n".join(message["content"] for message in messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _load(self) -> None:
        if self.model is not None:
            return

        adapter_path = Path(self.adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Trained CEO adapter not found at {adapter_path}. "
                "Run train_ceo_grpo.py or set MASS_CEO_ADAPTER_PATH to a trained adapter directory."
            )

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "The trained CEO mode requires transformers, peft, and torch. "
                "Install the Colab training dependencies first."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(model, str(adapter_path))
        self.model.eval()


def parse_action(text: Optional[str], allowed_actions) -> Optional[str]:
    if not text:
        return None

    lowered = text.strip().lower()
    allowed = {action.lower(): action for action in allowed_actions}

    match = re.search(r"action\s*:\s*([a-z_]+)", lowered)
    if match and match.group(1) in allowed:
        return allowed[match.group(1)]

    for token in re.findall(r"[a-z_]+", lowered):
        if token in allowed:
            return allowed[token]
    return None


def _format_role_prompt(observation: Dict[str, object], allowed_actions, fallback_action: str) -> str:
    return (
        f"{_format_observation(observation)}\n\n"
        f"Allowed actions: {', '.join(allowed_actions)}\n"
        f"Fallback action if uncertain: {fallback_action}\n"
        "Reason silently about trend direction, uncertainty, and whether repeating the last action is justified.\n"
        "If crisis mode is active, favor a decisive recovery action instead of do_nothing unless every active option is clearly worse.\n"
        "If runway is below 2, do not choose run_marketing_campaign or hire_employee.\n"
        "Do not hire unless runway is above 6, all last 3 growth values are positive, and recent_events does not include viral_growth.\n"
        "Respond with exactly one line in the form: Action: <action_name>"
    )


def _format_observation(observation: Dict[str, object]) -> str:
    return (
        "Observed startup state:\n"
        f"- Day: {observation['day']}\n"
        f"- Money: {observation['money']}\n"
        f"- Users: {observation['users']}\n"
        f"- Product Quality: {observation['product_quality']}\n"
        f"- Team Size: {observation['team_size']}\n"
        f"- Burn Rate: {observation['burn_rate']}\n"
        f"- Recent User Growth Signal: {observation['recent_user_growth']}\n"
        f"- Last 3 Growth Signals: {observation['last_3_growth']}\n"
        f"- Trend Direction: {observation['trend_direction']}\n"
        f"- Ad Performance: {observation['ad_performance']}\n"
        f"- Runway Hint: {observation['runway_hint']}\n"
        f"- Crisis Mode: {observation['is_crisis']}\n"
        f"- Crisis Level: {observation['crisis_level']}\n"
        f"- Crisis Reason: {observation['crisis_reason']}\n"
        f"- Recent Events: {observation['recent_events']}\n"
        f"- Recent Actions: {observation['recent_actions']}\n"
        f"- Last Action: {observation['last_action']}\n"
        f"- Consecutive Action Streak: {observation['consecutive_action_streak']}"
    )
