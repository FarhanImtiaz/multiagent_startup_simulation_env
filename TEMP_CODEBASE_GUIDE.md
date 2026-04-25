# TEMP Codebase Guide

This file is a temporary deep-dive guide to help you understand the project before making changes.

Use it as a reading companion, not as a source of truth over the code. When this file and the code disagree, trust the code.

## 1. What This Project Is

This repo implements a startup simulation called `MASS`:

- A shared startup environment evolves over time.
- Three specialized co-founders propose actions.
- A CEO chooses one final action.
- The environment applies the action, hidden world dynamics, random events, delayed effects, and rewards.
- Separate scripts run one episode, many episodes, or collect trajectories for future training.

At a practical level, the project is currently a simulation prototype plus scaffolding for future LLM control and future RL-style training.

## 2. Fast Mental Model

If you want the shortest possible understanding, think of the code like this:

1. `simulate.py` is the main orchestrator.
2. `agents.py` decides what each role wants to do.
3. `environment.py` is the actual world and reward system.
4. `evaluation.py` repeats simulations and aggregates metrics.
5. `train.py` collects logs/trajectories, but does not optimize a model.
6. `llm_agents.py` is a prompt-and-parse scaffold, not a real model integration yet.
7. `openenv_wrapper.py` is a thin adapter for training-style interfaces.

## 3. Suggested Read Order

Read the code in this order if your goal is deep understanding:

1. `README.md`
2. `simulate.py`
3. `agents.py`
4. `environment.py`
5. `evaluation.py`
6. `train.py`
7. `llm_agents.py`
8. `openenv_wrapper.py`
9. `Project_Overview.md`

Reason: this order helps you understand runtime flow first, then the decision logic, then the state transitions.

## 4. File-By-File Guide

### `README.md`

Purpose:

- High-level project summary.
- Quick run commands.
- Lists each core file.

What it is good for:

- Orienting yourself quickly.
- Seeing intended usage.

What it is not:

- Not deep enough to explain internals or edge cases.

### `Project_Overview.md`

Purpose:

- Hackathon/product vision document.
- Describes the intended design and pitch framing.

Important note:

- This is the conceptual blueprint, not the exact implementation.
- Use it to understand why the project exists, not to infer exact coefficients or actual code paths.

### `TEMP_IMPLEMENTATION_CHECKLIST.md`

Purpose:

- Temporary implementation tracker.
- Useful for understanding what is done vs still planned.

Important note:

- It reflects intent and status, not necessarily bugs or inconsistencies in the current code.

### `environment.py`

This is the heart of the project.

It defines:

- `HiddenState`
- `PendingEffect`
- `StartupState`
- `StartupEnvironment`

This file controls:

- public observations
- hidden market conditions
- delayed consequences
- action effects
- random external events
- reward calculation
- episode termination

If you change project behavior, this is the file most likely to matter.

### `agents.py`

This file defines the heuristic policy layer:

- `ActionProposal`
- `BaseCoFounder`
- `TechCoFounder`
- `GrowthCoFounder`
- `FinanceCoFounder`
- `CEO`
- `build_heuristic_agents()`

These are not learning agents. They are hand-written rules over noisy observations.

### `simulate.py`

This is the single-episode runner.

It:

- creates the agent stack
- resets the environment
- collects proposals
- asks the CEO to choose
- steps the environment
- logs results
- optionally prints verbose trace output
- optionally saves a summary JSON

If you want to understand the end-to-end loop, start here.

### `evaluation.py`

This is the multi-episode evaluation layer.

It:

- runs repeated episodes with different seeds
- computes aggregate metrics
- saves JSON and CSV outputs

Important current issue:

- As of the current code, `_save_step_csv()` writes full step rows from `simulate.py`, but those rows contain keys not present in the CSV schema. That causes `evaluation.py` to crash when saving step metrics.

### `llm_agents.py`

This is a scaffold for LLM-driven decision making.

It does not call a real model by itself.

It provides:

- prompt builders for each role
- a `generator` hook for injecting a model call
- parsing logic to extract valid actions
- safe fallback to heuristic behavior if generation is missing or invalid

This file is best viewed as an interface layer for future integration.

### `openenv_wrapper.py`

This is a small compatibility wrapper around `StartupEnvironment`.

It exposes a more training-style API:

- `reset()`
- `step(action)`
- `action_space`
- `render()`
- `observation_schema()`

This file is intentionally thin. It mostly repackages environment outputs.

### `train.py`

This is not training in the optimizer sense yet.

What it actually does:

- runs multiple episodes
- stores step logs and metadata as trajectories
- writes them to JSON

What it does not do:

- gradient updates
- policy optimization
- model fine-tuning

## 5. Runtime Flow

Here is the real runtime path for a normal simulation:

1. `simulate.py:run_episode()` starts an episode.
2. It calls `env.reset()`.
3. It builds either heuristic agents or prompted agents.
4. Each co-founder sees the current observation and returns an `ActionProposal`.
5. The CEO chooses one proposal.
6. `env.step()` applies that chosen action.
7. `environment.py` updates state through:
   - pending effects
   - immediate action effects
   - recurring environment dynamics
   - random events
   - reward computation
   - done check
8. `simulate.py` logs the step.
9. The loop repeats until horizon reached or environment says done.

## 6. Data Shapes You Should Know

### Public observation

Produced by `StartupEnvironment.get_observation()` in [environment.py](/home/techiester83/Desktop/Scaler_Hack_Finale/environment.py:70).

Fields:

- `day`
- `money`
- `users`
- `product_quality`
- `team_size`
- `burn_rate`
- `recent_user_growth`
- `ad_performance`
- `recent_actions`
- `recent_events`
- `runway_hint`

Important detail:

- `recent_user_growth` is noisy, not exact.
- `runway_hint` is also noisy.
- Agents do not directly observe hidden market variables.

### Hidden state

Defined in [environment.py](/home/techiester83/Desktop/Scaler_Hack_Finale/environment.py:6).

Fields:

- `market_demand`
- `competition_level`
- `economic_condition`

These values affect outcomes but are hidden from normal agent observations.

### Pending effects

Defined in [environment.py](/home/techiester83/Desktop/Scaler_Hack_Finale/environment.py:13).

This is how delayed consequences are modeled.

Examples:

- hiring boosts quality later
- product investment adds future quality boosts
- marketing adds later user decay
- pivoting shifts market demand later

### Step result

Returned by `StartupEnvironment.step()` in [environment.py](/home/techiester83/Desktop/Scaler_Hack_Finale/environment.py:112).

Fields:

- `state`
- `reward`
- `done`
- `action`
- `event`
- `agent_rewards`
- `debug_state`

## 7. `environment.py` Deep Dive

### `HiddenState`

Location: [environment.py](/home/techiester83/Desktop/Scaler_Hack_Finale/environment.py:6)

Role:

- Stores hidden world variables.
- These influence marketing success, churn, and general performance.

### `PendingEffect`

Location: [environment.py](/home/techiester83/Desktop/Scaler_Hack_Finale/environment.py:13)

Role:

- Represents a delayed future effect with a countdown (`eta`) and a payload.

### `StartupState`

Location: [environment.py](/home/techiester83/Desktop/Scaler_Hack_Finale/environment.py:20)

Role:

- Stores the public and bookkeeping state of the startup.

Important fields beyond the obvious ones:

- `last_users` and `last_money` are used for reward calculations.
- `last_ad_performance` is a feedback signal used by agents.
- `ignored_negative_trends` creates a penalty when the system keeps choosing weak reactions during decline.

### `StartupEnvironment.__init__`

Location: [environment.py](/home/techiester83/Desktop/Scaler_Hack_Finale/environment.py:54)

Role:

- Sets max horizon.
- Creates a seeded `random.Random`.
- Initializes state containers.
- Calls `reset()`.

Design note:

- Randomness is encapsulated in `self.random`, which makes runs reproducible per seed.

### `reset()`

Location: [environment.py](/home/techiester83/Desktop/Scaler_Hack_Finale/environment.py:63)

Role:

- Rebuilds startup state, hidden state, pending effects, and history.
- Returns the first observation.

### `get_observation()`

Location: [environment.py](/home/techiester83/Desktop/Scaler_Hack_Finale/environment.py:70)

Role:

- Converts full internal state into the partial/noisy view that agents receive.

Why it matters:

- This is the boundary between the “true world” and what agents can reason from.
- If you want richer observations or more/less uncertainty, start here.

### `get_debug_state()`

Location: [environment.py](/home/techiester83/Desktop/Scaler_Hack_Finale/environment.py:89)

Role:

- Returns public state, hidden state, and queued pending effects.
- Used for debugging and verbose inspection.

### `step(action, proposals=None)`

Location: [environment.py](/home/techiester83/Desktop/Scaler_Hack_Finale/environment.py:112)

This is the central transition function.

Internal order:

1. validate action
2. snapshot previous users and money
3. apply old delayed effects
4. apply the chosen action
5. apply recurring startup dynamics
6. sample a random event
7. compute reward
8. compute agent-specific rewards
9. check termination
10. increment day and append action to history
11. return observation plus metadata

Important implementation detail:

- Reward is computed before `day` increments, but after all state changes for the step are applied.

### `_apply_pending_effects()`

Location: [environment.py](/home/techiester83/Desktop/Scaler_Hack_Finale/environment.py:157)

Role:

- Counts down all queued delayed effects.
- Applies any whose `eta` reaches zero.
- Rebuilds the remaining queue.

Editing advice:

- If you add new delayed mechanics, you need changes in two places:
  - where the effect is enqueued
  - where the effect name is interpreted here

### `_apply_action(action)`

Location: [environment.py](/home/techiester83/Desktop/Scaler_Hack_Finale/environment.py:186)

This is where each action’s immediate logic lives.

Action behavior summary:

- `hire_employee`
  - costs money
  - increases team size and burn
  - queues a delayed productivity boost
- `fire_employee`
  - only works if `team_size > 1`
  - slightly restores money and lowers burn
  - slightly hurts quality
- `invest_in_product`
  - costs money
  - gives immediate quality gain
  - queues more future quality gains
- `run_marketing_campaign`
  - costs money
  - calculates probabilistic success from hidden state + quality
  - adds users immediately
  - queues future user decay
- `do_nothing`
  - only resets ad performance to `average`
- `pivot_strategy`
  - costs money
  - hurts quality immediately
  - queues a future market-demand boost

Editing advice:

- This is the place to change action economics and delayed effect scheduling.

### `_apply_environment_dynamics(action)`

Location: [environment.py](/home/techiester83/Desktop/Scaler_Hack_Finale/environment.py:257)

Role:

- Applies recurring revenue.
- Recomputes burn.
- subtracts an ongoing burn cost fraction from cash.
- Applies churn based on quality, competition, and economy.
- Randomly drifts hidden-state variables.
- Tracks whether the chosen action ignored negative trends.

Subtle point:

- `burn_rate` gets recalculated from team size here, so the earlier action-level burn adjustments are partly temporary setup for this recomputed value.

### `_sample_event()`

Location: [environment.py](/home/techiester83/Desktop/Scaler_Hack_Finale/environment.py:283)

Role:

- With fixed probability, applies one random external event.
- Otherwise logs `"none"`.

Event behaviors:

- `competitor_launch`: reduces users and raises competition
- `market_crash`: hurts market demand, economy, and money
- `viral_growth`: boosts users and market demand
- `tech_failure`: hurts quality and money

### `_compute_reward(action)`

Location: [environment.py](/home/techiester83/Desktop/Scaler_Hack_Finale/environment.py:308)

Reward ingredients:

- user growth term
- profit term
- product quality bonus
- burn penalty
- instability penalty
- negative trend penalty
- bankruptcy penalty

Important detail:

- This reward is shaped, not purely profit-based.
- It is trying to encourage strategic survivable growth, not just cash maximization.

### `_compute_agent_rewards(...)`

Location: [environment.py](/home/techiester83/Desktop/Scaler_Hack_Finale/environment.py:335)

Role:

- CEO gets the full company reward.
- Co-founders get full reward if their proposal matched the chosen action.
- Otherwise they get `20%` of that reward.

This is currently a simple alignment reward, not a truly separate agent objective.

### `_is_done()`

Location: [environment.py](/home/techiester83/Desktop/Scaler_Hack_Finale/environment.py:346)

Episode ends when:

- day reaches `max_days`
- money drops below `-25000`
- users drop to `0`

## 8. `agents.py` Deep Dive

### `ActionProposal`

Location: [agents.py](/home/techiester83/Desktop/Scaler_Hack_Finale/agents.py:6)

Simple container:

- `action`
- `reasoning`

This object is central because both heuristic and prompted agents return it.

### `BaseCoFounder`

Location: [agents.py](/home/techiester83/Desktop/Scaler_Hack_Finale/agents.py:12)

Acts as the base interface for specialized co-founders.

### `TechCoFounder.propose()`

Location: [agents.py](/home/techiester83/Desktop/Scaler_Hack_Finale/agents.py:24)

Main bias:

- protect/improve product quality

Behavior:

- chooses `invest_in_product` if quality is low or after `tech_failure`
- also chooses `invest_in_product` when growth is negative and quality is not yet strong
- otherwise hires

### `GrowthCoFounder.propose()`

Location: [agents.py](/home/techiester83/Desktop/Scaler_Hack_Finale/agents.py:50)

Main bias:

- pursue user growth and momentum

Behavior:

- doubles down on marketing after `viral_growth`
- usually runs marketing when user base is still small or growth signal is soft
- pivots when current growth/ad signals look weak

### `FinanceCoFounder.propose()`

Location: [agents.py](/home/techiester83/Desktop/Scaler_Hack_Finale/agents.py:77)

Main bias:

- preserve runway and contain burn

Behavior:

- fires when cash or runway is tight
- pauses with `do_nothing` if burn is high and growth is negative
- otherwise is willing to hire

### `CEO.choose_action()`

Location: [agents.py](/home/techiester83/Desktop/Scaler_Hack_Finale/agents.py:104)

This is a policy over proposals, not a direct policy over raw actions.

Decision order:

1. If money/runway is critical, trust Finance.
2. If product quality or tech risk is bad, trust Tech.
3. If growth momentum is strong, trust Growth.
4. Otherwise score all proposals and pick the best.

This means the CEO acts partly as:

- a crisis override layer
- a tiebreak/scoring layer

### `CEO._score_proposal()`

Location: [agents.py](/home/techiester83/Desktop/Scaler_Hack_Finale/agents.py:120)

This is the default arbitration heuristic.

If you feel the CEO behaves oddly, this method is one of the first places to inspect.

### `build_heuristic_agents()`

Location: [agents.py](/home/techiester83/Desktop/Scaler_Hack_Finale/agents.py:141)

Convenience constructor for the standard four-agent stack.

## 9. `simulate.py` Deep Dive

### `run_episode(...)`

Location: [simulate.py](/home/techiester83/Desktop/Scaler_Hack_Finale/simulate.py:10)

This is the main driver of a single episode.

Key responsibilities:

- reset environment
- build agents
- ask each co-founder for a proposal
- let the CEO choose
- call `env.step()`
- accumulate reward
- build a structured `episode_log`
- optionally print a detailed trace

Important note:

- The logged row stores more than just metrics. It also stores full observation snapshots, proposals, and chosen reasoning.
- That richer structure is useful for analysis and future training, but it is also the reason `evaluation.py` currently crashes when dumping step CSV rows without filtering fields.

### `_build_agent_stack(agent_mode)`

Location: [simulate.py](/home/techiester83/Desktop/Scaler_Hack_Finale/simulate.py:105)

Behavior:

- `"prompt_scaffold"` -> builds prompted agents
- anything else currently defaults to heuristic agents

### `_collect_prompt_debug(...)`

Location: [simulate.py](/home/techiester83/Desktop/Scaler_Hack_Finale/simulate.py:111)

Role:

- Pulls prompt/response artifacts off agents when available.
- Makes prompt debugging part of the episode log.

This is especially useful once a real generator is wired in.

### `main()`

Location: [simulate.py](/home/techiester83/Desktop/Scaler_Hack_Finale/simulate.py:126)

CLI entrypoint.

Flags:

- `--horizon`
- `--seed`
- `--quiet`
- `--show-hidden-state`
- `--agent-mode`
- `--save-summary`

## 10. `evaluation.py` Deep Dive

### `evaluate(...)`

Location: [evaluation.py](/home/techiester83/Desktop/Scaler_Hack_Finale/evaluation.py:12)

Role:

- Runs many episodes with incremented seeds.
- Builds aggregate metrics.
- Optionally saves JSON and CSV artifacts.

### `_build_aggregate_metrics(...)`

Location: [evaluation.py](/home/techiester83/Desktop/Scaler_Hack_Finale/evaluation.py:55)

Computed metrics:

- average total reward
- average final money
- average final users
- survival rate
- growth consistency
- decision efficiency
- best and worst reward

These are simple summary metrics, not deep statistical analysis.

### `_positive_reward_ratio(...)`

Location: [evaluation.py](/home/techiester83/Desktop/Scaler_Hack_Finale/evaluation.py:82)

Meaning:

- fraction of steps whose reward is positive

### `_growth_consistency(...)`

Location: [evaluation.py](/home/techiester83/Desktop/Scaler_Hack_Finale/evaluation.py:89)

Meaning:

- fraction of consecutive steps where user count does not drop below `85%` of the previous step

### `_save_json(...)`

Location: [evaluation.py](/home/techiester83/Desktop/Scaler_Hack_Finale/evaluation.py:99)

Straight JSON serializer for the evaluation payload.

### `_save_episode_csv(...)`

Location: [evaluation.py](/home/techiester83/Desktop/Scaler_Hack_Finale/evaluation.py:104)

Writes per-episode summary rows.

This part is fine structurally.

### `_save_step_csv(...)`

Location: [evaluation.py](/home/techiester83/Desktop/Scaler_Hack_Finale/evaluation.py:131)

Current problem:

- It writes `{"episode_index": ..., **row}` for each step.
- But `row` contains extra keys like `observation`, `proposals`, and `chosen_reasoning`.
- `csv.DictWriter` rejects keys not in `fieldnames`.

If you fix evaluation later, the safest fix is to explicitly project each row down to only the CSV fields.

### `main()`

Location: [evaluation.py](/home/techiester83/Desktop/Scaler_Hack_Finale/evaluation.py:152)

CLI entrypoint for batch evaluation.

## 11. `llm_agents.py` Deep Dive

This file reuses the same decision structure but swaps direct heuristics for prompt construction plus parsing.

### `ActionGenerator`

Location: [llm_agents.py](/home/techiester83/Desktop/Scaler_Hack_Finale/llm_agents.py:9)

This is the expected callable shape for a model backend:

- input: prompt string
- output: raw text string

### `PromptArtifacts`

Location: [llm_agents.py](/home/techiester83/Desktop/Scaler_Hack_Finale/llm_agents.py:12)

Stores:

- `system_prompt`
- `user_prompt`

### `PromptedAgentMixin`

Location: [llm_agents.py](/home/techiester83/Desktop/Scaler_Hack_Finale/llm_agents.py:18)

This is the main shared behavior for prompted co-founders.

Flow inside `propose()`:

1. ask fallback heuristic agent for a safe action
2. build prompt
3. if no generator exists, return fallback
4. otherwise generate text
5. parse action from text
6. if parsing fails, return fallback

This makes the scaffold robust even before real model integration is stable.

### `PromptedTechCoFounder`, `PromptedGrowthCoFounder`, `PromptedFinanceCoFounder`

Location:

- [llm_agents.py](/home/techiester83/Desktop/Scaler_Hack_Finale/llm_agents.py:58)
- [llm_agents.py](/home/techiester83/Desktop/Scaler_Hack_Finale/llm_agents.py:77)
- [llm_agents.py](/home/techiester83/Desktop/Scaler_Hack_Finale/llm_agents.py:96)

These classes only specialize role framing. The fallback and parsing logic are inherited.

### `PromptedCEO`

Location: [llm_agents.py](/home/techiester83/Desktop/Scaler_Hack_Finale/llm_agents.py:115)

This mirrors the CEO flow:

- compute heuristic fallback
- build a prompt with all proposals
- optionally call generator
- parse action
- fall back if invalid

### `build_prompted_agents(...)`

Location: [llm_agents.py](/home/techiester83/Desktop/Scaler_Hack_Finale/llm_agents.py:171)

Factory for the full prompted stack.

### `parse_action(text, allowed_actions)`

Location: [llm_agents.py](/home/techiester83/Desktop/Scaler_Hack_Finale/llm_agents.py:182)

Behavior:

- first looks for `Action: <action_name>`
- then falls back to scanning tokens

This is intentionally forgiving so the system can tolerate mildly messy model outputs.

### `_format_role_prompt(...)` and `_format_observation(...)`

Location:

- [llm_agents.py](/home/techiester83/Desktop/Scaler_Hack_Finale/llm_agents.py:199)
- [llm_agents.py](/home/techiester83/Desktop/Scaler_Hack_Finale/llm_agents.py:208)

These are pure prompt-format helpers.

## 12. `openenv_wrapper.py` Deep Dive

Location: [openenv_wrapper.py](/home/techiester83/Desktop/Scaler_Hack_Finale/openenv_wrapper.py:6)

This file is intentionally small.

Important thing to notice:

- `step(action)` calls the raw environment directly, with no proposal context.
- That means this wrapper is designed for single-policy or direct-action training interfaces, not for preserving the multi-agent proposal stage as-is.

If you later train a centralized policy, this is fine.
If you later want true multi-agent training, this wrapper will likely need redesign.

## 13. `train.py` Deep Dive

### `collect_trajectories(...)`

Location: [train.py](/home/techiester83/Desktop/Scaler_Hack_Finale/train.py:10)

Role:

- runs many episodes
- stores summary plus full step logs under `steps`

This is basically data collection, not learning.

### `save_trajectories(...)`

Location: [train.py](/home/techiester83/Desktop/Scaler_Hack_Finale/train.py:41)

Writes trajectory JSON to disk.

### `main()`

Location: [train.py](/home/techiester83/Desktop/Scaler_Hack_Finale/train.py:47)

CLI entrypoint for trajectory collection.

## 14. Important Cross-File Relationships

These are the main couplings worth remembering:

- `simulate.py` depends on both `agents.py` and `environment.py`
- `evaluation.py` and `train.py` both depend on `simulate.py`
- `llm_agents.py` depends on both `agents.py` and `environment.py`
- `openenv_wrapper.py` bypasses `simulate.py` and talks to `environment.py` directly

Practical consequence:

- If you change observation fields in `environment.py`, you may need to update:
  - heuristic agents
  - prompt formatting
  - wrappers
  - downstream analysis

## 15. Real Behavior Observed In A Quick Run

I ran a short 5-step simulation.

What stood out:

- The system frequently favored `invest_in_product` early because starting quality is low and the CEO heavily trusts Tech when quality is weak.
- Product quality rose quickly.
- Users still declined under churn/events.
- Total reward over that short run was negative.

Interpretation:

- The current system seems biased toward early product stabilization over immediate growth.
- That may be intended, but it is something to watch if you tune the heuristics or reward.

## 16. Current Risks / Surprises

### Evaluation script currently breaks

Source:

- [evaluation.py](/home/techiester83/Desktop/Scaler_Hack_Finale/evaluation.py:131)
- [simulate.py](/home/techiester83/Desktop/Scaler_Hack_Finale/simulate.py:34)

Why:

- logged rows are richer than the CSV schema expects

### Burn rate is both action-modified and then recomputed

Source:

- [environment.py](/home/techiester83/Desktop/Scaler_Hack_Finale/environment.py:186)
- [environment.py](/home/techiester83/Desktop/Scaler_Hack_Finale/environment.py:257)

Why it matters:

- It is easy to assume action-level burn adjustments persist exactly, but the recurring dynamics recalculate burn from team size.

### `do_nothing` does very little

Source:

- [environment.py](/home/techiester83/Desktop/Scaler_Hack_Finale/environment.py:243)

Why it matters:

- If you expected “pause and conserve cash” to have a large economic meaning, it currently does not beyond interacting with the rest of the step logic.

### The OpenEnv wrapper is not preserving multi-agent proposal semantics

Source:

- [openenv_wrapper.py](/home/techiester83/Desktop/Scaler_Hack_Finale/openenv_wrapper.py:24)

Why it matters:

- Training through this wrapper means learning direct environment actions, not reproducing the co-founder proposal process.

## 17. Where To Edit Depending On Your Goal

If your goal is to change startup dynamics:

- edit `environment.py`

If your goal is to change role personalities or disagreement:

- edit `agents.py`

If your goal is to change episode logging or CLI flow:

- edit `simulate.py`

If your goal is to change aggregate metrics or exports:

- edit `evaluation.py`

If your goal is to connect a real LLM backend:

- edit `llm_agents.py`

If your goal is to collect different training data:

- edit `train.py`

If your goal is to support external RL tooling:

- edit `openenv_wrapper.py`

## 18. Best First Changes For A New Contributor

If you want safe starter tasks, these are strong options:

1. Fix `evaluation.py` step CSV export.
2. Add tests for environment transitions and reward behavior.
3. Add configuration knobs for action costs and reward weights.
4. Improve the heuristic disagreement quality in `agents.py`.
5. Separate logging schemas for “rich step log” vs “flat CSV row”.

## 19. Final Takeaway

This codebase is small enough to understand fully in one sitting.

The most important thing to remember is that the project is organized around one central loop:

- agents reason from noisy observations
- the CEO chooses
- the environment applies action + hidden dynamics + events + delayed effects
- reward is computed from the resulting business state

If you keep that loop in mind, almost every file becomes easier to understand.
