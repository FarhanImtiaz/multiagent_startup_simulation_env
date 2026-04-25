# Temporary Implementation Checklist

This is a temporary working checklist derived from `Project_Overview.md`.
Use it as the running build sheet while the project is being implemented.

## Status Key

- `[x]` Done
- `[-]` Partially done
- `[ ]` Not started

## 1. Project Foundation

- [x] Define the project goal clearly
  - Multi-agent startup simulator
  - Long-horizon decision environment
  - Professional world modeling setup
- [x] Keep the repo aligned with `Project_Overview.md`
- [x] Keep temporary implementation notes up to date

## 2. Environment Core

- [x] Create a startup environment class
- [x] Support reset and step flow
- [x] Represent the startup over discrete timesteps
- [x] Maintain core public state
  - [x] `day`
  - [x] `money`
  - [x] `users`
  - [x] `product_quality`
  - [x] `team_size`
  - [x] `burn_rate`
  - [x] `recent_actions`
- [x] Define high-level action space
  - [x] `hire_employee`
  - [x] `fire_employee`
  - [x] `invest_in_product`
  - [x] `run_marketing_campaign`
  - [x] `do_nothing`
  - [x] `pivot_strategy`
- [x] Implement action transition logic
- [x] Apply recurring environment dynamics
- [x] Add episode termination conditions
- [-] Tune transition coefficients for better balance
- [ ] Add configuration knobs for easy experimentation

## 3. Advanced Environment Dynamics

- [x] Add hidden state
  - [x] `market_demand`
  - [x] `competition_level`
  - [x] `economic_condition`
- [x] Keep hidden state unavailable to agents during normal play
- [x] Add partial observability
  - [x] noisy user growth signal
  - [x] ad performance signal
  - [x] runway hint
  - [x] recent event hints
- [x] Add stochastic action outcomes
  - [x] probabilistic marketing success
- [x] Add unpredictable external events
  - [x] `competitor_launch`
  - [x] `market_crash`
  - [x] `viral_growth`
  - [x] `tech_failure`
- [x] Make events occur probabilistically
- [x] Add delayed consequences queue
  - [x] hiring delayed productivity
  - [x] product investment delayed quality gain
  - [x] marketing delayed decay
  - [x] pivot delayed demand shift
- [x] Support divergent beliefs by exposing only noisy observations
- [-] Improve event diversity and realism
- [ ] Add more scenario types
  - [ ] pricing shock
  - [ ] investor pressure
  - [ ] infrastructure outage
  - [ ] regulation change
- [ ] Add difficulty scaling across episodes
- [ ] Add longer-horizon scenario presets

## 4. Reward System

- [x] Implement company-level reward
- [x] Include long-term growth term
- [x] Include profit term
- [x] Include product quality term
- [x] Include burn penalty
- [x] Include bankruptcy penalty
- [x] Include instability penalty
- [x] Include ignoring-negative-trend penalty
- [x] Add agent-specific rewards
  - [x] CEO gets company reward
  - [x] co-founders get alignment-based reward
- [-] Validate whether reward weights create useful behavior
- [ ] Add reward ablation / comparison helpers

## 5. Multi-Agent System

- [x] Create specialized co-founder roles
  - [x] Tech
  - [x] Growth
  - [x] Finance
- [x] Create CEO role
- [x] Each co-founder proposes an action
- [x] Each proposal includes reasoning
- [x] CEO chooses final action
- [x] Make agent choices depend on imperfect observations
- [-] Improve disagreement quality between agents
- [ ] Add richer proposal formatting for training data
- [ ] Add memory / belief tracking per agent

## 6. Simulation Loop

- [x] Run a full episode
- [x] Reset environment before episode
- [x] Gather proposals from all co-founders
- [x] Let CEO select final action
- [x] Step environment with chosen action
- [x] Accumulate reward over episode
- [x] Stop on termination
- [x] Print step-by-step logs
- [x] Add compact readable simulation logs
- [x] Preserve full verbose logs behind a CLI flag
- [x] Return structured episode summary
- [x] Add multi-episode runner
- [x] Add command-line options for seeds, horizon, and verbosity

## 7. Logging and Metrics

- [x] Keep in-memory episode history
- [x] Track action, reward, event, money, users, and quality per step
- [x] Produce structured episode logs suitable for analysis
- [x] Save metrics to JSON
- [x] Save metrics to CSV
- [x] Compute total reward over many episodes
- [x] Compute survival rate
- [x] Compute growth consistency
- [x] Compute decision efficiency
- [x] Add baseline metric summaries
- [x] Add reward and outcome plots
- [x] Add action distribution export
- [x] Add baseline Markdown report

## 8. Evaluation Pipeline

- [x] Create a repeatable baseline evaluation script
- [x] Run many episodes with heuristic agents
- [x] Aggregate baseline results
- [x] Compare behavior across random seeds
- [-] Identify common failure modes
- [x] Produce judge-friendly summary outputs
- [x] Fix step-level CSV export mismatch
- [x] Add flat evaluation exports for analysis

## 9. OpenEnv / Training Interface

- [x] Create an OpenEnv-compatible wrapper
- [x] Standardize observation output for training
- [x] Standardize action output for training
- [x] Standardize reward and done signals
- [x] Expose trajectory collection hooks
- [x] Ensure compatibility with TRL / Unsloth pipeline expectations

## 10. LLM Agent Layer

- [-] Replace heuristic agents with LLM-driven agents
- [x] Write prompt template for CEO
- [x] Write prompt template for Tech co-founder
- [x] Write prompt template for Growth co-founder
- [x] Write prompt template for Finance co-founder
- [x] Constrain outputs to valid actions
- [x] Parse model responses robustly
- [x] Add fallback handling for invalid generations
- [x] Keep heuristic agents as baseline mode

## 11. Training Loop

- [x] Create training script
- [x] Run episodes to collect trajectories
- [x] Store prompts, actions, outcomes, rewards
- [x] Optimize policy using rewards
- [-] Support repeated training cycles
- [x] Save trajectory outputs for future training
- [x] Export SFT-style JSONL records
- [x] Export preference-style JSONL records
- [x] Document how to reproduce training

## 12. Before vs After Comparison

- [x] Run untrained or heuristic baseline
- [x] Run trained policy evaluation
- [x] Compare reward trends
- [x] Compare user growth
- [x] Compare profit / cash survival
- [-] Compare strategy quality qualitatively
- [x] Produce a simple graph or table for judges

## 13. Hackathon Readiness

- [x] README reflects the current state
- [x] Clean `.gitignore`
- [x] Remove temporary dev clutter from git status
- [-] Package the project cleanly
- [-] Add Colab notebook for training/demo
- [-] Prepare HuggingFace Spaces version
- [ ] Prepare short explainer flow

## 14. Automated Round Checks

These are the items the validation pass looks for. If any is missing or broken at the deadline, the submission will not make it to a human judge, regardless of how strong the underlying idea is. Verify each one explicitly before submitting.

- [-] Public, cloneable Hugging Face Space at the submitted URL
  - [ ] Test from a logged-out browser
  - [ ] Confirm the Space is not private
  - [ ] Confirm the submitted link is not dead or returning 404
- [-] Valid OpenEnv structure
  - [x] Proper `Environment` / `MCPEnvironment` base class
  - [x] Gym-style `reset`, `step`, and `state`
  - [x] Parseable `openenv.yaml`
- [x] Training evidence committed to the repo as image files
  - [x] Loss curve committed as `.png` or `.jpg`
  - [x] Reward curve committed as `.png` or `.jpg`
  - [x] Confirm plots are not only WandB links or Colab cell outputs
- [-] Runnable training script
  - [x] Unsloth, Hugging Face TRL, or another framework is acceptable
  - [ ] Prefer a linked Colab notebook that can be re-executed end to end
  - [x] Python script is acceptable if the full training flow is runnable
- [-] README links every deliverable
  - [ ] Hugging Face Space
  - [ ] Training notebook or script
  - [ ] Writeup, blog, video, or slides
  - [x] Key plots embedded inline
  - [ ] Confirm every deliverable is reachable from the README

## 15. Demo and Pitch

- [ ] Show environment state evolving
- [ ] Show co-founder proposals
- [ ] Show CEO action selection
- [ ] Show random bad events
- [ ] Show delayed effects
- [ ] Show baseline failure cases
- [x] Show improved behavior after training
- [ ] Prepare concise explanation of
  - [ ] problem
  - [ ] environment
  - [ ] agents
  - [ ] reward
  - [ ] results

## 16. Immediate Next Build Order

- [x] Create multi-episode evaluation script
- [x] Add saved logs and aggregate metrics
- [x] Clean `.gitignore`
- [x] Add OpenEnv-compatible wrapper
- [x] Add LLM prompt-based agent interface
- [x] Add training pipeline scaffold
- [x] Phase 2: repair evaluation exports
- [x] Phase 2: add baseline report and lightweight plots
- [x] Phase 2: make simulation logs easier to read
- [x] Phase 2: document evaluation artifacts in README
- [x] Slim `agents.py` back to the maintainable heuristic baseline
- [x] Add training-ready SFT and preference dataset exports
- [x] Phase 3: add OpenEnv-style package and `openenv.yaml`
- [x] Phase 3: add trained CEO mode, safety gate, and before/after metrics
- [x] Phase 3: commit PNG loss and reward artifacts
- [x] Phase 3: add Dockerfile and Colab notebook skeleton
- [x] Phase 4: add Gradio Space demo
- [x] Phase 4: add `compare_policies.py`
- [x] Phase 4: add OpenEnv validation helper

## 17. Files To Touch Next

- `evaluation.py`
  - optionally add a combined before/after comparison CLI
- `agents.py`
  - continue survival and cash behavior tuning
- `train.py`
  - keep Colab commands synchronized with the final notebook
- `README.md`
  - document new Phase 2 evaluation artifacts
- `llm_agents.py`
  - keep trained CEO backend documented and smoke-tested
- `mass_startup_env/`
  - validate against the latest OpenEnv CLI and HF Space runtime
- `TEMP_IMPLEMENTATION_CHECKLIST.md`
  - update statuses as work progresses

## 18. Current Snapshot

- Environment core: strong prototype
- Advanced uncertainty features: implemented first pass
- Multi-agent proposal flow: implemented
- Evaluation pipeline: Phase 2 baseline reporting implemented
- LLM integration: trained CEO path implemented with safety gate
- Training: TRL SFT script and Colab-tested CEO training complete
- OpenEnv packaging: manifest and package scaffold added; needs final CLI/Space validation
- Demo packaging: README and plots started; HF Space, Colab link, and video/blog still missing
