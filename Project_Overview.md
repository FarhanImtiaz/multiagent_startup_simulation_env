# 🚀 Project: Multi-Agent Startup Simulator (MASS)

## 🧠 Overview

We are building a **multi-agent reinforcement learning environment** that simulates a startup company operated by AI agents over a long time horizon.

This project combines:

- Multi-Agent Interactions
- Long-Horizon Planning
- World Modeling (Professional Tasks)

The goal is to create a structured environment where multiple AI agents (CEO + Co-founders) interact, make decisions, and learn to optimize business outcomes over time.

---

## 🎯 Problem Statement

### 🚀 Multi-Agent Startup Decision Intelligence Environment

Modern AI systems perform well on short, isolated tasks but struggle with coordinated decision-making in dynamic, multi-agent environments over extended time horizons.

In real-world scenarios such as startups and businesses, decisions are:

- Interdependent (multiple stakeholders with different goals)
- Sequential (early decisions impact future outcomes)
- Uncertain (outcomes are delayed and partially observable)

However, current LLM training setups lack environments that:

- Simulate conflicting incentives between multiple agents
- Require long-term planning under resource constraints
- Capture real-world trade-offs between growth, cost, and product quality

### 🎯 Problem We Are Solving

We aim to design a structured, interactive simulation environment where:

- Multiple AI agents (CEO and specialized co-founders)
- Operate within a shared startup environment
- Make sequential decisions under limited resources
- Balance competing objectives (growth vs cost vs quality)
- Experience delayed rewards and consequences

The goal is to enable AI agents to:

- Learn strategic coordination
- Develop long-horizon planning capabilities
- Improve decision-making under uncertainty

### 🧠 Why This Matters

Real-world AI applications require:

- Multi-step reasoning
- Collaboration between agents
- Adaptation to evolving environments

By simulating a startup ecosystem, we create a controlled yet realistic training ground that pushes beyond:

- Single-step reasoning
- Static benchmarks
- Isolated task performance

### 🔬 Expected Outcome

We expect that training agents in this environment will lead to:

- Improved decision consistency over long time horizons
- Emergence of strategic behavior across agents
- Better handling of trade-offs and resource constraints
- Observable reward improvement through training

### 🧩 Theme Alignment

This problem directly aligns with:

- Multi-Agent Interactions → CEO + co-founders with competing objectives
- Long-Horizon Planning → decisions impact future states over many steps
- World Modeling (Professional Tasks) → realistic business simulation

### 🎤 Short Version (for pitch)

If they ask quickly:

> “We’re building a multi-agent startup simulation where AI agents with different roles collaborate and compete over long time horizons, learning to make strategic decisions under resource constraints and delayed rewards.”

---

## 🌍 Environment Design

### Core Idea

The environment represents a **startup company evolving over discrete time steps (days/weeks)**.

At each step:

1. The environment provides the current state.
2. Co-founder agents propose actions.
3. CEO agent selects a final action.
4. Environment updates state based on action.
5. Rewards are assigned.

---

## 🧱 Environment State

```python
state = {
    "day": int,
    "money": float,
    "users": int,
    "product_quality": float,
    "team_size": int,
    "burn_rate": float,
    "market_demand": float,
    "recent_actions": list,
}
```

## ⚙️ Actions

### High-Level Actions

- hire_employee
- fire_employee
- invest_in_product
- run_marketing_campaign
- do_nothing
- pivot_strategy

## 🔄 Transition Logic

Each action updates the state.

Examples:

- Hiring → increases burn_rate, may improve output
- Marketing → increases users but costs money
- Product investment → improves quality slowly
- Bad decisions → long-term penalties

## 🤖 Agents

### 👑 CEO Agent

- Final decision maker
- Selects one action from proposals
- Optimizes long-term company success

### 🧑‍💻 Co-Founder Agents (2–3)

Each agent specializes:

#### Tech Co-founder

- Focus: product_quality
- Prefers: investing in product

#### Growth Co-founder

- Focus: users
- Prefers: marketing actions

#### Finance Co-founder

- Focus: money/burn_rate
- Prefers: cost-saving actions

## 🧩 Interaction Flow

1. Environment sends state → all agents.
2. Each co-founder proposes an action + reasoning.
3. CEO selects final action.
4. Environment executes action.
5. Reward is computed.

## 📋 Tasks

Agents must learn to:

- Balance growth vs sustainability
- Allocate limited resources
- Recover from bad decisions
- Plan across long horizons

## 🏆 Reward Model

### Global Reward (Company-Level)

```python
reward = (
    alpha * revenue_growth +
    beta * user_growth +
    gamma * product_quality -
    delta * burn_rate -
    penalty_if_bankrupt
)
```

### Agent-Specific Rewards

- CEO → total company reward
- Co-founders → reward if their proposal aligns with good outcomes

### Key Idea

Rewards are:

- Delayed
- Multi-objective
- Non-trivial

## 📈 Training Setup

We will:

- Use OpenEnv-compatible environment
- Train using TRL / Unsloth

### Training Loop

1. Initialize agent (LLM).
2. Run episodes (startup lifecycle).
3. Collect trajectories.
4. Optimize policy using rewards.
5. Repeat.

### Baseline vs Trained Comparison

We will show:

| Metric | Before Training | After Training |
| --- | --- | --- |
| Profit | Low | Higher |
| Users | Slow growth | Faster growth |
| Decisions | Random | Strategic |

## 🔁 Post-Training / Self-Improvement

We introduce:

- Increasing difficulty (harder market conditions)
- Random events (crashes, competition)
- Longer time horizons

Agents improve through:

- repeated simulation
- exposure to diverse scenarios

## 🧪 Evaluation Metrics

- Total reward over episodes
- Survival rate (avoid bankruptcy)
- Growth consistency
- Decision efficiency

## 🧩 Alignment with Hackathon Themes

- **✅ Multi-Agent Interactions:** CEO + Co-founders with conflicting goals
- **✅ Long-Horizon Planning:** Decisions impact future states over many steps
- **✅ World Modeling:** Realistic business simulation with constraints

## 🛠️ Technical Stack

| Component | Technology |
| --- | --- |
| Environment | OpenEnv |
| Training | HuggingFace TRL / Unsloth |
| Hosting | HuggingFace Spaces |
| Model | LLM (instruction-tuned) |

## 📦 Deliverables

- OpenEnv-compatible environment
- Training script (Colab)
- Reward function implementation
- Demo showing improvement
- Short explainer video (<2 min)

## 🎤 Pitch Summary

We built a multi-agent startup simulation where specialized AI co-founders propose strategies and a CEO agent makes decisions over long time horizons. The system learns to optimize business outcomes through structured interaction, delayed rewards, and evolving constraints.

## ⚡ Key Differentiators

- Multi-agent strategic reasoning
- Long-horizon decision-making
- Realistic world simulation
- Clear measurable learning improvement

---

## Phase-Wise PLAN

### 🚀 PHASE 0 — Understand the Core Idea (1–2 hours)

Before coding anything, lock this in:

👉 You are building:

> A simulation environment where decisions → consequences → rewards → learning

#### What to understand

- State = current situation of startup
- Action = decision taken
- Reward = how good that decision was
- Episode = one full startup lifecycle

#### Output of this phase

You can explain your project in 2–3 lines without confusion.

### 🧱 PHASE 1 — Build the Environment Core (MOST IMPORTANT)

⏱️ Time: 6–10 hours

This is your 80% scoring component.

#### Step 1: Define State

Start simple:

```python
state = {
    "money": 100000,
    "users": 100,
    "product_quality": 0.5,
    "team_size": 2,
    "day": 1,
}
```

What you’re learning:

👉 How systems represent real-world problems numerically

#### Step 2: Define Actions

```python
actions = [
    "hire",
    "fire",
    "build_product",
    "run_ads",
    "do_nothing",
]
```

Learning:

👉 How decisions are abstracted for AI

#### Step 3: Transition Function (MOST CRUCIAL)

```python
def step(state, action):
    if action == "hire":
        state["money"] -= 10000
        state["team_size"] += 1

    elif action == "run_ads":
        state["money"] -= 5000
        state["users"] += 50

    return state
```

Learning:

👉 Cause → Effect modeling (this is “world modeling”)

#### Step 4: Reward Function

```python
def reward(state):
    return state["users"] * 0.1 + state["money"] * 0.001
```

Learning:

👉 How AI defines “success”

#### Step 5: Episode Loop

```python
for day in range(30):
    action = agent(state)
    state = step(state, action)
    r = reward(state)
```

#### ✅ End of Phase 1

- You have a working simulation
- No agents yet — just logic

### 🤖 PHASE 2 — Add Multi-Agent System

⏱️ Time: 4–6 hours

#### Step 1: Create Roles

Instead of 1 agent → now:

- Tech agent
- Growth agent
- Finance agent
- CEO agent

#### Step 2: Proposal System

Each agent suggests an action:

```python
def tech_agent(state):
    return "build_product"

def growth_agent(state):
    return "run_ads"
```

#### Step 3: CEO Decision

```python
def ceo(proposals, state):
    return choose_best(proposals)
```

Learning:

👉 Multi-agent interaction + conflicting goals

#### ✅ End of Phase 2

- Agents interact
- CEO selects decisions

### 🧠 PHASE 3 — Plug in LLM (Agent Brain)

⏱️ Time: 4–8 hours

Replace hardcoded agents with LLM.

#### Prompt example

```text
You are a startup CEO.

State:
Money: 50000
Users: 200
Product Quality: 0.4

Choose ONE action:
- hire
- run_ads
- build_product

Output:
Action: build_product
```

Learning:

👉 How LLMs act as decision-makers

#### ✅ End of Phase 3

AI is now making decisions.

### 📈 PHASE 4 — Training + Improvement (CRITICAL FOR JUDGING)

⏱️ Time: 6–10 hours (can finalize onsite)

#### Step 1: Run baseline

- Random / untrained LLM
- Record rewards

#### Step 2: Train using

- TRL / Unsloth

#### Step 3: Compare

Show:

- reward graph 📈
- better decisions
- fewer failures

Learning:

👉 Reinforcement Learning fundamentals

#### ✅ End of Phase 4

You satisfy “reward improvement” criteria.

### 🌍 PHASE 5 — Make It “Hackathon Ready”

⏱️ Time: 3–5 hours

#### 1. OpenEnv Integration

Wrap your environment in required format.

#### 2. HuggingFace Spaces

Host simulation.

#### 3. Colab Notebook

Show training pipeline.

#### 4. Add Logs / Metrics

- reward per episode
- decisions made

### 🎤 PHASE 6 — Demo + Pitch (DON’T UNDERESTIMATE)

⏱️ Time: 2–3 hours

#### Your demo flow

- Show environment
- Show agents interacting
- Show bad decisions (before training)
- Show improvement (after training)

#### Your explanation

- Problem
- Environment
- Agents
- Reward
- Results

## ⚡ Suggested Timeline (Realistic)

| Day | Goal |
| --- | --- |
| Day 1 | Phase 0 + Phase 1 |
| Day 2 | Phase 2 + Phase 3 |
| Day 3 | Phase 4 (basic training) |
| Day 4 | Phase 5 + Demo |

## 🧠 Final mindset shift

Don’t think:

> “I’m building an AI project”

Think:

> “I’m designing a world where intelligence is required to succeed”

# 🌐 Advanced Environment Dynamics: Uncertainty & Partial Observability

## 🧠 Motivation

Real-world decision-making environments are inherently uncertain, partially observable, and dynamic. To better simulate realistic conditions and push agent intelligence, we extend our environment with hidden variables, stochastic outcomes, and external events.

This upgrade transforms the environment from deterministic decision-making to **adaptive reasoning under uncertainty**, significantly increasing its complexity and alignment with real-world scenarios.

---

## 🔍 Hidden State (Unobserved Variables)

We introduce internal environment variables that are **not directly visible to agents**:

```python
hidden_state = {
    "market_demand": float,   # Range: [0, 1]
    "competition_level": float,
    "economic_condition": float
}
Key Idea:

Agents must infer these values indirectly through observed outcomes (e.g., user growth, campaign success).

👁️ Partial Observability

Agents do not receive full state information. Instead, they observe noisy and indirect signals:

observed_state = {
    "money": float,
    "users": int,
    "product_quality": float,
    "recent_user_growth": float,
    "ad_performance": str,  # "good", "average", "poor"
    "team_size": int
}
Implication:

Agents must build internal beliefs about the environment over time.

⚡ Stochastic Action Outcomes

Actions no longer produce fixed results.

if action == "run_marketing_campaign":
    success_prob = hidden_state["market_demand"]
    if random() < success_prob:
        users += high_gain
    else:
        users += low_gain
Impact:
Same action → different outcomes
Encourages risk-aware decision making
🌪️ External Events System

We introduce a dynamic event engine that triggers unpredictable disruptions:

events = [
    "competitor_launch",
    "market_crash",
    "viral_growth",
    "tech_failure"
]
Example Effects:
competitor_launch → decrease in users
market_crash → reduced demand globally
viral_growth → sudden spike in users
tech_failure → drop in product quality

Events occur probabilistically at each timestep.

⏳ Delayed Consequences

Certain actions have delayed effects:

Hiring → immediate cost, delayed productivity gain
Product improvements → gradual quality increase over time
Marketing → short-term gains, long-term decay
Implementation Concept:

Use a queue or buffer system to apply effects after N steps.

🧑‍🤝‍🧑 Divergent Agent Beliefs

Due to partial observability, agents may interpret the same signals differently:

Growth agent may interpret poor performance as temporary noise
Finance agent may interpret it as systemic failure

This creates meaningful disagreement, requiring the CEO to reason over conflicting inputs.

🏆 Updated Reward Model

The reward function incorporates stability, adaptability, and long-term performance:

reward = (
    0.4 * long_term_user_growth +
    0.3 * profit +
    0.2 * product_quality -
    0.2 * burn_rate -
    instability_penalty -
    bankruptcy_penalty
)
Additional Penalties:
Frequent strategy switching
Ignoring negative trends
Bankruptcy events