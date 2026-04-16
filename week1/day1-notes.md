# Day 1: Introduction to Reinforcement Learning

## RL vs Other Learning Paradigms

- **Supervised learning**: Learns from labeled input-output pairs provided by an external supervisor
- **Unsupervised learning**: Discovers hidden patterns and structure in unlabeled data
- **Reinforcement learning**: Learns through interaction with an environment — the agent takes actions, observes outcomes, and improves its behavior over time

The key distinction: RL is about *interactive, goal-directed* learning. There is no labeled "correct action" — the agent must discover what works by trying things and observing consequences.

## Four Core Elements of RL

1. **Policy** — A mapping from states to actions. Defines the agent's behavior: "given this state, what action should I take?"
2. **Reward** — Immediate scalar feedback from the environment after taking an action. Tells the agent how good or bad a single step was.
3. **Value** — The long-term expected cumulative reward from a state (or state-action pair). Answers: "how good is it to *be* in this state, considering the future?"
4. **Model (of the environment)** — Predicts the next state and reward given the current state and action: (s, a) → (s', r)

## Model-Based vs Model-Free RL

- **Model-based RL**: The agent has (or learns) a model of the environment's dynamics, enabling it to *plan ahead* by simulating future trajectories
- **Model-free RL**: The agent learns directly from experience without building an internal model — relies on trial and error

## Key Insight

Value estimation is the critical challenge. Rewards are directly given by the environment, but values must be *estimated* from sequences of observations over the agent's lifetime. Most RL algorithms center on how to efficiently and accurately estimate value functions.
