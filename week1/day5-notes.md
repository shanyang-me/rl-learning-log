# Day 5: Finite Markov Decision Processes

## MDP Formulation

The agent-environment interaction is formalized as a probability distribution:

```
p(s', r | s, a) = Pr{S_{t+1} = s', R_{t+1} = r | S_t = s, A_t = a}
```

This four-argument dynamics function p(s', r | s, a) fully defines the environment. From it we can derive everything else by marginalization:

- **State-transition probabilities**: p(s' | s, a) = Σ_r p(s', r | s, a)
- **Expected reward**: r(s, a) = Σ_r r · Σ_{s'} p(s', r | s, a)
- **Expected reward given next state**: r(s, a, s') = Σ_r r · p(s', r | s, a) / p(s' | s, a)

The Markov property: everything you need to predict the future is in the current state — history doesn't matter.

## What Makes RL Different: Cumulative Reward

RL optimizes **cumulative** reward over time, not just immediate reward. This is the key distinction from bandits and supervised learning.

## Episodic vs Continuing Tasks

- **Episodic**: Interaction naturally breaks into episodes with terminal states (e.g., a game ends). Return is finite sum.
- **Continuing**: No natural end — the agent interacts forever (e.g., a thermostat). Need discounting to keep the return finite.

## Return and Discounting

The return G_t is the cumulative discounted reward from time t:

```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
    = Σ_{k=0}^{∞} γ^k R_{t+k+1}
```

**Discount factor γ ∈ [0, 1)**:
- γ → 0: **Myopic** agent — only cares about immediate reward
- γ → 1: **Farsighted** agent — future rewards matter almost as much as current ones
- γ = 1: Only valid for episodic tasks (sum must be finite)

The recursive form: G_t = R_{t+1} + γG_{t+1} — this is why the Bellman equation works.

## Key Takeaway

The MDP framework (s, a, s', r) + discounted cumulative return provides the mathematical foundation for all of RL. Every algorithm that follows is essentially a different strategy for maximizing G_t.
