# Day 3: Multi-Armed Bandits

## The Problem

A simplified RL setting: a slot machine with k arms (actions). Each pull gives a stochastic reward. Goal: maximize total reward over time.

- Each arm = an action
- Pulling an arm = taking an action → receiving a reward
- **Action value** q(a) = expected (mean) reward of action a

This strips away states and transitions — pure action selection problem.

## Greedy Action Selection

Estimate action values from past rewards, then always pick the action with the highest estimated value:

```
A_t = argmax_a Q_t(a)
```

Problem: greedy gets stuck. If an early estimate is wrong, it never tries other actions to discover they might be better.

## ε-Greedy (Epsilon-Greedy)

With probability (1 - ε): exploit — pick the greedy action
With probability ε: explore — pick a random action

Even small ε (e.g., 0.1) significantly outperforms pure greedy, because exploration discovers better actions that greedy would never try.

## Value Estimation: Stationary vs Non-Stationary

**Sample average** — estimate Q(a) as the mean of all observed rewards for action a:
```
Q_t(a) = sum of rewards for a / number of times a was chosen
```
Works well when reward distributions are fixed (stationary).

**Exponential recency-weighted average** — for non-stationary environments where reward distributions change over time, give recent rewards more weight:
```
Q_{t+1} = Q_t + α(R_t - Q_t)
```
With constant step-size α, recent rewards get exponentially more influence — older rewards decay by (1-α)^n. This lets the agent adapt to changing conditions. (By contrast, the sample average with α = 1/n gives all past rewards equal weight.)

## Core Takeaway

The fundamental tradeoff in RL: **exploration vs exploitation**.
- Exploit: use what you know to maximize immediate reward
- Explore: try new things to improve future decisions

How you estimate values directly affects which actions get selected, which in turn determines the quality of future estimates — a feedback loop at the heart of all RL.
