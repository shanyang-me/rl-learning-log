# Day 7: Dynamic Programming (Chapter 4)

## The Core Elements

- **π(a|s)** — Policy: which action to take in each state
- **p(s',r|s,a)** — Model: environment dynamics (required for DP)
- **v_π(s)** — State-value function: expected return from state s under policy π
- **q_π(s,a)** — Action-value function: expected return from state s taking action a then following π

## Policy Evaluation (Prediction)

Compute v_π for a given policy by iterative sweeps:

```
Sweep 1: V_1(s) for all s    (one pass through all states)
Sweep 2: V_2(s) for all s    (using V_1 values)
...
Sweep k: V_k → v_π           (converges to true value of the policy)
```

Each sweep applies the Bellman expectation equation to every state. Values stabilize when the update changes are below threshold.

## Policy Iteration (Control)

Alternates two steps to find the optimal policy:

1. **Policy Evaluation**: Given current π, compute v_π (full convergence)
2. **Policy Improvement**: For each state, find the best action given v_π → new π'

Repeat until the policy stops changing. Guaranteed to converge to π*.

## Value Iteration

Combines evaluation and improvement into a single sweep:

```
V(s) = max_a [r + γ·V(s')]    for all s
```

Faster than policy iteration because it doesn't wait for full policy evaluation to converge — it improves the policy at every single sweep. Equivalent to policy iteration with evaluation truncated to one sweep.

## Asynchronous DP

Standard DP sweeps update all states in a fixed order. Asynchronous DP relaxes this:
- States can be updated in any order
- Some states can be updated more often than others
- Evaluation and improvement can happen in parallel

As long as all states continue to be updated, convergence is still guaranteed.

## Generalized Policy Iteration (GPI)

The unifying idea behind all DP (and most RL) methods:

```
    evaluation
π ──────────────→ v_π
↑                   │
│   improvement     │
└───────────────────┘
```

Two interacting processes:
- **Policy evaluation**: Makes the value function consistent with the current policy
- **Policy improvement**: Makes the policy greedy with respect to the current value function

They push in complementary directions — evaluation asks "how good is this policy?" while improvement asks "can I do better?" GPI converges when both are stable, which happens only at the optimal policy.

Almost every RL method in the rest of the book is a form of GPI — they differ only in how they approximate the evaluation and improvement steps.
