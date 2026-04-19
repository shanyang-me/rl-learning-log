# Day 6: Value Functions and the Bellman Equation

## Policy

A policy π(a|s) is a probability distribution over actions given a state:

```
π(a|s) = Pr{A_t = a | S_t = s}
```

## Value Functions

**State-value function** — expected return starting from state s, following policy π:
```
v_π(s) = E_π[G_t | S_t = s]
```

**Action-value function** — expected return starting from state s, taking action a, then following π:
```
q_π(s, a) = E_π[G_t | S_t = s, A_t = a]
```

## Deriving the Bellman Equation

Start with q_π and expand G_t recursively:

```
q_π(s, a) = E_π[G_t | S_t = s, A_t = a]
           = E_π[R_{t+1} + γG_{t+1} | S_t = s, A_t = a]
```

By the Markov property, expand over all possible (s', r):

```
q_π(s, a) = Σ_{s',r} p(s', r | s, a) · [r + γ·v_π(s')]
```

This says: the value of taking action a in state s is the weighted sum over all possible next states s', of the immediate reward r plus the discounted future value v_π(s'), weighted by transition probability.

Now average over actions using the policy π(a|s):

```
v_π(s) = Σ_a π(a|s) · q_π(s, a)
       = Σ_a π(a|s) · Σ_{s',r} p(s', r | s, a) · [r + γ·v_π(s')]
```

This is the **Bellman equation for v_π** — it expresses the value of a state as a function of the values of its successor states. It's recursive: v_π(s) is defined in terms of v_π(s').

## The Backup Diagram

```
    (s)
   / | \        ← average over actions π(a|s)
  a  a  a
 /|\         ← for each action, sum over (s', r)
s' s' s'        weighted by p(s', r | s, a)
```

Each "backup" looks one step ahead: from s, consider all actions, then all possible next states, and combine.

## Key Insight

The Bellman equation turns the problem of computing value (an infinite sum of future rewards) into a system of simultaneous equations — one per state. If we can solve this system, we know the value of every state. That's exactly what dynamic programming does.

## Optimal Value Functions and Optimal Policy

**Optimal state-value function** — the best possible value achievable from any state:
```
v*(s) = max_π v_π(s)    for all s
```

**Optimal action-value function**:
```
q*(s, a) = max_π q_π(s, a)    for all s, a
```

**Bellman optimality equation** — replace the policy average (Σ_a π(a|s)) with a max:
```
v*(s) = max_a Σ_{s',r} p(s', r | s, a) · [r + γ·v*(s')]
```

Two paths to the optimal policy:
- **Value iteration**: Directly iterate on v using the Bellman optimality equation (apply max at each step) until v converges to v*, then derive π* by picking the greedy action at each state
- **Policy iteration**: Alternate between evaluating v_π (solve Bellman equation for current policy) and improving π (make it greedy w.r.t. current v_π) until the policy stops changing

Both are guaranteed to converge when γ < 1 (the Bellman optimality equation always has a unique solution).

## Why It's Intractable in Practice

The Bellman optimality equation is elegant but impractical for real problems:

1. **State/action space too large** — Can't enumerate all states in a table (e.g., images as states → billions of possible states)
2. **Memory impossible** — Tabular representation doesn't fit in memory
3. **Environment unknown** — We don't have p(s', r | s, a) to plug into the equation
4. **Computational cost** — Even if we had everything, iterating over all (s, a, s', r) combinations is exponential

## Optimal Control vs RL: Clarified

They solve the *same problem* (maximize cumulative reward in a sequential decision process) but under different assumptions:

| | Optimal Control (DP) | Reinforcement Learning |
|---|---|---|
| **Environment model** | Known — has full p(s', r \| s, a) | Unknown — must learn from experience |
| **Method** | Solve Bellman equation directly via computation | Approximate the solution through trial-and-error interaction |
| **State space** | Small enough for tabular enumeration | Can be massive — needs function approximation |
| **Key challenge** | Computational (solving the system) | Statistical (learning from limited samples) |

RL is not "model-less" exactly — it's that RL *doesn't require* a model. Model-free RL learns directly from experience. Model-based RL learns a model from experience and then uses it for planning (combining both threads). The rest of the book is about practical methods that approximate the optimal solution under these real-world constraints.
