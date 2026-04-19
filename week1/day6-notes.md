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
