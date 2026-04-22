# Day 8: Monte Carlo Methods (Chapter 5)

## From DP to Monte Carlo: No Model Required

| | DP | Monte Carlo |
|---|---|---|
| **Requires model** p(s',r\|s,a) | Yes | No |
| **Evaluation method** | Sweep all states, use Bellman equation | Run episodes, average actual returns |
| **Task type** | Episodic or continuing | Episodic only |
| **Updates** | Bootstraps from estimated values | Uses complete actual returns |

DP computes values by exhaustively iterating over all (s, a, s', r) combinations. MC doesn't know the environment dynamics — instead it *experiences* the environment by running episodes and observing what actually happens.

## MC Policy Evaluation

Generate episodes under policy π, then estimate values from observed returns:

```
Episode: S_0, A_0, R_1, S_1, A_1, R_2, ..., S_T
Return from state s: G_t = R_{t+1} + γR_{t+2} + ... + γ^{T-t-1}R_T
V(s) ≈ average of all G_t observed from state s
```

Must be **episodic** — need the episode to end so we can compute the full return G_t.

## First-Visit vs Every-Visit MC

- **First-visit MC**: Only use the return from the *first* time state s appears in an episode
- **Every-visit MC**: Use the return from *every* occurrence of state s in an episode

Both converge to v_π. First-visit has unbiased estimates; every-visit has lower variance with more samples per episode.

## Why MC Uses q(s,a) Instead of v(s)

In DP, we can do policy improvement with v(s) because we have the model — we can look ahead one step to evaluate each action:
```
π'(s) = argmax_a Σ_{s',r} p(s',r|s,a)[r + γV(s')]
```

Without a model, we can't do this lookahead. So MC estimates **q(s,a)** directly — the value of each state-action pair — and improves the policy by picking the action with highest q:
```
π'(s) = argmax_a Q(s,a)
```

No model needed for improvement.

## On-Policy vs Off-Policy

**On-policy** (e.g., ε-greedy MC):
- One policy does everything: it both explores and is the policy being evaluated/improved
- Uses ε-greedy: mostly exploit (1-ε), sometimes explore (ε)
- Simple but limited — can only learn about the policy you're currently following
- Converges to the best policy *among ε-soft policies* (not truly optimal since it always has ε exploration)

**Off-policy** (two policies):
- **Target policy** π: the policy we want to evaluate/improve (can be fully greedy = optimal)
- **Behavior policy** b: the policy that generates episodes (exploratory, must cover all actions)
- Uses **importance sampling** to correct for the mismatch: weight returns by π(a|s)/b(a|s)
- More complex but can learn the truly optimal policy while exploring freely

## Still GPI

MC methods are still generalized policy iteration:
- **Evaluation**: Estimate q_π from episode returns (instead of Bellman sweeps)
- **Improvement**: Make policy greedy w.r.t. Q (same idea as DP)

The only difference from DP is *how* the evaluation step works — experience replaces the model.
