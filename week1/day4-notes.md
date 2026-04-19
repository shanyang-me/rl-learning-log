# Day 4: Gradient Bandits and Bayesian Approaches

## Gradient Bandit Method

Instead of estimating action values, estimate **action preferences** H(a) and derive selection probabilities via softmax:

```
π(a) = exp(H(a)) / Σ_b exp(H(b))
```

Update preferences using gradient ascent:
```
H_{t+1}(A_t) = H_t(A_t) + α(R_t - R̄)(1 - π_t(A_t))     # chosen action
H_{t+1}(a)   = H_t(a)   - α(R_t - R̄)π_t(a)               # all other actions
```

Where R̄ is the average reward baseline — if reward exceeds baseline, increase preference for the chosen action; otherwise decrease it.

Key insight: **optimize in logit space (preferences), not probability space.** This is a recurring ML pattern — logits → softmax → probabilities is more numerically stable and gives better gradients. The softmax saturates in probability space, but preferences can move freely in (-∞, +∞).

## Bayesian Approach to Exploration

The "correct" way to balance exploration vs exploitation: maintain a full posterior distribution over each action's value using Bayes' rule.

- Start with prior beliefs about each action's reward distribution
- After each observation, update the posterior
- Choose actions based on posterior (e.g., Thompson sampling: sample from each posterior, pick the highest)

**Problem**: Computing the optimal Bayesian solution is intractable — the state space of beliefs grows exponentially with the horizon.

## Gittins Index

A special case where the Bayesian approach is tractable:
- Assigns each action a single index number based on its posterior
- Always play the action with the highest index
- Provably optimal for stationary, discounted, independent bandit problems
- But doesn't generalize well to full RL (non-stationary, dependent states)

## Chapter 2 Summary

| Method | Learns | Mechanism |
|--------|--------|-----------|
| ε-greedy | Action values | Random exploration with fixed probability |
| Gradient bandit | Preferences | Gradient ascent on softmax policy |
| UCB | Action values | Optimistic bonus for under-explored actions |
| Bayesian/Gittins | Posterior distributions | Optimal but intractable in general |

On the 10-armed testbed, most methods perform comparably — no single method dominates universally. Simple methods like ε-greedy and UCB are competitive with more sophisticated approaches. The theoretical optimum (Bayesian) is computationally intractable — the open question throughout RL is how to approximate it efficiently.
