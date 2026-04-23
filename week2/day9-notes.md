# Day 9: Temporal Difference Learning (Chapter 6)

## TD vs MC vs DP

| | DP | MC | TD |
|---|---|---|---|
| **Model required** | Yes | No | No |
| **Waits for episode end** | No | Yes | No |
| **Update target** | r + γV(s') from model | G_t (actual full return) | R_{t+1} + γV(S_{t+1}) (bootstrapped) |

TD combines the best of both:
- **Model-free** like MC — learns from experience
- **Bootstraps** like DP — updates from estimated values, no need to wait for episode end

## TD(0) Update Rule

After each step (not episode), update immediately:

```
V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
```

- **TD target**: R_{t+1} + γV(S_{t+1}) — one actual reward + estimated future
- **TD error**: δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t) — how far off was our prediction?

The key insight: we only need to observe (S_t, R_{t+1}, S_{t+1}) — one single transition — to make an update. No need to wait for G_t.

## MC Error as a Sum of TD Errors

The MC update uses the full return:
```
V(S_t) ← V(S_t) + α[G_t - V(S_t)]
```

The MC error (G_t - V(S_t)) can be decomposed into a sum of TD errors along the trajectory:

```
G_t - V(S_t) = δ_t + γδ_{t+1} + γ²δ_{t+2} + ... + γ^{T-t-1}δ_{T-1}
```

This shows that MC and TD are deeply connected — MC's single big correction at episode end is equivalent to the discounted sum of all the small TD corrections made step by step. TD just spreads the learning across every timestep rather than waiting until the end.

## Why TD Matters

- **Online learning**: Update after every step, don't wait for episode end
- **Works for continuing tasks**: No episode boundary needed (unlike MC)
- **Lower variance** than MC (uses one random reward + estimate, not full noisy trajectory)
- **Biased** (bootstraps from estimated V, which may be wrong) — but bias shrinks as V improves
- **In practice**: TD often converges faster than MC despite the bias
