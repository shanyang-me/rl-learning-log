# Day 10: SARSA and Q-Learning (Chapter 6 cont.)

## From V to Q: TD for Control

TD(0) estimates V(s), but for control (choosing actions) without a model, we need Q(s,a). SARSA and Q-learning are TD applied to action-value functions.

## SARSA (On-Policy TD Control)

Update uses the tuple (S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}) — hence the name:

```
Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γQ(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
```

- **On-policy**: The next action A_{t+1} is chosen by the *same* ε-greedy policy being improved
- Evaluates and improves the policy it's actually following
- Learns Q for the ε-greedy policy (not the optimal policy)
- Accounts for exploration costs — if the policy sometimes takes bad actions, Q reflects that

## Q-Learning (Off-Policy TD Control)

```
Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γ·max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
```

The key difference: instead of using Q(S_{t+1}, A_{t+1}) from the behavior policy, it uses **max_a Q(S_{t+1}, a)** — the greedy action.

- **Off-policy**: Explores with ε-greedy (behavior policy), but learns about the greedy policy (target policy)
- The max makes it learn Q* directly — the optimal action-value function
- No importance sampling needed (unlike off-policy MC) because it only bootstraps one step ahead

## SARSA vs Q-Learning Side by Side

```
SARSA:      Q(S,A) ← Q(S,A) + α[R + γ·Q(S', A')        - Q(S,A)]
Q-learning: Q(S,A) ← Q(S,A) + α[R + γ·max_a Q(S', a)   - Q(S,A)]
                                          ↑
                              A' from policy (on)  vs  max (off)
```

| | SARSA | Q-Learning |
|---|---|---|
| **Type** | On-policy | Off-policy |
| **Next action in update** | A' from current ε-greedy policy | max_a (greedy, regardless of what was actually taken) |
| **Learns** | Q for the ε-greedy policy it follows | Q* for the optimal greedy policy |
| **Behavior** | Safer — penalizes risky exploration | Bolder — ignores exploration costs |

## On-Policy vs Off-Policy Clarified

- **On-policy** (SARSA): The policy generating experience is the same policy being evaluated. Q reflects what the agent *actually does*, including exploratory actions.
- **Off-policy** (Q-learning): The policy generating experience (ε-greedy) differs from the policy being learned about (greedy). Q reflects what the agent *should* do optimally, ignoring the fact that it sometimes explores.
