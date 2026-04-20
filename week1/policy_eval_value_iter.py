"""
Day 3-4: Policy Evaluation + Value Iteration
=============================================
Implement on your 5x5 GridWorld (no walls, goal at (4,4)).

Rewards: -1 per step, +10 at goal, -10 hitting boundary (stay put)
γ = 0.9

TODO: Fill in the methods marked with `# YOUR CODE HERE`
"""

import numpy as np
import matplotlib.pyplot as plt


# ── Environment dynamics ─────────────────────────────────────
# You need p(s', r | s, a) to do DP.
# For your deterministic GridWorld, this is simple:
# given (s, a), there's exactly one (s', r) with probability 1.

SIZE = 5
GAMMA = 0.9
ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # up, right, down, left
GOAL = (4, 4)


def get_next_state_and_reward(s, a):
    """
    Given state s=(row,col) and action a, return (s', reward).
    This IS your environment model — DP requires it.
    """
    # YOUR CODE HERE
    # Reuse your GridWorld logic: boundary check, goal check, etc.
    # Return (next_state, reward)
    (dr, dc) = ACTIONS[a]
    new_row = s[0] + dr
    new_col = s[1] + dc
    reward = -1
    if new_row < 0 or new_row >= SIZE or new_col < 0 or new_col >= SIZE:
        # Hit boundary
        reward = -10
        new_row, new_col = s  # stay put
    elif (new_row, new_col) == GOAL:
        reward = 10
    return (new_row, new_col), reward

# ── Task 1: Policy Evaluation ────────────────────────────────
def policy_evaluation(policy, theta=1e-6):
    """
    Compute v_π for a given policy.

    Args:
        policy: dict mapping (row,col) -> action (int)
        theta: convergence threshold

    Returns:
        V: 2D numpy array of state values

    Algorithm (Sutton Ch 4.1):
        Initialize V(s) = 0 for all s
        Repeat:
            delta = 0
            For each s:
                v = V(s)
                V(s) = Σ_{s',r} p(s',r|s,π(s)) · [r + γ·V(s')]
                       (deterministic, so just one (s',r))
                delta = max(delta, |v - V(s)|)
        Until delta < theta
    """
    V = np.zeros((SIZE, SIZE))
    while True:
        delta = 0
        for r in range(SIZE):
            for c in range(SIZE):
                s = (r, c)
                if s == GOAL:
                    continue  # V(GOAL) = 0 by definition
                a_next = policy[s]
                (s_next, reward) = get_next_state_and_reward(s, a_next)
                v_old = V[s]
                V[s] = reward + GAMMA * V[s_next]  # one-step lookahead
                delta = max(delta, abs(V[s] - v_old))
        if delta < theta:
            break
    return V


# ── Task 2: Value Iteration ──────────────────────────────────
def value_iteration(theta=1e-6):
    """
    Compute v* directly.

    Returns:
        V: optimal value function
        policy: optimal policy (dict mapping state -> action)

    Algorithm (Sutton Ch 4.4):
        Initialize V(s) = 0 for all s
        Repeat:
            delta = 0
            For each s:
                v = V(s)
                V(s) = max_a Σ_{s',r} p(s',r|s,a) · [r + γ·V(s')]
                delta = max(delta, |v - V(s)|)
        Until delta < theta

        Extract policy: π(s) = argmax_a [r + γ·V(s')]
    """
    V = np.zeros((SIZE, SIZE))
    policy = {}
    while True:
        delta = 0
        for r in range(SIZE):
            for c in range(SIZE):
                s = (r, c)
                if s == GOAL:
                    continue  # V(GOAL) = 0 by definition
                max_value = float('-inf')
                best_action = None
                for a in ACTIONS.keys():
                    (s_next, reward) = get_next_state_and_reward(s, a)
                    value = reward + GAMMA * V[s_next]
                    if value > max_value:
                        max_value = value
                        best_action = a
                v_old = V[s]
                V[s] = max_value
                delta = max(delta, abs(V[s] - v_old))
        if delta < theta:
            break
    
    for r in range(SIZE):
        for c in range(SIZE):
            s = (r, c)
            if s == GOAL:
                policy[s] = 0  # no action needed at goal
                continue
            best_action = None
            max_value = float('-inf')
            for a in ACTIONS.keys():
                (s_next, reward) = get_next_state_and_reward(s, a)
                value = reward + GAMMA * V[s_next]
                if value > max_value:
                    max_value = value
                    best_action = a
            policy[s] = best_action

    return V, policy


# ── Visualization ─────────────────────────────────────────────
def plot_values(V, title="Value Function"):
    plt.figure(figsize=(6, 5))
    plt.imshow(V, cmap='viridis', origin='upper')
    for r in range(SIZE):
        for c in range(SIZE):
            plt.text(c, r, f"{V[r,c]:.1f}", ha='center', va='center',
                     color='white' if V[r,c] < V.mean() else 'black')
    plt.colorbar(label='Value')
    plt.title(title)
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=100)
    plt.show()


def plot_policy(policy, title="Policy"):
    arrows = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    plt.figure(figsize=(6, 5))
    plt.imshow(np.zeros((SIZE, SIZE)), cmap='Greys', vmin=0, vmax=1)
    for r in range(SIZE):
        for c in range(SIZE):
            if (r, c) == GOAL:
                plt.text(c, r, 'G', ha='center', va='center', fontsize=20)
            else:
                plt.text(c, r, arrows[policy[(r,c)]], ha='center', va='center', fontsize=20)
    plt.title(title)
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=100)
    plt.show()


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Evaluate random policy
    random_policy = {(r, c): np.random.randint(4) for r in range(SIZE) for c in range(SIZE)}
    V_random = policy_evaluation(random_policy)
    plot_values(V_random, "Random Policy Value")
    print(f"V(0,0) under random policy: {V_random[0,0]:.2f}")
    print(f"V(3,4) under random policy: {V_random[3,4]:.2f}")

    # 2. Value Iteration
    V_star, pi_star = value_iteration()
    plot_values(V_star, "Optimal Value Function")
    plot_policy(pi_star, "Optimal Policy")
    print(f"\nV*(0,0): {V_star[0,0]:.2f}")
    print(f"V*(3,4): {V_star[3,4]:.2f}")  # should be 10.0
    print(f"π*(3,3): {pi_star[(3,3)]}")    # should be 1 (right)
    print(f"π*(3,4): {pi_star[(3,4)]}")    # should be 2 (down)
