"""
Day 1: GridWorld Environment
============================
5x5 grid. Agent starts at (0,0), goal at (4,4).
Actions: 0=up, 1=right, 2=down, 3=left
Rewards: -1 per step, +10 at goal, -10 if hitting wall (agent stays put)
Episode ends at goal or after 100 steps.

TODO: Fill in the methods marked with `# YOUR CODE HERE`
"""

import numpy as np


class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.goal = (size - 1, size - 1)
        self.max_steps = 100

        # action mapping: 0=up, 1=right, 2=down, 3=left
        # each action is a (row_delta, col_delta)
        self.action_deltas = {
            0: (-1, 0),   # up
            1: (0, 1),    # right
            2: (1, 0),    # down
            3: (0, -1),   # left
        }

        self.reset()

    def reset(self):
        """Reset agent to start position. Return initial state."""
        self.agent_pos = (0, 0)
        self.steps = 0
        return self.agent_pos

    def step(self, action: int):
        """
        Take an action in the environment.

        Returns: (next_state, reward, done, info)

        Logic:
        1. Compute the new position from current position + action delta
        2. If new position is out of bounds → agent stays put, reward = -10
        3. If new position is the goal → move there, reward = +10, done = True
        4. Otherwise → move there, reward = -1
        5. Also done = True if steps >= max_steps
        """
        self.steps += 1

        # YOUR CODE HERE
        # Hint: use self.action_deltas[action] to get (dr, dc)
        # Hint: use self._is_valid(new_row, new_col) to check bounds
        (dr, dc) = self.action_deltas[action]
        new_row = self.agent_pos[0] + dr
        new_col = self.agent_pos[1] + dc
        reward = -1
        done = False
        if self._is_valid(new_row, new_col):
            self.agent_pos = (new_row, new_col)
            if self.agent_pos == self.goal:
                reward = 10
                done = True
        else:
            reward = -10
        if self.steps >= self.max_steps:
            done = True
        return self.agent_pos, reward, done, {}

    def _is_valid(self, row, col):
        """Check if (row, col) is within the grid."""
        # YOUR CODE HERE (one line)
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        return True 

    def render(self):
        """Print the grid. A = agent, G = goal, . = empty."""
        for r in range(self.size):
            row_str = ""
            for c in range(self.size):
                if (r, c) == self.agent_pos:
                    row_str += " A"
                elif (r, c) == self.goal:
                    row_str += " G"
                else:
                    row_str += " ."
            print(row_str)
        print()


# ── Acceptance test ──────────────────────────────────────────
if __name__ == "__main__":
    env = GridWorld()
    state = env.reset()
    total_reward = 0

    print("=== Random agent on 5x5 GridWorld ===\n")
    env.render()

    for t in range(20):
        action = np.random.randint(4)
        state, reward, done, info = env.step(action)
        action_name = {0: "up", 1: "right", 2: "down", 3: "left"}[action]
        print(f"Step {t+1}: {action_name} → {state}, reward={reward}")
        env.render()
        total_reward += reward
        if done:
            print(f"Episode finished at step {t+1}")
            break

    print(f"\nTotal reward: {total_reward}")