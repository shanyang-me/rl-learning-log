# RL Learning Log

My personal log of learning Reinforcement Learning — notes, experiments, and insights.

## Structure

```
rl-learning-log/
├── week1/          ← notes + code per week
│   ├── day1-notes.md
│   ├── day2-notes.md
│   ├── gridworld.py
│   ├── policy_eval.py
│   └── value_iteration.py
├── week2/
├── README.md
└── pyproject.toml
```

## Topics

- [x] Introduction to RL, RL vs supervised/unsupervised
- [x] MDPs and Bellman equations
- [ ] Dynamic programming (policy/value iteration)
- [ ] Monte Carlo methods
- [ ] Temporal difference learning (TD, SARSA, Q-learning)
- [ ] Policy gradient methods (REINFORCE, A2C, PPO)
- [ ] Actor-critic architectures
- [ ] Model-based RL and world models
- [ ] Multi-agent RL
- [ ] Offline RL
- [ ] RL for LLMs (RLHF, GRPO)

## Setup

```bash
uv sync   # install deps
uv run ipython   # interactive Python
uv run python week1/gridworld.py   # run scripts
```
