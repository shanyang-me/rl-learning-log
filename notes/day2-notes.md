# Day 2: MDPs and History of RL

## Markov Decision Process (MDP)

RL formalizes the agent-environment interaction as a Markov Decision Process. An MDP captures:
- **States (S)** — possible situations the agent can be in
- **Actions (A)** — choices available to the agent
- **Rewards (R)** — feedback from the environment after each action
- **Transitions** — how the environment evolves: P(s', r | s, a)

The Markov property means the future depends only on the current state, not the full history. This makes the problem tractable.

Value computation is a central topic — how to estimate the long-term worth of states and actions under a given policy.

## Historical Threads of RL

RL emerged from two largely separate threads:

### Thread 1: Trial-and-Error Learning
- Rooted in animal psychology and learning theory
- The agent learns by trying actions and observing outcomes
- Emphasis on learning from *experience* without a model

### Thread 2: Optimal Control
- Rooted in control theory and operations research
- Formalized through the **Bellman equation**, which expresses the value of a state as the immediate reward plus the discounted value of the next state
- **Dynamic programming** — methods for solving the Bellman equation (policy iteration, value iteration)
- More focused on *analytical/computational* solutions given a known model

### Why They Were Separate
The optimal control thread assumed a known model and focused on solving the Bellman equation analytically or computationally. The learning thread focused on acquiring knowledge from experience without a model. Modern RL unifies both — using the Bellman equation as the foundation, but learning to solve it from interaction rather than from a complete model.
