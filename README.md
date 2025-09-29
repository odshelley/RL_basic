# RL Gridworld: Stochastic 5×5 Gridworld with Cliff

A comprehensive implementation of reinforcement learning algorithms on a stochastic gridworld environment, starting with Policy Iteration.

## Environment Description

This project implements the recommended toy problem: a **5×5 Stochastic Gridworld with a cliff**.

### Environment Specifications

- **States**: 25 cells (i,j) on a 5×5 grid
- **Start state**: Fixed at (4,0) (bottom-left)
- **Goal**: (0,4) (top-right), terminal reward +1.0
- **Pit (cliff)**: Entire top row except goal, i.e., (0,0)…(0,3), terminal reward −1.0
- **Actions**: up, down, left, right (4 actions)
- **Dynamics (slip)**: 
  - 80% probability: move as intended
  - 10% probability: veer 90° left
  - 10% probability: veer 90° right
  - Bumping a wall keeps you in place
- **Step reward**: −0.01 per move (encourages shorter paths)
- **Discount factor**: γ = 0.99
- **Episode termination**: Hitting Goal or Pit, or after T=200 steps (failsafe)

## Project Structure

```
rl_gridworld/
├── src/
│   ├── gridworld/
│   │   ├── __init__.py
│   │   └── environment.py      # Stochastic gridworld implementation
│   ├── algorithms/
│   │   ├── __init__.py
│   │   └── policy_iteration.py # Policy Iteration algorithm
│   └── utils/
│       ├── __init__.py
│       └── visualization.py    # Plotting and visualization utilities
├── examples/
│   └── policy_iteration_example.py  # Complete example with visualizations
├── tests/
│   └── test_basic.py           # Basic tests
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## Quick Start

1. **Set up the environment** (assuming you have uv installed):
   ```bash
   cd rl_gridworld
   uv sync
   ```

2. **Run the tests** (using pytest):
   ```bash
   uv run --with pytest pytest tests/ -v
   ```

3. **Run the Policy Iteration example**:
   ```bash
   uv run python examples/policy_iteration_example.py
   ```

## Policy Iteration Algorithm

The implemented Policy Iteration algorithm alternates between:

1. **Policy Evaluation**: Compute value function V^π for current policy π using iterative policy evaluation
2. **Policy Improvement**: Update policy π to be greedy with respect to V^π

The algorithm is guaranteed to converge to the optimal policy π* and optimal value function V*.

### Key Features

- **Convergence guarantee**: The algorithm will find the optimal policy
- **Exact solution**: Unlike approximate methods, this gives the exact optimal policy
- **Configurable precision**: Adjustable convergence threshold (θ)
- **Comprehensive tracking**: Records value function and policy evolution
- **Policy evaluation**: Includes methods to evaluate the learned policy

## Usage Example

```python
from gridworld import StochasticGridworld
from algorithms import PolicyIteration
from utils import plot_policy, plot_value_function

# Create environment
env = StochasticGridworld(size=5, gamma=0.99, max_steps=200)

# Run Policy Iteration
pi = PolicyIteration(env, theta=1e-6)
results = pi.run(max_iterations=100, verbose=True)

# Visualize results
plot_value_function(env, results['final_value_function'])
plot_policy(env, results['final_policy'])

# Evaluate policy
evaluation = pi.evaluate_policy(num_episodes=1000)
print(f"Success rate: {evaluation['success_rate']:.3f}")
```

## Planned Algorithms

This project will eventually include implementations of:

- [x] **Policy Iteration** - Exact dynamic programming solution
- [ ] **Value Iteration** - Alternative exact DP approach  
- [ ] **Q-learning (SARSA-max)** - Model-free temporal difference learning
- [ ] **TD(λ) (forward and backward view)** - Temporal difference with eligibility traces
- [ ] **REINFORCE** - Policy gradient method
- [ ] **PPO** - Proximal Policy Optimization

## Visualization Features

The project includes comprehensive visualization tools:

- **Environment layout**: Visual representation of the gridworld
- **Value function heatmaps**: Color-coded state values
- **Policy arrows**: Visual representation of the learned policy
- **Learning curves**: Convergence analysis over iterations
- **Algorithm comparisons**: Side-by-side performance analysis

## Dependencies

- **Python 3.8+**
- **NumPy**: Numerical computations
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Enhanced statistical visualizations

## Development

To contribute to this project:

1. **Install dependencies**: `uv sync`
2. **Run tests**: `uv run python tests/test_basic.py`
3. **Add new algorithms** in `src/algorithms/`
4. **Add examples** in `examples/`
5. **Update tests** in `tests/`

## Results and Analysis

The Policy Iteration algorithm typically converges in 5-15 iterations for this gridworld. Key insights:

- **Optimal policy**: Generally directs the agent towards the goal while avoiding the pit
- **Value function**: Shows clear gradient from low values near the pit to high values near the goal
- **Stochasticity impact**: The slip probability creates more cautious policies compared to deterministic environments
- **Convergence**: Fast and guaranteed convergence to the optimal solution

## License

This project is for educational and research purposes. Feel free to use and modify as needed.

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.
- Dynamic Programming and Policy Iteration algorithms
- Gridworld environments as standard RL testbeds
