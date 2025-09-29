"""
Basic tests for the Stochastic Gridworld and Policy Iteration implementation.
"""

import sys
import os
import pytest
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import modules directly
from gridworld.environment import StochasticGridworld, Action
from algorithms.policy_iteration import PolicyIteration
from algorithms.value_iteration import ValueIteration
from algorithms.sarsa import SARSA
from algorithms.qlearning import QLearning


@pytest.fixture
def small_env():
    """Fixture for a small 3x3 environment for faster testing."""
    return StochasticGridworld(size=3)


@pytest.fixture
def standard_env():
    """Fixture for the standard 5x5 environment."""
    return StochasticGridworld(size=5)


def test_environment_initialization(standard_env):
    """Test basic environment initialization."""
    env = standard_env
    
    # Test initialization
    assert env.size == 5
    assert env.start_state == (4, 0)
    assert env.goal_state == (0, 4)
    assert len(env.pit_states) == 4


def test_environment_terminal_states(standard_env):
    """Test terminal state detection."""
    env = standard_env
    
    # Test state properties
    assert env.is_terminal(env.goal_state)
    assert all(env.is_terminal(pit) for pit in env.pit_states)
    assert not env.is_terminal(env.start_state)
    assert not env.is_terminal((2, 2))


def test_environment_transitions(standard_env):
    """Test transition probability computation."""
    env = standard_env
    
    # Test transitions
    transitions = env.get_transition_probabilities((2, 2), Action.UP)
    assert len(transitions) <= 3  # Intended + 2 slips
    assert abs(sum(transitions.values()) - 1.0) < 1e-10


def test_policy_iteration_initialization(small_env):
    """Test Policy Iteration algorithm initialization."""
    env = small_env
    pi = PolicyIteration(env, theta=1e-4)
    
    # Test initialization
    assert len(pi.V) == 9  # 3x3 grid
    assert len(pi.policy) == 9


def test_policy_iteration_convergence(small_env):
    """Test that Policy Iteration converges."""
    env = small_env
    pi = PolicyIteration(env, theta=1e-4)
    
    # Test policy iteration
    results = pi.run(max_iterations=50, verbose=False)
    
    # Should converge
    assert results['converged']
    assert results['iterations'] > 0


def test_policy_iteration_value_function(small_env):
    """Test that the learned value function is reasonable."""
    env = small_env
    pi = PolicyIteration(env, theta=1e-4)
    pi.run(max_iterations=50, verbose=False)
    
    # Value function should be reasonable
    start_value = pi.get_state_value(env.start_state)
    goal_value = pi.get_state_value(env.goal_state)
    
    # Goal should have higher value than start (after discounting)
    assert goal_value >= start_value


def test_policy_iteration_policy_completeness(small_env):
    """Test that policy is defined for all non-terminal states."""
    env = small_env
    pi = PolicyIteration(env, theta=1e-4)
    pi.run(max_iterations=50, verbose=False)
    
    # Policy should exist for non-terminal states
    for state in env.get_all_states():
        if not env.is_terminal(state):
            action = pi.get_policy_action(state)
            assert action is not None
            assert action in env.get_all_actions()


def test_policy_evaluation(small_env):
    """Test policy evaluation functionality."""
    env = small_env
    pi = PolicyIteration(env, theta=1e-4)
    pi.run(max_iterations=50, verbose=False)
    
    # Test policy evaluation
    evaluation = pi.evaluate_policy(num_episodes=10)
    
    assert 'mean_return' in evaluation
    assert 'success_rate' in evaluation
    assert 0 <= evaluation['success_rate'] <= 1


def test_transition_probabilities_edge_cases(standard_env):
    """Test transition probabilities for edge cases like walls and corners."""
    env = standard_env
    
    # Test corner case - top-left corner trying to go up
    transitions = env.get_transition_probabilities((0, 0), Action.UP)
    # Should stay in place due to wall, but might be a pit state
    assert sum(transitions.values()) == pytest.approx(1.0, abs=1e-10)
    
    # Test wall case - middle of bottom row trying to go down
    transitions = env.get_transition_probabilities((4, 2), Action.DOWN)
    # Should stay in place due to wall
    assert sum(transitions.values()) == pytest.approx(1.0, abs=1e-10)


def test_terminal_state_transitions(standard_env):
    """Test that terminal states have proper transition probabilities."""
    env = standard_env
    
    # Goal state should stay in goal
    transitions = env.get_transition_probabilities(env.goal_state, Action.UP)
    assert len(transitions) == 1
    assert transitions[env.goal_state] == 1.0
    
    # Pit states should stay in pit
    for pit in env.pit_states:
        transitions = env.get_transition_probabilities(pit, Action.RIGHT)
        assert len(transitions) == 1
        assert transitions[pit] == 1.0


def test_value_iteration_initialization(small_env):
    """Test Value Iteration algorithm initialization."""
    env = small_env
    vi = ValueIteration(env, theta=1e-4)
    
    # Test initialization
    assert len(vi.V) == 9  # 3x3 grid
    assert all(v == 0.0 for v in vi.V.values())  # Should start with zeros


def test_value_iteration_convergence(small_env):
    """Test that Value Iteration converges."""
    env = small_env
    vi = ValueIteration(env, theta=1e-4)
    
    # Test value iteration
    results = vi.run(max_iterations=1000, verbose=False)
    
    # Should converge
    assert results['converged']
    assert results['iterations'] > 0
    assert results['final_delta'] < vi.theta


def test_value_iteration_policy_extraction(small_env):
    """Test that Value Iteration extracts a valid policy."""
    env = small_env
    vi = ValueIteration(env, theta=1e-4)
    vi.run(max_iterations=1000, verbose=False)
    
    # Policy should exist for all non-terminal states
    for state in env.get_all_states():
        if not env.is_terminal(state):
            action = vi.get_policy_action(state)
            assert action is not None
            assert action in env.get_all_actions()


def test_value_iteration_vs_policy_iteration(small_env):
    """Test that Value Iteration and Policy Iteration find similar solutions."""
    env = small_env
    
    # Run both algorithms
    vi = ValueIteration(env, theta=1e-6)
    vi.run(max_iterations=1000, verbose=False)
    
    pi = PolicyIteration(env, theta=1e-6)
    pi.run(max_iterations=50, verbose=False)
    
    # Value functions should be very similar
    max_diff = 0
    for state in env.get_all_states():
        diff = abs(vi.get_state_value(state) - pi.get_state_value(state))
        max_diff = max(max_diff, diff)
    
    assert max_diff < 1e-4  # Should be very close
    
    # Both should have similar performance
    vi_eval = vi.evaluate_policy(num_episodes=100)
    pi_eval = pi.evaluate_policy(num_episodes=100)
    
    # Success rates should be close (allowing some randomness in evaluation)
    assert abs(vi_eval['success_rate'] - pi_eval['success_rate']) < 0.2


def test_action_value_computation(small_env):
    """Test computation of action values (Q-values)."""
    env = small_env
    vi = ValueIteration(env, theta=1e-4)
    vi.run(max_iterations=1000, verbose=False)
    
    # Test action values for a non-terminal state
    state = (1, 1)  # Center of 3x3 grid
    if not env.is_terminal(state):
        action_values = vi.get_action_values(state)
        
        # Should have values for all actions
        assert len(action_values) == len(env.get_all_actions())
        
        # All values should be finite
        assert all(np.isfinite(v) for v in action_values.values())
        
        # The policy action should have the maximum value
        policy_action = vi.get_policy_action(state)
        if policy_action is not None:
            policy_value = action_values[policy_action]
            assert all(policy_value >= v - 1e-10 for v in action_values.values())


# ============================================================================
# SARSA Algorithm Tests
# ============================================================================

@pytest.fixture
def sarsa_agent(small_env):
    """Fixture for a SARSA agent on small environment."""
    return SARSA(small_env, alpha=0.1, epsilon=0.3)


def test_sarsa_initialization(sarsa_agent):
    """Test SARSA agent initialization."""
    sarsa = sarsa_agent
    
    # Test basic properties
    assert sarsa.alpha == 0.1
    assert sarsa.initial_epsilon == 0.3
    assert sarsa.epsilon == 0.3
    
    # Test Q-table initialization
    assert len(sarsa.Q) == len(sarsa.states)
    for state in sarsa.states:
        assert len(sarsa.Q[state]) == len(sarsa.actions)
        
        # Terminal states should have zero Q-values
        if sarsa.env.is_terminal(state):
            assert all(q_val == 0.0 for q_val in sarsa.Q[state].values())


def test_sarsa_epsilon_greedy_action(sarsa_agent):
    """Test epsilon-greedy action selection."""
    sarsa = sarsa_agent
    state = (1, 1)  # Non-terminal state
    
    if not sarsa.env.is_terminal(state):
        # Test that actions are valid
        for _ in range(10):
            action = sarsa.get_epsilon_greedy_action(state)
            assert action in sarsa.actions
        
        # Test terminal state
        if sarsa.env.is_terminal(sarsa.env.goal_state):
            assert sarsa.get_epsilon_greedy_action(sarsa.env.goal_state) is None


def test_sarsa_greedy_action(sarsa_agent):
    """Test greedy action selection."""
    sarsa = sarsa_agent
    state = (1, 1)  # Non-terminal state
    
    if not sarsa.env.is_terminal(state):
        # Initially, actions might be random due to tie-breaking
        greedy_action = sarsa.get_greedy_action(state)
        assert greedy_action in sarsa.actions
        
        # Modify Q-values to create a clear winner
        for action in sarsa.actions:
            sarsa.Q[state][action] = 0.0
        sarsa.Q[state][Action.UP] = 1.0
        
        # Now greedy action should be UP
        assert sarsa.get_greedy_action(state) == Action.UP


def test_sarsa_q_value_update():
    """Test Q-value updates using SARSA rule."""
    env = StochasticGridworld(size=3, gamma=0.9)
    sarsa = SARSA(env, alpha=0.5, epsilon=0.1)
    
    state = (2, 1)
    action = Action.UP
    reward = -0.01
    next_state = (1, 1)
    next_action = Action.RIGHT
    
    # Store initial Q-value
    initial_q = sarsa.Q[state][action]
    
    # Update Q-value
    sarsa.update_q_value(state, action, reward, next_state, next_action)
    
    # Q-value should have changed
    assert sarsa.Q[state][action] != initial_q
    
    # Verify SARSA update rule: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
    expected_target = reward + env.gamma * sarsa.Q[next_state][next_action]
    expected_q = initial_q + sarsa.alpha * (expected_target - initial_q)
    assert abs(sarsa.Q[state][action] - expected_q) < 1e-10


def test_sarsa_episode_run():
    """Test running a single SARSA episode."""
    env = StochasticGridworld(size=3, gamma=0.99, max_steps=50)
    sarsa = SARSA(env, alpha=0.1, epsilon=0.3)
    
    # Run an episode
    episode_return, episode_length = sarsa.run_episode(verbose=False)
    
    # Check basic properties
    assert isinstance(episode_return, (int, float))
    assert isinstance(episode_length, int)
    assert episode_length >= 0
    assert episode_length <= env.max_steps


def test_sarsa_learning():
    """Test SARSA learning over multiple episodes."""
    env = StochasticGridworld(size=3, gamma=0.99, max_steps=50)
    sarsa = SARSA(env, alpha=0.2, epsilon=0.5, epsilon_decay=0.99, min_epsilon=0.1)
    
    # Ensure start state Q-values are initialized
    start_state = env.start_state
    initial_start_q = dict(sarsa.Q[start_state]) if start_state in sarsa.Q else sarsa.get_action_values(start_state)
    
    # Run learning
    results = sarsa.run(num_episodes=500, verbose=False, save_frequency=100)
    
    # Check results structure
    assert 'episode_returns' in results
    assert 'episode_lengths' in results
    assert 'epsilon_history' in results
    assert 'final_evaluation' in results
    assert len(results['episode_returns']) == 500
    assert len(results['episode_lengths']) == 500
    
    # Check that Q-values for start state have changed (learning occurred)
    final_start_q = sarsa.Q[start_state]
    changed_count = 0
    for action in initial_start_q:
        if abs(final_start_q[action] - initial_start_q[action]) > 1e-3:
            changed_count += 1
    
    # At least some Q-values should have changed significantly
    assert changed_count > 0
    
    # Epsilon should have decayed
    assert sarsa.epsilon < sarsa.initial_epsilon
    assert sarsa.epsilon >= sarsa.min_epsilon
    
    # Final evaluation should show reasonable performance
    assert results['final_evaluation']['success_rate'] >= 0.0


def test_sarsa_policy_evaluation():
    """Test SARSA policy evaluation."""
    env = StochasticGridworld(size=3, gamma=0.99, max_steps=30)
    sarsa = SARSA(env, alpha=0.2, epsilon=0.3)
    
    # Run some learning first
    sarsa.run(num_episodes=200, verbose=False)
    
    # Evaluate policy
    evaluation = sarsa.evaluate_policy(num_episodes=100, use_greedy=True)
    
    # Check evaluation structure
    assert 'mean_return' in evaluation
    assert 'std_return' in evaluation
    assert 'mean_length' in evaluation
    assert 'std_length' in evaluation
    assert 'success_rate' in evaluation
    
    # Check that values are reasonable
    assert isinstance(evaluation['mean_return'], (int, float))
    assert isinstance(evaluation['success_rate'], (int, float))
    assert 0.0 <= evaluation['success_rate'] <= 1.0


def test_sarsa_state_value_extraction():
    """Test state value extraction from Q-values."""
    env = StochasticGridworld(size=3, gamma=0.99)
    sarsa = SARSA(env, alpha=0.1, epsilon=0.1)
    
    # Manually set some Q-values
    state = (1, 1)
    if not env.is_terminal(state):
        sarsa.Q[state][Action.UP] = 0.8
        sarsa.Q[state][Action.DOWN] = 0.6
        sarsa.Q[state][Action.LEFT] = 0.7
        sarsa.Q[state][Action.RIGHT] = 0.9
        
        # State value should be maximum Q-value
        assert sarsa.get_state_value(state) == 0.9
        
        # Policy action should be the one with max Q-value
        assert sarsa.get_policy_action(state) == Action.RIGHT
        
        # Action values should return all Q-values
        action_values = sarsa.get_action_values(state)
        assert len(action_values) == 4
        assert action_values[Action.RIGHT] == 0.9


# ============================================================================
# Q-Learning Algorithm Tests
# ============================================================================

@pytest.fixture
def qlearning_agent(small_env):
    """Fixture for a Q-learning agent on small environment."""
    return QLearning(small_env, alpha=0.1, epsilon=0.3)


def test_qlearning_initialization(qlearning_agent):
    """Test Q-learning agent initialization."""
    ql = qlearning_agent
    
    # Test basic properties
    assert ql.alpha == 0.1
    assert ql.initial_epsilon == 0.3
    assert ql.epsilon == 0.3
    
    # Test Q-table initialization
    assert len(ql.Q) == len(ql.states)
    for state in ql.states:
        assert len(ql.Q[state]) == len(ql.actions)
        
        # Terminal states should have zero Q-values
        if ql.env.is_terminal(state):
            assert all(q_val == 0.0 for q_val in ql.Q[state].values())


def test_qlearning_max_q_value():
    """Test max Q-value computation (key difference from SARSA)."""
    env = StochasticGridworld(size=3, gamma=0.9)
    ql = QLearning(env, alpha=0.1, epsilon=0.1)
    
    state = (1, 1)
    if not env.is_terminal(state):
        # Set different Q-values
        ql.Q[state][Action.UP] = 0.5
        ql.Q[state][Action.DOWN] = 0.8
        ql.Q[state][Action.LEFT] = 0.3
        ql.Q[state][Action.RIGHT] = 0.7
        
        # Max Q-value should be 0.8
        assert ql.get_max_q_value(state) == 0.8
        assert ql.get_state_value(state) == 0.8


def test_qlearning_vs_sarsa_update_rule():
    """Test the key difference: Q-learning vs SARSA update rules."""
    env = StochasticGridworld(size=3, gamma=0.9)
    ql = QLearning(env, alpha=0.5, epsilon=0.1)
    
    state = (2, 1)
    action = Action.UP
    reward = -0.01
    next_state = (1, 1)
    
    # Set Q-values in next state
    ql.Q[next_state][Action.UP] = 0.5
    ql.Q[next_state][Action.DOWN] = 0.8  # This is the max
    ql.Q[next_state][Action.LEFT] = 0.3
    ql.Q[next_state][Action.RIGHT] = 0.7
    
    # Store initial Q-value
    initial_q = ql.Q[state][action]
    
    # Q-learning update (uses max Q-value, doesn't need next_action)
    ql.update_q_value(state, action, reward, next_state)
    
    # Verify Q-learning update rule: uses MAX Q-value in next state
    max_next_q = 0.8  # Maximum Q-value in next state
    expected_target = reward + env.gamma * max_next_q
    expected_q = initial_q + ql.alpha * (expected_target - initial_q)
    assert abs(ql.Q[state][action] - expected_q) < 1e-10


def test_qlearning_episode_run():
    """Test running a single Q-learning episode."""
    env = StochasticGridworld(size=3, gamma=0.99, max_steps=50)
    ql = QLearning(env, alpha=0.1, epsilon=0.3)
    
    # Run an episode
    episode_return, episode_length = ql.run_episode(verbose=False)
    
    # Check basic properties
    assert isinstance(episode_return, (int, float))
    assert isinstance(episode_length, int)
    assert episode_length >= 0
    assert episode_length <= env.max_steps


def test_qlearning_learning():
    """Test Q-learning over multiple episodes."""
    env = StochasticGridworld(size=3, gamma=0.99, max_steps=50)
    ql = QLearning(env, alpha=0.2, epsilon=0.5, epsilon_decay=0.99, min_epsilon=0.1)
    
    # Ensure start state Q-values are initialized
    start_state = env.start_state
    initial_start_q = dict(ql.Q[start_state]) if start_state in ql.Q else ql.get_action_values(start_state)
    
    # Run learning
    results = ql.run(num_episodes=500, verbose=False, save_frequency=100)
    
    # Check results structure
    assert 'episode_returns' in results
    assert 'episode_lengths' in results
    assert 'epsilon_history' in results
    assert 'final_evaluation' in results
    assert len(results['episode_returns']) == 500
    assert len(results['episode_lengths']) == 500
    
    # Check that Q-values for start state have changed (learning occurred)
    final_start_q = ql.Q[start_state]
    changed_count = 0
    for action in initial_start_q:
        if abs(final_start_q[action] - initial_start_q[action]) > 1e-3:
            changed_count += 1
    
    # At least some Q-values should have changed significantly
    assert changed_count > 0
    
    # Epsilon should have decayed
    assert ql.epsilon < ql.initial_epsilon
    assert ql.epsilon >= ql.min_epsilon
    
    # Final evaluation should show reasonable performance
    assert results['final_evaluation']['success_rate'] >= 0.0


def test_qlearning_policy_evaluation():
    """Test Q-learning policy evaluation."""
    env = StochasticGridworld(size=3, gamma=0.99, max_steps=30)
    ql = QLearning(env, alpha=0.2, epsilon=0.3)
    
    # Run some learning first
    ql.run(num_episodes=200, verbose=False)
    
    # Evaluate policy
    evaluation = ql.evaluate_policy(num_episodes=100, use_greedy=True)
    
    # Check evaluation structure
    assert 'mean_return' in evaluation
    assert 'std_return' in evaluation
    assert 'mean_length' in evaluation
    assert 'std_length' in evaluation
    assert 'success_rate' in evaluation
    
    # Check that values are reasonable
    assert isinstance(evaluation['mean_return'], (int, float))
    assert isinstance(evaluation['success_rate'], (int, float))
    assert 0.0 <= evaluation['success_rate'] <= 1.0


def test_sarsa_vs_qlearning_interface():
    """Test that SARSA and Q-learning have the same interface."""
    env = StochasticGridworld(size=3, gamma=0.99)
    sarsa = SARSA(env, alpha=0.1, epsilon=0.1)
    ql = QLearning(env, alpha=0.1, epsilon=0.1)
    
    # Both should have the same methods
    assert hasattr(sarsa, 'get_policy_action')
    assert hasattr(ql, 'get_policy_action')
    
    assert hasattr(sarsa, 'get_state_value')
    assert hasattr(ql, 'get_state_value')
    
    assert hasattr(sarsa, 'get_action_values')
    assert hasattr(ql, 'get_action_values')
    
    assert hasattr(sarsa, 'evaluate_policy')
    assert hasattr(ql, 'evaluate_policy')
    
    # Both should return the same types
    state = (1, 1)
    if not env.is_terminal(state):
        sarsa_action = sarsa.get_policy_action(state)
        ql_action = ql.get_policy_action(state)
        
        assert type(sarsa_action) == type(ql_action)
        assert isinstance(sarsa.get_state_value(state), (int, float))
        assert isinstance(ql.get_state_value(state), (int, float))
