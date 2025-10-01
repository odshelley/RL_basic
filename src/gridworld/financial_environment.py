"""
Financial Cliff Walking Environment

A financial adaptation of the cliff walking problem:
- Start: Bank account with zero balance
- Goal: End with zero balance at final time step
- Actions: Deposit (+1), Withdraw (-1), Do Nothing (0)
- Cliff: Going below 0 = bankruptcy (immediate termination with large penalty)
- Transaction costs: Each deposit/withdraw action has a small cost
- Surplus penalty: Ending with positive balance incurs penalty
- Stochastic dynamics: Actions may fail or have unintended effects

This environment is ideal for CPT analysis because:
- Clear reference point (zero balance)
- Asymmetric outcomes (bankruptcy vs surplus)
- Natural risk-return tradeoffs
- Loss aversion effects (bankruptcy much worse than surplus)
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from enum import Enum


class FinancialAction(Enum):
    DEPOSIT = 0    # +1 to balance
    WITHDRAW = 1   # -1 to balance  
    NOTHING = 2    # 0 to balance


class FinancialCliffWalking:
    def __init__(self, 
                 max_balance: int = 10,
                 max_steps: int = 20,
                 gamma: float = 0.99,
                 transaction_cost: float = 0.02,
                 bankruptcy_penalty: float = -10.0,
                 surplus_penalty_rate: float = 0.1,
                 success_reward: float = 1.0,
                 action_noise_prob: float = 0.1):
        """
        Initialize the Financial Cliff Walking environment.
        
        Args:
            max_balance: Maximum allowed balance (upper bound)
            max_steps: Number of time steps (episode length)
            gamma: Discount factor
            transaction_cost: Cost per deposit/withdraw action
            bankruptcy_penalty: Penalty for going below 0 (bankruptcy)
            surplus_penalty_rate: Penalty rate for ending with surplus (per unit)
            success_reward: Reward for ending with exactly 0 balance
            action_noise_prob: Probability that action fails or has noise
        """
        self.max_balance = max_balance
        self.max_steps = max_steps
        self.gamma = gamma
        self.transaction_cost = transaction_cost
        self.bankruptcy_penalty = bankruptcy_penalty
        self.surplus_penalty_rate = surplus_penalty_rate
        self.success_reward = success_reward
        self.action_noise_prob = action_noise_prob
        
        # State space: balance can range from -1 (bankruptcy) to max_balance
        # -1 is special bankruptcy state (terminal)
        self.min_balance = -1
        self.start_balance = 0
        self.goal_balance = 0
        
        # Initialize state
        self.current_balance = self.start_balance
        self.step_count = 0
        self.terminal = False
        self.bankrupt = False
        
    def get_all_states(self) -> List[int]:
        """Return all possible balance states."""
        return list(range(self.min_balance, self.max_balance + 1))
    
    def get_all_actions(self) -> List[FinancialAction]:
        """Return all possible actions."""
        return list(FinancialAction)
    
    def get_valid_actions(self, balance: int) -> List[FinancialAction]:
        """Get valid actions for a given balance."""
        if balance <= 0:  # Bankruptcy or zero balance
            if balance < 0:
                return []  # No actions allowed in bankruptcy
            else:  # balance == 0
                return [FinancialAction.DEPOSIT, FinancialAction.NOTHING]
        else:  # Positive balance
            return [FinancialAction.DEPOSIT, FinancialAction.WITHDRAW, FinancialAction.NOTHING]
    
    def is_terminal(self, balance: int) -> bool:
        """Check if a balance state is terminal."""
        return balance < 0  # Bankruptcy is terminal
    
    def is_valid_balance(self, balance: int) -> bool:
        """Check if balance is within valid range."""
        return self.min_balance <= balance <= self.max_balance
    
    def get_next_balance(self, balance: int, action: FinancialAction) -> int:
        """Get next balance given current balance and action (deterministic)."""
        if self.is_terminal(balance):
            return balance
        
        if action == FinancialAction.DEPOSIT:
            next_balance = balance + 1
        elif action == FinancialAction.WITHDRAW:
            next_balance = balance - 1
        else:  # NOTHING
            next_balance = balance
        
        # Clamp to valid range (except bankruptcy which is allowed)
        if next_balance > self.max_balance:
            next_balance = self.max_balance
        
        return next_balance
    
    def get_transition_probabilities(self, balance: int, action: FinancialAction) -> Dict[int, float]:
        """
        Get transition probabilities for all possible next balances.
        
        Stochastic effects:
        - Action may fail (stay same balance)
        - Deposit may overshoot (+2 instead of +1)
        - Withdraw may overshoot (-2 instead of -1)
        
        Returns:
            Dictionary mapping next_balance -> probability
        """
        if self.is_terminal(balance):
            return {balance: 1.0}
        
        if action not in self.get_valid_actions(balance):
            # Invalid action - stay in same state
            return {balance: 1.0}
        
        transitions = {}
        
        # Intended action (high probability)
        intended_next = self.get_next_balance(balance, action)
        transitions[intended_next] = 1.0 - self.action_noise_prob
        
        # Noise effects
        if action == FinancialAction.DEPOSIT:
            # May overshoot: +2 instead of +1
            overshoot_balance = min(balance + 2, self.max_balance)
            transitions[overshoot_balance] = transitions.get(overshoot_balance, 0) + self.action_noise_prob * 0.5
            # May fail: stay same
            transitions[balance] = transitions.get(balance, 0) + self.action_noise_prob * 0.5
            
        elif action == FinancialAction.WITHDRAW:
            # May overshoot: -2 instead of -1 (dangerous!)
            overshoot_balance = balance - 2
            transitions[overshoot_balance] = transitions.get(overshoot_balance, 0) + self.action_noise_prob * 0.5
            # May fail: stay same
            transitions[balance] = transitions.get(balance, 0) + self.action_noise_prob * 0.5
            
        else:  # NOTHING
            # Small chance of random fluctuation
            if balance > 0:
                # Might randomly lose 1 (market fluctuation)
                fluctuation_balance = balance - 1
                transitions[fluctuation_balance] = transitions.get(fluctuation_balance, 0) + self.action_noise_prob * 0.3
                transitions[balance] = transitions.get(balance, 0) + self.action_noise_prob * 0.7
            else:
                # At zero, doing nothing is safe
                transitions[balance] = 1.0
        
        return transitions
    
    def get_reward(self, balance: int, action: FinancialAction, next_balance: int, is_final_step: bool = False) -> float:
        """
        Get reward for transitioning from balance to next_balance via action.
        
        Reward structure:
        - Bankruptcy: Large negative penalty
        - Transaction costs: Small cost for deposit/withdraw
        - End-of-episode: Success reward for zero balance, surplus penalty otherwise
        """
        reward = 0.0
        
        # Bankruptcy penalty (terminal)
        if next_balance < 0:
            return self.bankruptcy_penalty
        
        # Transaction costs
        if action in [FinancialAction.DEPOSIT, FinancialAction.WITHDRAW]:
            reward -= self.transaction_cost
        
        # End-of-episode evaluation
        if is_final_step:
            if next_balance == self.goal_balance:
                reward += self.success_reward
            else:
                # Surplus penalty (linear in surplus amount)
                surplus_penalty = self.surplus_penalty_rate * next_balance
                reward -= surplus_penalty
        
        return reward
    
    def reset(self) -> int:
        """Reset the environment to initial state."""
        self.current_balance = self.start_balance
        self.step_count = 0
        self.terminal = False
        self.bankrupt = False
        return self.current_balance
    
    def step(self, action: FinancialAction) -> Tuple[int, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Financial action to take
            
        Returns:
            (next_balance, reward, done, info)
        """
        if self.terminal:
            return self.current_balance, 0, True, {"message": "Episode already terminated"}
        
        # Check if action is valid
        valid_actions = self.get_valid_actions(self.current_balance)
        if action not in valid_actions:
            # Invalid action - stay same, no reward, continue
            reward = -0.01  # Small penalty for invalid action
            info = {
                "step_count": self.step_count,
                "valid_actions": [a.name for a in valid_actions],
                "message": f"Invalid action {action.name} for balance {self.current_balance}"
            }
            return self.current_balance, reward, False, info
        
        # Sample next balance based on transition probabilities
        transitions = self.get_transition_probabilities(self.current_balance, action)
        balances = list(transitions.keys())
        probs = list(transitions.values())
        
        next_balance = balances[np.random.choice(len(balances), p=probs)]
        
        # Check if this is the final step
        is_final_step = (self.step_count + 1 >= self.max_steps)
        
        # Get reward
        reward = self.get_reward(self.current_balance, action, next_balance, is_final_step)
        
        # Update state
        self.current_balance = next_balance
        self.step_count += 1
        
        # Check terminal conditions
        done = (self.is_terminal(next_balance) or self.step_count >= self.max_steps)
        self.terminal = done
        
        if next_balance < 0:
            self.bankrupt = True
        
        info = {
            "step_count": self.step_count,
            "max_steps_reached": self.step_count >= self.max_steps,
            "bankrupt": self.bankrupt,
            "valid_actions": [a.name for a in self.get_valid_actions(next_balance)],
            "is_final_step": is_final_step
        }
        
        return next_balance, reward, done, info
    
    def render(self) -> str:
        """Render the current state of the financial environment."""
        status = []
        status.append(f"Step: {self.step_count}/{self.max_steps}")
        status.append(f"Balance: ${self.current_balance}")
        
        if self.current_balance < 0:
            status.append("STATUS: BANKRUPT 💀")
        elif self.current_balance == 0:
            status.append("STATUS: Zero Balance ⚖️")
        else:
            status.append(f"STATUS: Surplus +${self.current_balance} 💰")
        
        if self.terminal:
            if self.bankrupt:
                status.append("RESULT: Bankruptcy!")
            elif self.step_count >= self.max_steps:
                if self.current_balance == 0:
                    status.append("RESULT: Success! 🎉")
                else:
                    status.append(f"RESULT: Surplus penalty -${self.surplus_penalty_rate * self.current_balance:.2f}")
        
        # Show balance timeline
        balance_line = "Balance: ["
        for i in range(self.step_count + 1):
            if i == self.step_count:
                balance_line += f"${self.current_balance}]"
            else:
                balance_line += f"$?, "
        
        # Show valid actions
        if not self.terminal:
            valid_actions = self.get_valid_actions(self.current_balance)
            action_names = [a.name for a in valid_actions]
            status.append(f"Valid actions: {action_names}")
        
        return "\n".join(status)
    
    def get_balance_index(self, balance: int) -> int:
        """Convert balance to array index."""
        return balance - self.min_balance
    
    def get_balance_from_index(self, index: int) -> int:
        """Convert array index to balance."""
        return index + self.min_balance
    
    def evaluate_policy_performance(self, policy_results: List[Tuple[int, float, bool]]) -> Dict:
        """
        Evaluate policy performance from episode results.
        
        Args:
            policy_results: List of (final_balance, total_reward, bankrupt) tuples
            
        Returns:
            Performance statistics
        """
        if not policy_results:
            return {}
        
        total_episodes = len(policy_results)
        bankruptcy_count = sum(1 for _, _, bankrupt in policy_results if bankrupt)
        success_count = sum(1 for balance, _, bankrupt in policy_results 
                          if balance == 0 and not bankrupt)
        surplus_count = sum(1 for balance, _, bankrupt in policy_results 
                          if balance > 0 and not bankrupt)
        
        total_rewards = [reward for _, reward, _ in policy_results]
        
        return {
            "total_episodes": total_episodes,
            "bankruptcy_rate": bankruptcy_count / total_episodes,
            "success_rate": success_count / total_episodes,
            "surplus_rate": surplus_count / total_episodes,
            "mean_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "min_reward": np.min(total_rewards),
            "max_reward": np.max(total_rewards)
        }


# Utility functions for creating different financial scenarios
def create_conservative_environment() -> FinancialCliffWalking:
    """Create a conservative financial environment (low risk, low transaction costs)."""
    return FinancialCliffWalking(
        max_balance=5,
        max_steps=10,
        transaction_cost=0.01,
        bankruptcy_penalty=-5.0,
        surplus_penalty_rate=0.05,
        action_noise_prob=0.05
    )

def create_risky_environment() -> FinancialCliffWalking:
    """Create a risky financial environment (high volatility, high penalties)."""
    return FinancialCliffWalking(
        max_balance=15,
        max_steps=25,
        transaction_cost=0.05,
        bankruptcy_penalty=-20.0,
        surplus_penalty_rate=0.2,
        action_noise_prob=0.2
    )

def create_standard_environment() -> FinancialCliffWalking:
    """Create the standard financial cliff walking environment."""
    return FinancialCliffWalking()
