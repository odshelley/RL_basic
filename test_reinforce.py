"""
Test script for REINFORCE implementation.

This script tests the basic functionality of our REINFORCE agent
with a quick training run to verify everything works.
"""

import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.append(src_path)

from algorithms.reinforce import REINFORCEAgent, REINFORCEConfig


def test_reinforce_basic():
    """Test basic REINFORCE functionality."""
    print("Testing REINFORCE agent...")
    
    # Simple configuration for quick test
    config = REINFORCEConfig(
        hidden_dims=(32, 16),  # Smaller network
        learning_rate=0.01,
        gamma=0.99,
        grid_size=5,
        max_steps_per_episode=50,  # Shorter episodes
        observation_type="coordinates",
        num_episodes=50,  # Few episodes for quick test
        log_interval=10,
        save_interval=25,
        log_dir="logs/test_reinforce",
        save_model=False  # Don't save for test
    )
    
    try:
        # Create agent
        agent = REINFORCEAgent(config)
        print("✅ Agent created successfully")
        
        # Test episode collection
        states, actions, rewards, log_probs = agent.collect_episode()
        print(f"✅ Episode collection works: {len(states)} steps, total reward: {sum(rewards):.2f}")
        
        # Test return computation
        returns = agent.compute_returns(rewards)
        print(f"✅ Return computation works: returns range [{min(returns):.2f}, {max(returns):.2f}]")
        
        # Test policy update
        policy_loss = agent.update_policy(states, actions, returns, log_probs)
        print(f"✅ Policy update works: loss = {policy_loss:.3f}")
        
        # Quick training
        print("\nStarting quick training...")
        training_stats = agent.train()
        print(f"✅ Training completed: {len(training_stats['episode_rewards'])} episodes")
        
        # Quick evaluation
        eval_stats = agent.evaluate(num_episodes=10, render=False)
        print(f"✅ Evaluation works: success rate = {eval_stats['success_rate']:.1%}")
        
        print("\n🎉 All tests passed! REINFORCE implementation is working.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_observation_types():
    """Test different observation types."""
    print("\nTesting different observation types...")
    
    observation_types = ["coordinates", "one_hot", "grid"]
    
    for obs_type in observation_types:
        print(f"\nTesting {obs_type}...")
        
        config = REINFORCEConfig(
            hidden_dims=(16,),  # Very small network
            learning_rate=0.01,
            gamma=0.99,
            grid_size=3,  # Smaller grid
            max_steps_per_episode=20,
            observation_type=obs_type,
            num_episodes=5,  # Just a few episodes
            log_interval=2,
            log_dir=None,  # No logging
            save_model=False
        )
        
        try:
            agent = REINFORCEAgent(config)
            
            # Test one episode
            states, actions, rewards, log_probs = agent.collect_episode()
            print(f"  ✅ {obs_type}: episode with {len(states)} steps, reward {sum(rewards):.2f}")
            
            # Check observation shape
            obs_shape = states[0].shape
            print(f"  ✅ {obs_type}: observation shape {obs_shape}")
            
        except Exception as e:
            print(f"  ❌ {obs_type} failed: {str(e)}")
            return False
    
    print("✅ All observation types work!")
    return True


if __name__ == "__main__":
    print("REINFORCE Implementation Test")
    print("=" * 40)
    
    # Basic functionality test
    success = test_reinforce_basic()
    
    if success:
        # Test observation types
        test_observation_types()
        
        print("\n" + "=" * 40)
        print("🚀 Ready to run full comparison!")
        print("Run: python src/algorithms/comparison.py")
    else:
        print("\n❌ Fix the issues above before proceeding.")
