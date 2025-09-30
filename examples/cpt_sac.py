# CPT-SAC: Cumulative Prospect Theory Soft Actor-Critic
# Based on the working behavioral_sac.py with CPT evaluation for critic targets

import os
import random
import time
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from cleanrl_utils.buffers import ReplayBuffer

# Import our behavioral utilities
from behavioral import (
    create_distortion, create_utility, create_reference_provider, create_discounting_mode,
    cpt_functional, DistortionFunction, PowerUtility, ReferenceProvider, DiscountingMode
)


@dataclass
class BehavioralConfig:
    """Configuration for behavioral aspects"""
    mode: str = "cpt"  # "none", "choquet", "cpt"
    
    # Probability weighting - very conservative
    g_type: str = "prelec"  # "identity", "prelec", "wang" 
    g_params: Dict[str, Any] = field(default_factory=lambda: {"alpha": 0.95, "eta": 1.0})  # Nearly identity    # CPT parameters - very conservative for stability  
    lambda_loss_aversion: float = 1.05  # Minimal loss aversion
    u_plus: Dict[str, Any] = field(default_factory=lambda: {"type": "power", "alpha": 0.99, "eps": 1e-6})  # Nearly linear
    u_minus: Dict[str, Any] = field(default_factory=lambda: {"type": "power", "alpha": 0.99, "eps": 1e-6})  # Nearly linear
    
    # Reference point - conservative for Pendulum  
    reference: Dict[str, Any] = field(default_factory=lambda: {
        "type": "constant", 
        "constant": -1000.0,  # Conservative reference
        "ema_tau": 0.01
    })
    
    # Discounting
    discounting: Dict[str, Any] = field(default_factory=lambda: {
        "type": "standard",
        "gamma": 0.99,
        "beta": 1.0,
        "delta": 0.99,
        "mixture": {"gammas": [0.99, 0.95, 0.90], "probs": [0.6, 0.3, 0.1]}
    })
    
    # Minimal action sampling to reduce noise
    n_action_samples: int = 2  # Minimal sampling for less noise


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cpt-sac"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Pendulum-v1"
    """the environment id of the task"""
    total_timesteps: int = 200000
    """total timesteps of the experiments"""
    num_envs: int = 4
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Pendulum-v1"
    """the environment id of the task"""
    total_timesteps: int = 50000  # Reduced for quicker testing
    """total timesteps of the experiments"""
    num_envs: int = 8  # Increased for better parallelization
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    
    # Behavioral configuration
    behavioral: BehavioralConfig = field(default_factory=BehavioralConfig)


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class CPTSACAgent:
    """CPT-SAC Agent with behavioral evaluation"""
    
    def __init__(self, envs, device, behavioral_config: BehavioralConfig):
        self.device = device
        self.behavioral_config = behavioral_config
        
        # Initialize behavioral components
        self.setup_behavioral_components()
        
    def setup_behavioral_components(self):
        """Initialize behavioral components"""
        cfg = self.behavioral_config
        
        # Create distortion functions
        if cfg.mode == "cpt":
            # CPT uses separate distortions for gains and losses
            self.g_plus = create_distortion(cfg.g_type, cfg.g_params)
            self.g_minus = create_distortion(cfg.g_type, cfg.g_params)
        else:
            # Choquet uses single distortion
            self.g_plus = create_distortion(cfg.g_type, cfg.g_params)
            self.g_minus = None
        
        # Create utility functions for CPT
        if cfg.mode == "cpt":
            self.u_plus = create_utility(cfg.u_plus["type"], cfg.u_plus).to(self.device)
            self.u_minus = create_utility(cfg.u_minus["type"], cfg.u_minus).to(self.device)
        else:
            self.u_plus = None
            self.u_minus = None
        
        # Create reference provider
        self.reference_provider = create_reference_provider(
            cfg.reference["type"], cfg.reference
        )
        
        # Create discounting mode
        self.discounting_mode = create_discounting_mode(
            cfg.discounting["type"], cfg.discounting
        )
        
    def compute_behavioral_target(self, next_obs, rewards, dones, actor, qf1_target, qf2_target, alpha):
        """Compute behavioral critic targets - EXACTLY like standard SAC but with behavioral expectation"""
        cfg = self.behavioral_config
        
        if cfg.mode == "none":
            # Pure standard SAC - exactly like behavioral_sac.py
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(next_obs)
                qf1_next_target = qf1_target(next_obs, next_state_actions)
                qf2_next_target = qf2_target(next_obs, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = rewards.flatten() + (1 - dones.flatten()) * 0.99 * min_qf_next_target.view(-1)
            return next_q_value
        
        # For behavioral modes: use proper Choquet/CPT expectation
        # but keep the same target structure as standard SAC
        batch_size = next_obs.shape[0]
        
        with torch.no_grad():
            # Sample multiple next-state actions for expectation
            next_obs_expanded = next_obs.unsqueeze(1).expand(-1, cfg.n_action_samples, -1)
            next_obs_flat = next_obs_expanded.reshape(-1, next_obs.shape[-1])
            
            next_actions_flat, next_log_probs_flat, _ = actor.get_action(next_obs_flat)
            qf1_next_flat = qf1_target(next_obs_flat, next_actions_flat)
            qf2_next_flat = qf2_target(next_obs_flat, next_actions_flat)
            
            # Reshape to [batch, n_samples]
            qf1_next = qf1_next_flat.reshape(batch_size, cfg.n_action_samples)
            qf2_next = qf2_next_flat.reshape(batch_size, cfg.n_action_samples)
            next_log_probs = next_log_probs_flat.reshape(batch_size, cfg.n_action_samples)
            
            # Compute soft Q-values: min(Q1, Q2) - α * log_prob (standard SAC formula)
            min_qf_next = torch.min(qf1_next, qf2_next)
            soft_q_values = min_qf_next - alpha * next_log_probs  # [batch, n_samples]
            
            # Apply behavioral expectation instead of mean
            if cfg.mode == "choquet":
                behavioral_expectation = self._compute_choquet_batch(soft_q_values)
            elif cfg.mode == "cpt":
                references = self.reference_provider.get_reference(next_obs)
                behavioral_expectation = self._compute_cpt_batch(soft_q_values, references)
            else:
                raise ValueError(f"Unknown behavioral mode: {cfg.mode}")
            
            # Standard SAC target structure with behavioral expectation
            next_q_value = rewards.flatten() + (1 - dones.flatten()) * 0.99 * behavioral_expectation
            
        return next_q_value
    
    def _compute_choquet_batch(self, values):
        """Simple Choquet expectation using behavioral package style"""
        from behavioral.distortions import choquet_expectation
        
        # Apply Choquet expectation to each batch item
        # ** CHOQUET INTEGRAL COMPUTATION HAPPENS HERE **
        # This computes ρ_g(X) = ∫_0^1 F^{-1}(u) d g(u) where:
        # - values[i] are the samples of random var§iable X (soft Q-values)
        # - self.g_plus is the probability distortion function g
        # - choquet_expectation implements the discrete estimator with weights π^(i)=g(C_i)-g(C_{i-1})
        batch_results = []
        for i in range(values.size(0)):
            result = choquet_expectation(values[i], self.g_plus)  
            batch_results.append(result)
        
        return torch.stack(batch_results)
    
    def _compute_cpt_batch(self, values, references):
        """Simple CPT computation using behavioral package style"""  
        from behavioral.distortions import cpt_functional
        
        # Apply CPT functional to each batch item
        batch_results = []
        for i in range(values.size(0)):
            result = cpt_functional(
                values[i].unsqueeze(0),  # Add sample dimension 
                references[i].unsqueeze(0),  # Add batch dimension
                self.u_plus, 
                self.u_minus,
                self.g_plus,
                self.g_plus,  # Use same distortion for gains and losses
                lambda_loss_aversion=self.behavioral_config.lambda_loss_aversion
            )
            batch_results.append(result.squeeze())
        
        return torch.stack(batch_results)
    
    def update_reference_provider(self, episode_return: float):
        """Update reference provider with episode return"""
        if hasattr(self.reference_provider, 'update'):
            self.reference_provider.update(episode_return=episode_return)


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Initialize networks
    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    
    # Initialize CPT-SAC agent
    cpt_agent = CPTSACAgent(envs, device, args.behavioral)
    
    # Optimizers
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # Performance tracking
    episode_rewards = []
    episode_lengths = []

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    episode_reward = info['episode']['r']
                    episode_length = info['episode']['l']
                    
                    print(f"global_step={global_step}, episodic_return={episode_reward}")
                    
                    # Log to tensorboard
                    writer.add_scalar("charts/episodic_return", episode_reward, global_step)
                    writer.add_scalar("charts/episodic_length", episode_length, global_step)
                    
                    # Log to wandb if tracking is enabled
                    if args.track:
                        wandb.log({
                            "charts/episodic_return": episode_reward,
                            "charts/episodic_length": episode_length,
                            "global_step": global_step
                        })
                    
                    # Store for local tracking and update reference provider
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    cpt_agent.update_reference_provider(episode_reward)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                if "final_observation" in infos:
                    real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            
            # CPT/Behavioral critic targets
            next_q_value = cpt_agent.compute_behavioral_target(
                data.next_observations, data.rewards, data.dones,
                actor, qf1_target, qf2_target, alpha
            )

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:  # Log frequently like behavioral_sac.py
                current_time = time.time()
                sps = int(global_step / (current_time - start_time))
                
                # Prepare metrics
                metrics = {
                    "losses/qf1_values": qf1_a_values.mean().item(),
                    "losses/qf2_values": qf2_a_values.mean().item(),
                    "losses/qf1_loss": qf1_loss.item(),
                    "losses/qf2_loss": qf2_loss.item(),
                    "losses/qf_loss": qf_loss.item() / 2.0,
                    "losses/actor_loss": actor_loss.item(),
                    "losses/alpha": alpha,
                    "charts/SPS": sps,
                    "global_step": global_step,
                    "behavioral/mode": args.behavioral.mode,
                    "behavioral/lambda_loss_aversion": args.behavioral.lambda_loss_aversion,
                }
                
                if args.autotune:
                    metrics["losses/alpha_loss"] = alpha_loss.item()
                
                # Add recent performance metrics
                if episode_rewards:
                    metrics["performance/mean_episode_reward"] = np.mean(episode_rewards[-10:])
                    metrics["performance/mean_episode_length"] = np.mean(episode_lengths[-10:])
                
                # Log to tensorboard
                for key, value in metrics.items():
                    if key not in ["global_step", "behavioral/mode"]:
                        writer.add_scalar(key, value, global_step)
                
                # Log to wandb if tracking
                if args.track:
                    wandb.log(metrics)
                
                # Print SPS like behavioral_sac.py
                print("SPS:", int(global_step / (time.time() - start_time)))

    # Final logging
    if args.track and episode_rewards:
        wandb.log({
            "final/total_episodes": len(episode_rewards),
            "final/mean_reward": np.mean(episode_rewards),
            "final/std_reward": np.std(episode_rewards),
            "final/max_reward": np.max(episode_rewards),
            "final/min_reward": np.min(episode_rewards)
        })

    envs.close()
    writer.close()
    if args.track:
        wandb.finish()