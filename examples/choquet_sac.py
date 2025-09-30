# Choquet SAC - Modified from behavioral_sac.py to use Choquet integral with Prelec distortion
# CHANGES FROM BEHAVIORAL_SAC.PY:
# 1. Added behavioral imports for Choquet integral and Prelec distortion 
# 2. Added configuration for distortion function and discounting mode
# 3. Replaced standard expectation with Choquet integral in target computation (line ~285)
# 4. Added support for hyperbolic discounting instead of fixed gamma
# 5. Added multiple action sampling for proper Choquet expectation computation

import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from cleanrl_utils.buffers import ReplayBuffer

# CHANGE 1: Import behavioral utilities for Choquet integral and discounting
from behavioral.distortions import PrelecDistortion, choquet_expectation
from behavioral.discounting import create_discounting_mode


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
    wandb_project_name: str = "risk-averse-choquet-sac"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Pendulum-v1"
    """the environment id of the task"""
    total_timesteps: int = 50000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e5)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma (used for standard discounting)"""
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
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    
    # CHANGE 2: Added Choquet integral configuration
    use_choquet: bool = True
    """whether to use Choquet integral instead of standard expectation"""
    n_action_samples: int = 12
    """number of action samples for Choquet expectation computation"""
    prelec_alpha: float = 0.35
    """Prelec distortion parameter alpha (< 1.0 for risk aversion) - VERY risk averse"""
    prelec_eta: float = 0.8
    """Prelec distortion parameter eta - increases curvature"""
    
    # CHANGE 3: Added discounting mode configuration
    discounting_mode: str = "standard"
    """discounting mode: standard, beta_delta, or hyperbolic_mixture"""
    beta: float = 1.0
    """beta parameter for beta-delta discounting"""
    delta: float = 0.99
    """delta parameter for beta-delta discounting"""
    hyperbolic_gammas: str = "0.99,0.95,0.90"
    """comma-separated gamma values for hyperbolic mixture"""
    hyperbolic_probs: str = "0.6,0.3,0.1"
    """comma-separated probability weights for hyperbolic mixture"""


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
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

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


# CHANGE 4: Fixed utility function to compute Choquet expectation over batch
def compute_choquet_expectation_batch(values_batch, distortion_fn):
    """
    Compute Choquet expectation for a batch of value samples.
    
    Args:
        values_batch: tensor of shape [batch_size, n_samples]
        distortion_fn: Prelec distortion function
        
    Returns:
        Choquet expectations of shape [batch_size]
    """
    # The choquet_expectation function expects samples in the last dimension
    # values_batch is already [batch_size, n_samples], which is correct
    return choquet_expectation(values_batch, distortion_fn)


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

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # CHANGE 5: Initialize Prelec distortion function
    prelec_distortion = PrelecDistortion(alpha=args.prelec_alpha, eta=args.prelec_eta)
    
    # CHANGE 6: Initialize discounting mode
    if args.discounting_mode == "standard":
        discounting_params = {"gamma": args.gamma}
    elif args.discounting_mode == "beta_delta":
        discounting_params = {"beta": args.beta, "delta": args.delta}
    elif args.discounting_mode == "hyperbolic_mixture":
        gammas = [float(x) for x in args.hyperbolic_gammas.split(",")]
        probs = [float(x) for x in args.hyperbolic_probs.split(",")]
        discounting_params = {"mixture": {"gammas": gammas, "probs": probs}}
    else:
        raise ValueError(f"Unknown discounting mode: {args.discounting_mode}")
    
    discounting_mode = create_discounting_mode(args.discounting_mode, discounting_params)
    effective_gamma = discounting_mode.get_effective_gamma()
    
    print(f"Using discounting mode: {args.discounting_mode}")
    print(f"Effective gamma: {effective_gamma}")
    print(f"Using Choquet integral: {args.use_choquet}")
    if args.use_choquet:
        print(f"Prelec parameters: alpha={args.prelec_alpha}, eta={args.prelec_eta}")

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
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            # Store behavioral metrics for logging
            behavioral_metrics = {}
            with torch.no_grad():
                if args.use_choquet:
                    # CHANGE 7: Replace standard expectation with Choquet integral
                    # Sample multiple actions for proper expectation computation
                    batch_size = data.next_observations.size(0)
                    
                    # Expand observations to sample multiple actions
                    next_obs_expanded = data.next_observations.unsqueeze(1).expand(-1, args.n_action_samples, -1)
                    next_obs_flat = next_obs_expanded.reshape(-1, data.next_observations.shape[-1])
                    
                    # Sample multiple actions
                    next_actions_flat, next_log_probs_flat, _ = actor.get_action(next_obs_flat)
                    
                    # Compute Q-values for all action samples
                    qf1_next_flat = qf1_target(next_obs_flat, next_actions_flat)
                    qf2_next_flat = qf2_target(next_obs_flat, next_actions_flat)
                    
                    # Reshape to [batch_size, n_action_samples]
                    qf1_next = qf1_next_flat.reshape(batch_size, args.n_action_samples)
                    qf2_next = qf2_next_flat.reshape(batch_size, args.n_action_samples)
                    next_log_probs = next_log_probs_flat.reshape(batch_size, args.n_action_samples)
                    
                    # Compute soft Q-values: min(Q1, Q2) - α * log_prob
                    min_qf_next = torch.min(qf1_next, qf2_next)
                    soft_q_values = min_qf_next - alpha * next_log_probs  # [batch_size, n_action_samples]
                    
                    # ** CHOQUET INTEGRAL COMPUTATION **
                    # Apply Choquet expectation with Prelec distortion instead of standard mean
                    choquet_expectation_values = compute_choquet_expectation_batch(soft_q_values, prelec_distortion)
                    
                    # Ensure proper tensor shape for target computation
                    # choquet_expectation_values should be [batch_size], same as rewards.flatten()
                    if choquet_expectation_values.dim() > 1:
                        choquet_expectation_values = choquet_expectation_values.view(-1)
                    
                    # Use Choquet expectation in target computation
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * effective_gamma * choquet_expectation_values
                    
                    # Store behavioral metrics for logging
                    with torch.no_grad():
                        standard_target = min_qf_next.mean(dim=1) - alpha * next_log_probs.mean(dim=1)
                        behavioral_metrics = {
                            'choquet_target_mean': choquet_expectation_values.mean().item(),
                            'choquet_target_std': choquet_expectation_values.std().item(),
                            'standard_target_mean': standard_target.mean().item(),
                            'distortion_bias': (choquet_expectation_values.mean() - standard_target.mean()).item()
                        }
                else:
                    # Standard SAC expectation (original behavioral_sac.py behavior)
                    next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                    qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * effective_gamma * (min_qf_next_target).view(-1)

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
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
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

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("charts/effective_gamma", effective_gamma, global_step)
                
                # Enhanced behavioral tracking for risk aversion analysis
                if args.use_choquet:
                    writer.add_scalar("behavioral/prelec_alpha", args.prelec_alpha, global_step)
                    writer.add_scalar("behavioral/prelec_eta", args.prelec_eta, global_step)
                    writer.add_scalar("behavioral/n_action_samples", args.n_action_samples, global_step)
                    
                    # Track Q-value statistics for risk analysis
                    writer.add_scalar("behavioral/q1_mean", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("behavioral/q1_std", qf1_a_values.std().item(), global_step)
                    writer.add_scalar("behavioral/q2_mean", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar("behavioral/q2_std", qf2_a_values.std().item(), global_step)
                    
                    # Track reward statistics
                    writer.add_scalar("behavioral/reward_mean", data.rewards.mean().item(), global_step)
                    writer.add_scalar("behavioral/reward_std", data.rewards.std().item(), global_step)
                    
                    # Track behavioral metrics if available
                    if behavioral_metrics:
                        writer.add_scalar("behavioral/choquet_target_mean", behavioral_metrics['choquet_target_mean'], global_step)
                        writer.add_scalar("behavioral/choquet_target_std", behavioral_metrics['choquet_target_std'], global_step)
                        writer.add_scalar("behavioral/standard_target_mean", behavioral_metrics['standard_target_mean'], global_step)
                        writer.add_scalar("behavioral/distortion_bias", behavioral_metrics['distortion_bias'], global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()