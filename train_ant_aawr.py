"""
AAWR (Asymmetric Advantage Weighted Regression) Training for Ant Navigation.

Adapted for continuous actions from the original AAWR implementation.

Algorithm:
1. Collect offline data with noisy expert (epsilon-greedy)
2. Train asymmetric critic (Q + V) using IQL with privileged goal info
3. Extract policy via AWR using advantage weights with continuous action NLL
"""

import torch.multiprocessing as mp

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn", force=True)

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import random
import tqdm
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt

from envs.ant_env import create_ant_envs, AntVecEnv
from models import DecisionTransformer
from get_rollout_policy import get_rollout_policy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Data Collection
# ============================================================================

class NoisyExpertPolicy:
    """Noisy expert policy for data collection."""
    
    def __init__(self, env, epsilon=0.25):
        self.env = env
        self.epsilon = epsilon
        self.action_dim = env.action_dim
    
    def get_action(self, obs):
        """Get noisy expert action (epsilon-greedy with Gaussian noise)."""
        expert_actions = self.env.opt_action(obs)
        
        # Add noise with probability epsilon
        n_envs = obs.shape[0]
        noise_mask = np.random.random(n_envs) < self.epsilon
        
        # Add Gaussian noise to expert actions
        noise = np.random.randn(n_envs, self.action_dim) * 0.3
        noisy_actions = expert_actions + noise_mask[:, None] * noise
        
        # Clip to action space
        return np.clip(noisy_actions, -1, 1).astype(np.float32)


def collect_aawr_data(env, horizon, epsilon=0.25):
    """
    Collect offline data with noisy expert policy for AAWR.
    
    Args:
        env: AntVecEnv
        horizon: Number of steps to collect per environment
        epsilon: Probability of adding noise
    
    Returns:
        List of trajectory dicts with goal information
    """
    policy = NoisyExpertPolicy(env, epsilon=epsilon)
    n_envs = env.num_envs
    
    state = env.reset()
    
    states = []
    actions = []
    expert_actions = []
    rewards = []
    dones_list = []
    next_states = []

    for t in tqdm.tqdm(range(horizon), desc="Collecting data"):
        # Get noisy action
        action = policy.get_action(state)
        
        # Get true expert action for supervision
        expert_action = env.opt_action(state)

        # Step environment
        next_state, reward, done, infos = env.step(action)
        
        # Store transition
        states.append(state.copy())
        actions.append(action.copy())
        expert_actions.append(expert_action.copy())
        rewards.append(reward.copy())
        dones_list.append(done.copy())
        next_states.append(next_state.copy())
        
        state = next_state

    # Stack arrays: (T, n_envs, dim) -> (n_envs, T, dim)
    states = np.stack(states, axis=1)
    actions = np.stack(actions, axis=1)
    expert_actions = np.stack(expert_actions, axis=1)
    rewards = np.stack(rewards, axis=1)
    dones = np.stack(dones_list, axis=1)
    next_states = np.stack(next_states, axis=1)
    
    # Create trajectory dicts with goal info
    trajs = []
    for k in range(n_envs):
        traj = {
            "states": states[k],
            "actions": actions[k],
            "expert_actions": expert_actions[k],
            "rewards": rewards[k],
            "dones": dones[k],
            "next_states": next_states[k],
            "goal": env._goals[k].copy(),  # Privileged info
        }
        trajs.append(traj)
    
    return trajs


class AAWRDataset(torch.utils.data.Dataset):
    """Dataset for AAWR that includes goal information."""
    
    def __init__(self, trajs, config):
        self.trajs = trajs
        self.config = config
        self.horizon = config['horizon']
    
    def __len__(self):
        return len(self.trajs)
    
    def __getitem__(self, index):
        traj = self.trajs[index]
        return {
            'states': torch.from_numpy(traj['states']).float(),
            'actions': torch.from_numpy(traj['actions']).float(),
            'expert_actions': torch.from_numpy(traj['expert_actions']).float(),
            'rewards': torch.from_numpy(traj['rewards']).float(),
            'dones': torch.from_numpy(traj['dones']).float(),
            'next_states': torch.from_numpy(traj['next_states']).float(),
            'goal': torch.from_numpy(traj['goal']).float(),
        }


def aawr_collate_fn(batch):
    """Collate function for AAWR dataset."""
    from torch.nn.utils.rnn import pad_sequence
    
    padded_batch = {}
    for key in batch[0]:
        if key == 'goal':
            # Goals don't need padding, just stack
            padded_batch[key] = torch.stack([item[key] for item in batch])
        else:
            padded_batch[key] = pad_sequence(
                [item[key] for item in batch], 
                batch_first=True
            )
    
    # Create attention mask
    lengths = torch.tensor([item['states'].shape[0] for item in batch])
    max_len = lengths.max()
    attention_mask = torch.zeros((len(lengths), max_len), dtype=torch.bool)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1
    
    padded_batch['attention_mask'] = attention_mask
    return padded_batch


# ============================================================================
# Asymmetric Critic for IQL
# ============================================================================

class AsymmetricCritic(nn.Module):
    """
    Asymmetric critic that uses privileged goal information.
    
    - Q-network: Q(s, a, g) - state-action value with goal
    - V-network: V(s, g) - state value with goal
    """
    
    def __init__(self, state_dim, action_dim, goal_dim, hidden_dim=256):
        super().__init__()
        
        # Q-network: inputs state + action + goal
        self.q_net = nn.Sequential(
            nn.Linear(state_dim + action_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # V-network: inputs state + goal
        self.v_net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Target networks
        self.q_target = None
        self.v_target = None
    
    def init_target(self):
        """Initialize target networks."""
        import copy
        self.q_target = copy.deepcopy(self.q_net)
        self.v_target = copy.deepcopy(self.v_net)
        for param in self.q_target.parameters():
            param.requires_grad = False
        for param in self.v_target.parameters():
            param.requires_grad = False
    
    def update_target(self, tau=0.005):
        """Soft update target networks."""
        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.v_net.parameters(), self.v_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def q_value(self, state, action, goal):
        """Compute Q(s, a, g)."""
        x = torch.cat([state, action, goal], dim=-1)
        return self.q_net(x)
    
    def v_value(self, state, goal):
        """Compute V(s, g)."""
        x = torch.cat([state, goal], dim=-1)
        return self.v_net(x)
    
    def advantage(self, state, action, goal):
        """Compute advantage A(s, a, g) = Q(s, a, g) - V(s, g)."""
        q = self.q_value(state, action, goal)
        v = self.v_value(state, goal)
        return (q - v).squeeze(-1)


# ============================================================================
# IQL Training (Critic)
# ============================================================================

def expectile_loss(pred, target, expectile=0.9):
    """Expectile regression loss for IQL."""
    diff = target - pred
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return (weight * (diff ** 2)).mean()


def train_iql_critic(critic, train_loader, args, save_dir):
    """Train the asymmetric critic using IQL."""
    os.makedirs(save_dir, exist_ok=True)
    
    critic.init_target()
    optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr)
    
    for epoch in tqdm.tqdm(range(args.critic_epochs), desc="Training IQL Critic"):
        critic.train()
        epoch_stats = defaultdict(list)
        
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            B, T = batch['states'].shape[:2]
            mask = batch['attention_mask']
            
            # Expand goal to match sequence length
            goal_expanded = batch['goal'].unsqueeze(1).expand(-1, T, -1)
            
            # Flatten all tensors
            states_flat = batch['states'].reshape(-1, batch['states'].shape[-1])
            actions_flat = batch['actions'].reshape(-1, batch['actions'].shape[-1])
            next_states_flat = batch['next_states'].reshape(-1, batch['next_states'].shape[-1])
            rewards_flat = batch['rewards'].reshape(-1)
            dones_flat = batch['dones'].reshape(-1)
            goal_flat = goal_expanded.reshape(-1, goal_expanded.shape[-1])
            mask_flat = mask.reshape(-1)
            
            valid_idx = mask_flat > 0
            if valid_idx.sum() == 0:
                continue
                
            states = states_flat[valid_idx]
            actions = actions_flat[valid_idx]
            next_states = next_states_flat[valid_idx]
            rewards = rewards_flat[valid_idx]
            dones = dones_flat[valid_idx]
            goals = goal_flat[valid_idx]
            
            # V-network update (expectile regression on Q values)
            with torch.no_grad():
                q_values = critic.q_value(states, actions, goals).squeeze(-1)
            
            v_values = critic.v_value(states, goals).squeeze(-1)
            v_loss = expectile_loss(v_values, q_values, args.expectile)
            
            # Q-network update (TD loss)
            with torch.no_grad():
                next_v = critic.v_value(next_states, goals).squeeze(-1)
                q_target = rewards + args.gamma * next_v * (1 - dones)
            
            q_pred = critic.q_value(states, actions, goals).squeeze(-1)
            q_loss = F.mse_loss(q_pred, q_target)
            
            loss = v_loss + q_loss
            
            optimizer.zero_grad()
            loss.backward()
            if args.gradient_clip:
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()
            
            critic.update_target(tau=args.target_tau)
            
            epoch_stats['v_loss'].append(v_loss.item())
            epoch_stats['q_loss'].append(q_loss.item())
            epoch_stats['q_mean'].append(q_pred.mean().item())
            epoch_stats['v_mean'].append(v_values.mean().item())
        
        if args.log_wandb:
            for k, v in epoch_stats.items():
                wandb.log({f"iql/{k}": np.mean(v), "iql/epoch": epoch})
        
        if epoch % max(1, args.critic_epochs // 10) == 0:
            print(f"Epoch {epoch}: Q_loss={np.mean(epoch_stats['q_loss']):.4f}, "
                  f"V_loss={np.mean(epoch_stats['v_loss']):.4f}")
    
    torch.save(critic.state_dict(), os.path.join(save_dir, "critic_final.pth"))
    return critic


# ============================================================================
# AWR Policy Training (Continuous Actions)
# ============================================================================

def get_loss_mask(attention_mask, horizon):
    """Get mask for loss computation on last horizon tokens."""
    loss_mask = torch.zeros_like(attention_mask)
    for i in range(loss_mask.size(0)):
        non_zero_indices = torch.nonzero(attention_mask[i], as_tuple=False).squeeze()
        if len(non_zero_indices.shape) == 0:
            non_zero_indices = non_zero_indices.unsqueeze(0)
        if len(non_zero_indices) >= horizon:
            loss_mask[i, non_zero_indices[-horizon:]] = 1
        else:
            loss_mask[i, non_zero_indices] = 1
    return loss_mask


def train_awr_policy(model, critic, train_loader, args, save_dir, action_dim, env_horizon):
    """
    Train policy using AWR with continuous action NLL loss.
    
    AWR: weight = exp(A / temperature), where A = Q(s,a,g) - V(s,g)
    Policy loss: weighted NLL (negative log-likelihood) of expert actions
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Freeze critic
    critic.eval()
    for param in critic.parameters():
        param.requires_grad = False
    
    total_steps = len(train_loader) * args.policy_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_steps]
    )
    
    for epoch in tqdm.tqdm(range(args.policy_epochs), desc="Training AWR Policy"):
        model.train()
        epoch_stats = defaultdict(list)
        
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get policy predictions (returns mean and std for continuous actions)
            pred_actions, action_stds = model(batch, sample_time=True)
            true_actions = batch['expert_actions']
            
            # Compute advantages using frozen critic
            B, T = batch['states'].shape[:2]
            goal_expanded = batch['goal'].unsqueeze(1).expand(-1, T, -1)
            
            with torch.no_grad():
                states_flat = batch['states'].reshape(-1, batch['states'].shape[-1])
                actions_flat = batch['actions'].reshape(-1, batch['actions'].shape[-1])
                goal_flat = goal_expanded.reshape(-1, goal_expanded.shape[-1])
                
                advantages = critic.advantage(states_flat, actions_flat, goal_flat)
                advantages = advantages.reshape(B, T)
                
                # AWR weights
                advantages_clipped = torch.clamp(advantages / args.awr_temperature, -10, 10)
                weights = torch.exp(advantages_clipped)
                
                if args.awr_filter == "indicator":
                    weights = (advantages > 0).float()
                elif args.awr_filter == "exp_clamp":
                    weights = torch.clamp(weights, 0, 100)
            
            # Continuous action NLL loss (Gaussian)
            # log_prob = -0.5 * ((x - mu) / sigma)^2 - log(sigma) - 0.5 * log(2 * pi)
            # NLL = -log_prob = 0.5 * ((x - mu) / sigma)^2 + log(sigma) + const
            diff = true_actions - pred_actions
            var = action_stds ** 2 + 1e-6
            nll = 0.5 * (diff ** 2 / var) + torch.log(action_stds + 1e-6)
            nll = nll.sum(-1)  # Sum over action dimensions
            
            # Apply loss mask (only last env_horizon tokens)
            loss_mask = get_loss_mask(batch['attention_mask'], env_horizon)
            
            # Weighted loss
            weighted_loss = nll * weights * loss_mask
            loss = weighted_loss.sum() / (loss_mask.sum() + 1e-8)
            
            optimizer.zero_grad()
            loss.backward()
            
            if args.gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_stats['loss'].append(loss.item())
            epoch_stats['advantage_mean'].append(advantages.mean().item())
            epoch_stats['weight_mean'].append(weights.mean().item())
        
        if args.log_wandb:
            for k, v in epoch_stats.items():
                wandb.log({f"awr/{k}": np.mean(v), "awr/epoch": epoch})
            wandb.log({"awr/lr": optimizer.param_groups[0]['lr']})
        
        if epoch % max(1, args.policy_epochs // 10) == 0:
            print(f"Epoch {epoch}: loss={np.mean(epoch_stats['loss']):.4f}")
        
        if epoch % max(1, args.policy_epochs // 5) == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch}.pth"))
    
    torch.save(model.state_dict(), os.path.join(save_dir, "model_final.pth"))
    return model


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_policy(model, env, num_episodes, env_horizon, context_horizon=4000):
    """Evaluate policy on environment."""
    from get_rollout_policy import get_rollout_policy
    
    eval_policy = get_rollout_policy(
        "decision_transformer",
        model=model,
        context_horizon=context_horizon,
        env_horizon=env_horizon,
        context_accumulation=False,
        sliding_window=False,
        continuous_action=True,
        low_noise_eval=True,
    )
    
    obs = env.reset()
    eval_policy.reset()
    
    episode_returns = []
    current_returns = np.zeros(env.num_envs)
    episodes_done = np.zeros(env.num_envs, dtype=int)
    step = 0
    
    while min(episodes_done) < num_episodes:
        action = eval_policy.get_action(obs)
        obs, rewards, dones, infos = env.step(action)
        current_returns += rewards
        
        eval_policy.update_context(obs, action, rewards, dones)
        step += 1
        
        # Check for episode completions
        for i in range(env.num_envs):
            if step % env_horizon == 0:
                episode_returns.append(current_returns[i])
                current_returns[i] = 0
                episodes_done[i] += 1
    
    return episode_returns


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AAWR Training for Ant")
    
    # Experiment
    parser.add_argument("--exp_name", type=str, default="ant-aawr")
    parser.add_argument("--seed", type=int, default=0)
    
    # Environment
    parser.add_argument("--num_goals", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--n_envs", type=int, default=100)
    parser.add_argument("--n_meta_episodes", type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=20)
    
    # Data collection
    parser.add_argument("--epsilon", type=float, default=0.25, help="Noisy expert epsilon")
    
    # Model
    parser.add_argument("--context_horizon", type=int, default=4000)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Critic (IQL)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--critic_epochs", type=int, default=100)
    parser.add_argument("--critic_hidden_dim", type=int, default=256)
    parser.add_argument("--expectile", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target_tau", type=float, default=0.005)
    
    # Policy (AWR)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--policy_epochs", type=int, default=100)
    parser.add_argument("--awr_temperature", type=float, default=3.0)
    parser.add_argument("--awr_filter", type=str, default="exp_clamp", 
                        choices=["none", "indicator", "exp_clamp"])
    
    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--gradient_clip", action="store_true")
    
    # Logging
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dpt-sweep")
    parser.add_argument("--wandb_entity", type=str, default="sriyash")
    parser.add_argument("--save_dir", type=str, default="./results/ant_aawr")

    args = parser.parse_args()

    # Initialize wandb
    if args.log_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"{args.exp_name}-seed{args.seed}",
        )

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Using device: {device}")
    
    save_dir = os.path.join(args.save_dir, f"{args.exp_name}-seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    # Create environments
    print(f"Creating Ant environments...")
    train_envs, test_envs, eval_envs = create_ant_envs(
        num_goals=args.num_goals,
        dataset_size=1000,
        n_envs=args.n_envs,
        horizon=args.horizon,
        seed=args.seed,
    )
    
    train_env = train_envs[0]
    eval_env = eval_envs[0]
    
    state_dim = train_env.state_dim
    action_dim = train_env.action_dim
    goal_dim = 2
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}, Goal dim: {goal_dim}")

    # ========================================================================
    # Phase 1: Collect offline data
    # ========================================================================
    print("\n" + "="*60)
    print("Phase 1: Collecting data with noisy expert")
    print("="*60)
    
    data_dir = os.path.join(save_dir, "data")
    train_data_path = os.path.join(data_dir, "train_data.pkl")
    
    if os.path.exists(train_data_path):
        print(f"Loading existing data from {data_dir}")
        with open(train_data_path, "rb") as f:
            train_trajs = pickle.load(f)
    else:
        os.makedirs(data_dir, exist_ok=True)
        data_horizon = args.n_meta_episodes * args.horizon
        train_trajs = collect_aawr_data(train_env, data_horizon, args.epsilon)
        
        with open(train_data_path, "wb") as f:
            pickle.dump(train_trajs, f)
    
    print(f"Train trajectories: {len(train_trajs)}")

    # Create dataset
    config = {
        "horizon": args.context_horizon,
        "state_dim": state_dim,
        "action_dim": action_dim,
    }
    train_dataset = AAWRDataset(train_trajs, config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=aawr_collate_fn,
    )

    # ========================================================================
    # Phase 2: Train IQL critic
    # ========================================================================
    print("\n" + "="*60)
    print("Phase 2: Training IQL Critic")
    print("="*60)
    
    critic = AsymmetricCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        hidden_dim=args.critic_hidden_dim,
    ).to(device)
    
    print(f"Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")
    
    critic = train_iql_critic(
        critic=critic,
        train_loader=train_loader,
        args=args,
        save_dir=os.path.join(save_dir, "critic"),
    )

    # ========================================================================
    # Phase 3: Train policy via AWR
    # ========================================================================
    print("\n" + "="*60)
    print("Phase 3: Training Policy via AWR")
    print("="*60)
    
    model_args = {
        "horizon": args.context_horizon,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "n_layer": args.num_layers,
        "n_head": args.num_heads,
        "n_embd": args.n_embd,
        "dropout": args.dropout,
        "shuffle": True,
        "test": False,
        "continuous_action": True,
        "gmm_heads": 1,
        "std_min": 0.007,
        "std_max": 2.0,
        "init_std": 0.3,
    }
    
    with open(os.path.join(save_dir, "model_args.pkl"), "wb") as f:
        pickle.dump(model_args, f)
    
    model = DecisionTransformer(model_args).to(device)
    print(f"Policy parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model = train_awr_policy(
        model=model,
        critic=critic,
        train_loader=train_loader,
        args=args,
        save_dir=os.path.join(save_dir, "policy"),
        action_dim=action_dim,
        env_horizon=args.horizon,
    )

    # ========================================================================
    # Phase 4: Evaluation
    # ========================================================================
    print("\n" + "="*60)
    print("Phase 4: Evaluation")
    print("="*60)
    
    episode_returns = evaluate_policy(
        model, eval_env, args.eval_episodes, args.horizon, args.context_horizon
    )
    
    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    
    print(f"Evaluation: {mean_return:.2f} ± {std_return:.2f}")
    
    if args.log_wandb:
        wandb.log({
            "eval/final_return": mean_return,
            "eval/final_return_std": std_return,
        })
    
    # Save results
    with open(os.path.join(save_dir, "eval_results.pkl"), "wb") as f:
        pickle.dump({"returns": episode_returns}, f)
    
    print(f"\nTraining complete! Results saved to {save_dir}")
    
    train_env.close()
    eval_env.close()
    
    if args.log_wandb:
        wandb.finish()

