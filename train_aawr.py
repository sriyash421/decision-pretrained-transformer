"""
AAWR (Asymmetric Advantage Weighted Regression) Training Algorithm.

Based on: "Real-world RL for Active Perception Behaviors" (Penn PAL Lab)
https://github.com/penn-pal-lab/aawr

The key idea is to use privileged information (goal) during critic training
to compute high-quality advantages, then use advantage-weighted BC for policy.

Algorithm:
1. Collect offline data with noisy expert (epsilon-greedy)
2. Train asymmetric critic (Q + V) using IQL with privileged goal info
3. Extract policy via AWR using advantage weights
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
import gym
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt

from create_envs import create_env
from dataset import collate_fn, SequenceDataset
from eval_policy import evaluate_policy_on_envs
from models import DecisionTransformer, AsymmetricCritic
from get_rollout_policy import get_rollout_policy, NoisyExpertPolicy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Data Collection
# ============================================================================

def collect_aawr_data(envs, horizon, epsilon=0.25):
    """
    Collect offline data with noisy expert policy for AAWR.
    
    Args:
        envs: List of vectorized environments
        horizon: Number of steps to collect per environment
        epsilon: Probability of random action
    
    Returns:
        List of trajectory dicts with goal information
    """
    policy = NoisyExpertPolicy(epsilon=epsilon)
    trajs = []
    
    for env in tqdm.tqdm(envs, desc="Collecting AAWR data"):
        policy.set_env(env)
        state = env.reset()
        policy.reset()
        n_envs = env.num_envs
        
        states = []
        actions = []
        expert_actions = []
        rewards = []
        dones_list = []
        next_states = []

        for t in range(horizon):
            # Get noisy expert action
            action = policy.get_action(state)
            
            # Get true expert action for supervision
            if hasattr(env, "have_keys"):
                expert_action = env.opt_action(state, env.have_keys)
            else:
                expert_action = env.opt_action(state)

            # Step environment
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action)
            expert_actions.append(expert_action)
            rewards.append(reward)
            dones_list.append(done)
            next_states.append(next_state)
            
            # Update policy context
            policy.update_context(state, action, reward, done)
            
            # Handle episode resets
            if np.any(done):
                next_state = env.reset()
            
            state = next_state

        # Stack arrays
        states = np.stack(states, axis=1)
        actions = np.stack(actions, axis=1)
        expert_actions = np.stack(expert_actions, axis=1)
        rewards = np.stack(rewards, axis=1)
        dones = np.stack(dones_list, axis=1)
        next_states = np.stack(next_states, axis=1)
        
        # Create trajectory dicts with goal info
        for k in range(n_envs):
            # Handle different goal access patterns
            if hasattr(env, '_goals'):
                # Ant env uses _goals array
                goal = env._goals[k].copy()
            else:
                # Darkroom env uses _envs[k].goal
                goal = env._envs[k].goal.copy()
            
            traj = {
                "states": states[k],
                "actions": actions[k],
                "expert_actions": expert_actions[k],
                "rewards": rewards[k],
                "dones": dones[k],
                "next_states": next_states[k],
                "goal": goal,  # Privileged info
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
# IQL Training (Critic)
# ============================================================================

def expectile_loss(pred, target, expectile=0.9):
    """
    Expectile regression loss for IQL.
    
    When expectile > 0.5, this emphasizes the upper quantile of the target
    distribution, effectively learning an optimistic value estimate.
    
    Args:
        pred: Predicted values
        target: Target values
        expectile: Expectile parameter (0.9 emphasizes 90th percentile)
    
    Returns:
        Loss value
    """
    diff = target - pred
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return (weight * (diff ** 2)).mean()


def train_iql_critic(
    critic,
    train_loader,
    args,
    save_dir,
):
    """
    Train the asymmetric critic using IQL.
    
    IQL training:
    - Q-network: TD loss with target Q
    - V-network: Expectile regression on Q values
    
    Args:
        critic: AsymmetricCritic model
        train_loader: DataLoader with AAWR dataset
        args: Training arguments
        save_dir: Directory to save checkpoints
    
    Returns:
        Trained critic
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize target network
    critic.init_target()
    
    # Optimizer
    optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr)
    
    # Training loop
    global_step = 0
    
    for epoch in tqdm.tqdm(range(args.critic_epochs), desc="Training IQL Critic"):
        critic.train()
        epoch_stats = defaultdict(list)
        
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Flatten batch for transition-level training
            # batch['states']: (B, T, state_dim)
            # batch['actions']: (B, T, action_dim)
            # batch['rewards']: (B, T)
            # batch['dones']: (B, T)
            # batch['next_states']: (B, T, state_dim)
            # batch['goal']: (B, goal_dim)
            
            B, T = batch['states'].shape[:2]
            mask = batch['attention_mask']  # (B, T)
            
            # Expand goal to match sequence length
            goal_expanded = batch['goal'].unsqueeze(1).expand(-1, T, -1)  # (B, T, goal_dim)
            
            # Flatten all tensors
            states_flat = batch['states'].reshape(-1, batch['states'].shape[-1])
            actions_flat = batch['actions'].reshape(-1, batch['actions'].shape[-1])
            next_states_flat = batch['next_states'].reshape(-1, batch['next_states'].shape[-1])
            rewards_flat = batch['rewards'].reshape(-1)
            dones_flat = batch['dones'].reshape(-1)
            goal_flat = goal_expanded.reshape(-1, goal_expanded.shape[-1])
            mask_flat = mask.reshape(-1)
            
            # Only train on valid (non-padded) transitions
            valid_idx = mask_flat > 0
            if valid_idx.sum() == 0:
                continue
                
            states = states_flat[valid_idx]
            actions = actions_flat[valid_idx]
            next_states = next_states_flat[valid_idx]
            rewards = rewards_flat[valid_idx]
            dones = dones_flat[valid_idx]
            goals = goal_flat[valid_idx]
            
            # ==================== V-network update ====================
            # V-network is trained with expectile regression on Q values
            with torch.no_grad():
                # Get Q value for current (s, a) pair
                q_values = critic.q_value(states, actions, goals).squeeze(-1)
            
            v_values = critic.v_value(states, goals).squeeze(-1)
            v_loss = expectile_loss(v_values, q_values, args.expectile)
            
            # ==================== Q-network update ====================
            # Q-network: TD loss with bootstrapped target
            with torch.no_grad():
                # V(s') for bootstrapping
                next_v = critic.v_value(next_states, goals).squeeze(-1)
                # TD target: r + gamma * V(s') * (1 - done)
                q_target = rewards + args.gamma * next_v * (1 - dones)
            
            q_pred = critic.q_value(states, actions, goals).squeeze(-1)
            q_loss = F.mse_loss(q_pred, q_target)
            
            # Total loss
            loss = v_loss + q_loss
            
            # Update
            optimizer.zero_grad()
            loss.backward()
            if args.gradient_clip:
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()
            
            # Update target network
            critic.update_target(tau=args.target_tau)
            
            # Log stats
            epoch_stats['v_loss'].append(v_loss.item())
            epoch_stats['q_loss'].append(q_loss.item())
            epoch_stats['total_loss'].append(loss.item())
            epoch_stats['q_mean'].append(q_pred.mean().item())
            epoch_stats['v_mean'].append(v_values.mean().item())
            
            global_step += 1
        
        # Log epoch stats
        if args.log_wandb:
            for k, v in epoch_stats.items():
                wandb.log({f"iql/{k}": np.mean(v), "iql/epoch": epoch})
        
        if epoch % max(1, args.critic_epochs // 10) == 0:
            print(f"Epoch {epoch}: Q_loss={np.mean(epoch_stats['q_loss']):.4f}, "
                  f"V_loss={np.mean(epoch_stats['v_loss']):.4f}")
        
        # Save checkpoint
        if epoch % max(1, args.critic_epochs // 5) == 0:
            torch.save(critic.state_dict(), os.path.join(save_dir, f"critic_epoch_{epoch}.pth"))
    
    # Save final critic
    torch.save(critic.state_dict(), os.path.join(save_dir, "critic_final.pth"))
    
    return critic


# ============================================================================
# Joint Critic + Policy Training
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


def train_aawr_joint(
    model,
    critic,
    train_loader,
    args,
    save_dir,
    action_dim,
    env_horizon,
    continuous_action=False,
):
    """
    Joint training of critic (IQL) and policy (AWR) for specified gradient steps.
    
    Each gradient step:
    1. Update critic with IQL (V-network expectile loss + Q-network TD loss)
    2. Update policy with AWR (advantage-weighted BC)
    
    Args:
        model: Policy model (DecisionTransformer)
        critic: AsymmetricCritic model
        train_loader: DataLoader with AAWR dataset
        args: Training arguments
        save_dir: Directory to save checkpoints
        action_dim: Action dimension
        env_horizon: Environment horizon
        continuous_action: Whether actions are continuous
    
    Returns:
        Trained model and critic
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize critic target network
    critic.init_target()
    
    # Setup optimizers
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr)
    policy_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Setup scheduler for policy
    warmup_steps = int(args.total_gradient_steps * args.warmup_ratio)
    warmup = torch.optim.lr_scheduler.LinearLR(
        policy_optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        policy_optimizer, T_max=max(1, args.total_gradient_steps - warmup_steps)
    )
    policy_scheduler = torch.optim.lr_scheduler.SequentialLR(
        policy_optimizer, [warmup, cosine], milestones=[warmup_steps]
    )
    
    # Training loop
    global_step = 0
    data_iter = iter(train_loader)
    
    log_freq = max(1, args.total_gradient_steps // 100)
    save_freq = max(1, args.total_gradient_steps // 10)
    
    pbar = tqdm.tqdm(total=args.total_gradient_steps, desc="Training AAWR (joint)")
    
    while global_step < args.total_gradient_steps:
        # Get batch (cycle through dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        B, T = batch['states'].shape[:2]
        mask = batch['attention_mask']
        goal_expanded = batch['goal'].unsqueeze(1).expand(-1, T, -1)
        
        # Flatten tensors for critic
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
        
        # ==================== Critic Update (IQL) ====================
        critic.train()
        
        # V-network: expectile regression on Q values
        with torch.no_grad():
            q_values = critic.q_value(states, actions, goals).squeeze(-1)
        
        v_values = critic.v_value(states, goals).squeeze(-1)
        v_loss = expectile_loss(v_values, q_values, args.expectile)
        
        # Q-network: TD loss
        with torch.no_grad():
            next_v = critic.v_value(next_states, goals).squeeze(-1)
            q_target = rewards + args.gamma * next_v * (1 - dones)
        
        q_pred = critic.q_value(states, actions, goals).squeeze(-1)
        q_loss = F.mse_loss(q_pred, q_target)
        
        critic_loss = v_loss + q_loss
        
        critic_optimizer.zero_grad()
        critic_loss.backward()
        if args.gradient_clip:
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_optimizer.step()
        
        # Update target network
        critic.update_target(tau=args.target_tau)
        
        # ==================== Policy Update (AWR) ====================
        model.train()
        critic.eval()
        
        # Get policy predictions
        pred_actions, pred_stds = model(batch, sample_time=False)
        true_actions = batch['expert_actions']
        
        # Compute advantages using critic (no grad)
        with torch.no_grad():
            advantages = critic.advantage(states_flat[valid_idx], actions_flat[valid_idx], goal_flat[valid_idx])
            # Map back to (B, T) shape
            adv_full = torch.zeros(B * T, device=device)
            adv_full[valid_idx] = advantages
            advantages_2d = adv_full.reshape(B, T)
            
            # AWR weights
            advantages_clipped = torch.clamp(advantages_2d / args.awr_temperature, -10, 10)
            weights = torch.exp(advantages_clipped)
            
            if args.awr_filter == "indicator":
                weights = (advantages_2d > 0).float()
            elif args.awr_filter == "exp_clamp":
                weights = torch.clamp(weights, 0, 100)
        
        # Compute action loss
        if continuous_action:
            # Gaussian NLL loss for continuous actions
            diff = true_actions - pred_actions
            var = pred_stds ** 2 + 1e-6
            nll = 0.5 * (diff ** 2 / var) + torch.log(pred_stds + 1e-6)
            action_loss = nll.sum(-1)  # Sum over action dims -> (B, T)
        else:
            # Cross entropy loss for discrete actions
            # pred_actions: (B, T, action_dim) logits
            # true_actions: (B, T) indices OR (B, T, 1) indices
            if len(true_actions.shape) == 3:
                true_actions = true_actions.squeeze(-1)
            action_loss = F.cross_entropy(
                pred_actions.reshape(-1, action_dim),
                true_actions.reshape(-1).long(),
                reduction='none'
            )
            action_loss = action_loss.reshape(B, T)
        
        # Apply loss mask
        loss_mask = get_loss_mask(batch['attention_mask'], env_horizon).float()
        
        # Weighted loss
        weighted_loss = action_loss * weights * loss_mask
        policy_loss = weighted_loss.sum() / (loss_mask.sum() + 1e-8)
        
        policy_optimizer.zero_grad()
        policy_loss.backward()
        if args.gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        policy_optimizer.step()
        policy_scheduler.step()
        
        global_step += 1
        pbar.update(1)
        
        # Logging
        if global_step % log_freq == 0:
            if args.log_wandb:
                wandb.log({
                    "train/critic_loss": critic_loss.item(),
                    "train/v_loss": v_loss.item(),
                    "train/q_loss": q_loss.item(),
                    "train/policy_loss": policy_loss.item(),
                    "train/advantage_mean": advantages.mean().item(),
                    "train/weight_mean": weights.mean().item(),
                    "train/lr": policy_optimizer.param_groups[0]['lr'],
                    "train/step": global_step,
                })
            pbar.set_postfix({
                'c_loss': f'{critic_loss.item():.4f}',
                'p_loss': f'{policy_loss.item():.4f}',
                'adv': f'{advantages.mean().item():.4f}',
            })
        
        # Save checkpoint
        if global_step % save_freq == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_step_{global_step}.pth"))
            torch.save(critic.state_dict(), os.path.join(save_dir, f"critic_step_{global_step}.pth"))
    
    pbar.close()
    
    # Save final models
    torch.save(model.state_dict(), os.path.join(save_dir, "model_final.pth"))
    torch.save(critic.state_dict(), os.path.join(save_dir, "critic_final.pth"))
    
    return model, critic


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AAWR Training")
    
    # Experiment
    parser.add_argument("--exp_name", type=str, default="aawr")
    parser.add_argument("--env_name", type=str, default="darkroom-easy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--continuous_action", action="store_true", help="Use continuous actions (for Ant)")
    
    # Ant-specific
    parser.add_argument("--num_goals", type=int, default=50, help="Number of goals for Ant")
    parser.add_argument("--radius", type=float, default=2.0, help="Goal sampling radius for Ant")
    parser.add_argument("--env_horizon", type=int, default=20, help="Environment horizon for Ant")
    
    # Data collection
    parser.add_argument("--dataset_size", type=int, default=10000)
    parser.add_argument("--n_envs", type=int, default=10000)
    parser.add_argument("--horizon", type=int, default=1000, help="Model/context horizon")
    parser.add_argument("--epsilon", type=float, default=0.25, help="Noisy expert epsilon")
    parser.add_argument("--n_meta_episodes", type=int, default=10, help="Meta-episodes for data collection")
    
    # Evaluation
    parser.add_argument("--eval_episodes", type=int, default=40, help="Episodes for evaluation")
    
    # Model
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Training
    parser.add_argument("--total_gradient_steps", type=int, default=100000, help="Total gradient steps for joint training")
    
    # Critic (IQL)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--critic_hidden_dim", type=int, default=256)
    parser.add_argument("--expectile", type=float, default=0.9, help="IQL expectile")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--target_tau", type=float, default=0.005, help="Target network update rate")
    
    # Policy (AWR)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--awr_temperature", type=float, default=3.0, help="AWR temperature")
    parser.add_argument("--awr_filter", type=str, default="exp_clamp", 
                        choices=["none", "indicator", "exp_clamp"])
    
    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--gradient_clip", action="store_true")
    parser.add_argument("--eval_interval", type=float, default=0.1)
    parser.add_argument("--save_interval", type=float, default=0.1)
    
    # Logging
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dpt-sweep")
    parser.add_argument("--wandb_entity", type=str, default=None)
    
    # Paths
    parser.add_argument("--save_dir", type=str, default="./aawr_results")

    args = parser.parse_args()

    # Initialize wandb
    if args.log_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"{args.exp_name}-{args.env_name}-seed{args.seed}",
        )

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Using device: {device}")
    
    # Save directory
    save_dir = os.path.join(args.save_dir, f"{args.exp_name}-{args.env_name}-seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    # Create environments
    print(f"Creating environments: {args.env_name}")
    
    if args.env_name == "ant":
        from envs.ant_env import create_ant_envs
        train_envs, test_envs, eval_envs = create_ant_envs(
            num_goals=args.num_goals,
            dataset_size=args.dataset_size,
            n_envs=args.n_envs,
            horizon=args.env_horizon,
            radius=args.radius,
            seed=args.seed,
        )
        # Ant env uses different attribute names
        state_dim = train_envs[0].state_dim  # 29
        action_dim = train_envs[0].action_dim  # 8
        goal_dim = 2
        env_horizon = train_envs[0].horizon
        # Force continuous action for Ant
        args.continuous_action = True
    else:
        train_envs, test_envs, eval_envs = create_env(args.env_name, args.dataset_size, args.n_envs)
        state_dim = train_envs[0]._envs[0].state_dim
        action_dim = train_envs[0]._envs[0].action_dim
        env_horizon = train_envs[0]._envs[0].horizon
        goal_dim = len(train_envs[0]._envs[0].goal)
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}, Env horizon: {env_horizon}")
    print(f"Goal dim: {goal_dim}")
    print(f"Model horizon: {args.horizon}")
    print(f"Continuous action: {args.continuous_action}")

    # ========================================================================
    # Phase 1: Collect offline data with noisy expert
    # ========================================================================
    print("\n" + "="*60)
    print("Phase 1: Collecting data with noisy expert")
    print("="*60)
    
    data_dir = os.path.join(save_dir, "data")
    train_data_path = os.path.join(data_dir, "train_data.pkl")
    test_data_path = os.path.join(data_dir, "test_data.pkl")
    
    if os.path.exists(train_data_path) and os.path.exists(test_data_path):
        print(f"Loading existing data from {data_dir}")
        with open(train_data_path, "rb") as f:
            train_trajs = pickle.load(f)
        with open(test_data_path, "rb") as f:
            test_trajs = pickle.load(f)
    else:
        os.makedirs(data_dir, exist_ok=True)
        
        # Collect data over n_meta_episodes per env
        data_horizon = args.n_meta_episodes * env_horizon
        
        train_trajs = collect_aawr_data(train_envs, data_horizon, args.epsilon)
        test_trajs = collect_aawr_data(test_envs, data_horizon, args.epsilon)
        
        with open(train_data_path, "wb") as f:
            pickle.dump(train_trajs, f)
        with open(test_data_path, "wb") as f:
            pickle.dump(test_trajs, f)
    
    print(f"Train trajectories: {len(train_trajs)}")
    print(f"Test trajectories: {len(test_trajs)}")

    # Create datasets
    config = {
        "horizon": args.horizon,
        "state_dim": state_dim,
        "action_dim": action_dim,
    }
    train_dataset = AAWRDataset(train_trajs, config)
    test_dataset = AAWRDataset(test_trajs, config)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=aawr_collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=aawr_collate_fn,
    )

    # ========================================================================
    # Phase 2: Train IQL critic with privileged goal info
    # ========================================================================
    print("\n" + "="*60)
    print("Phase 2: Training IQL Critic (asymmetric with goal)")
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
    
    # Model configuration
    model_args = {
        "horizon": args.horizon,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "n_layer": args.num_layers,
        "n_head": args.num_heads,
        "n_embd": 128,
        "dropout": args.dropout,
        "shuffle": True,
        "test": False,
        "continuous_action": args.continuous_action,
        "gmm_heads": 1,
    }
    
    # Add continuous action parameters
    if args.continuous_action:
        model_args.update({
            "std_min": 0.007,
            "std_max": 2.0,
            "init_std": 0.3,
        })
    
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
        env_horizon=env_horizon,
        continuous_action=args.continuous_action,
    )

    # ========================================================================
    # Phase 4: Evaluation
    # ========================================================================
    print("\n" + "="*60)
    print("Phase 4: Evaluation")
    print("="*60)
    
    eval_policy = get_rollout_policy(
        "decision_transformer",
        model=model,
        context_horizon=args.horizon,
        env_horizon=env_horizon,
        context_accumulation=False,
        sliding_window=False,
        continuous_action=args.continuous_action,
        low_noise_eval=args.continuous_action,  # Low noise for continuous actions
    )
    
    eval_save_dir = os.path.join(save_dir, "eval")
    eval_horizon = args.eval_episodes * env_horizon
    
    eval_results = evaluate_policy_on_envs(
        eval_envs=eval_envs,
        policy=eval_policy,
        eval_horizon=eval_horizon,
        env_horizon=env_horizon,
        save_dir=eval_save_dir,
        env_name=args.env_name,
        plot=True,
    )
    
    # Log to wandb
    if args.log_wandb:
        final_mean = eval_results['mean_returns'][-1]
        final_std = eval_results['std_returns'][-1]
        wandb.log({
            "eval/final_return": final_mean,
            "eval/final_return_std": final_std,
            "eval/mean_return": np.mean(eval_results['mean_returns']),
        })
        
        # Log returns plot
        fig, ax = plt.subplots(figsize=(10, 6))
        episodes = np.arange(len(eval_results['mean_returns']))
        ax.plot(episodes, eval_results['mean_returns'], label='Mean Return', linewidth=2)
        ax.fill_between(
            episodes,
            eval_results['mean_returns'] - eval_results['std_returns'],
            eval_results['mean_returns'] + eval_results['std_returns'],
            alpha=0.2,
        )
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.set_title(f'AAWR Eval Returns - {args.env_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        wandb.log({"eval/returns_plot": wandb.Image(fig)})
        plt.close(fig)
    
    print(f"\nEvaluation complete - Final return: {eval_results['mean_returns'][-1]:.2f} "
          f"± {eval_results['std_returns'][-1]:.2f}")
    print(f"\nTraining complete! Results saved to {save_dir}")
    
    if args.log_wandb:
        wandb.finish()

