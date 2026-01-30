"""
DPT Training and Evaluation for Ant (Continuous Actions).

This script combines data collection, training, and evaluation for the Ant
environment with continuous action space, following the DPT paradigm.

Usage:
    python train_ant_dpt.py --env ant --num_goals 100 --horizon 200 --num_epochs 100
"""

import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)

import argparse
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from models import Transformer
from utils import convert_to_tensor
from eval_ant_dpt import evaluate_and_save

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Data Collection for Continuous Actions
# =============================================================================

def collect_dpt_data_ant(envs, horizon, rollin_type='expert'):
    """
    Collect DPT-style data from Ant environments.
    
    Args:
        envs: List of AntVecEnv instances
        horizon: Total context horizon (across multiple episodes)
        rollin_type: 'expert' or 'uniform'
    
    Returns:
        List of trajectory dicts with context and query data
    """
    trajs = []
    
    for env in tqdm.tqdm(envs, desc="Collecting data"):
        n_envs = env.num_envs
        env_horizon = env.horizon
        n_episodes = horizon // env_horizon
        
        # Collect context data
        context_states = []
        context_actions = []
        context_next_states = []
        context_rewards = []
        
        state = env.reset()
        
        for ep in range(n_episodes):
            for t in range(env_horizon):
                if rollin_type == 'expert':
                    action = env.opt_action(state)
                elif rollin_type == 'uniform':
                    action = env.sample_action()
                else:
                    raise ValueError(f"Invalid rollin_type: {rollin_type}")
                
                next_state, reward, done, _ = env.step(action)
                
                context_states.append(state.copy())
                context_actions.append(action.copy())
                context_next_states.append(next_state.copy())
                context_rewards.append(reward.copy())
                
                state = next_state
            
            # Reset for next episode
            if ep < n_episodes - 1:
                state = env.reset()
        
        # Stack context data: (n_envs, horizon, dim)
        context_states = np.stack(context_states, axis=1)
        context_actions = np.stack(context_actions, axis=1)
        context_next_states = np.stack(context_next_states, axis=1)
        context_rewards = np.stack(context_rewards, axis=1)
        
        query_states = [env.observation_space.sample()[:29] for _ in range(n_envs)] # Sample a random state
        query_states = np.array(query_states)
        optimal_actions = env.opt_action(query_states)

        # Create trajectories for each env in the batch
        for k in range(n_envs):
            # Query: sample a state and get optimal action
            # query_state = context_states[k, -1]  # Use last state as query
            # optimal_action = env.opt_action(query_state[None])[0]
            # query_state = env.observation_space.sample() # Sample a random state
            # optimal_action = env.opt_action(query_state[None])[0]
            
            traj = {
                'query_state': query_states[k],
                'optimal_action': optimal_actions[k],
                'context_states': context_states[k],
                'context_actions': context_actions[k],
                'context_next_states': context_next_states[k],
                'context_rewards': context_rewards[k],
                'goal': env._goals[k] if hasattr(env, '_goals') else None,
            }
            trajs.append(traj)
    
    return trajs


# =============================================================================
# Dataset
# =============================================================================

class DPTDataset(torch.utils.data.Dataset):
    """Dataset for DPT training with continuous actions."""
    
    def __init__(self, trajs, config, normalize_actions=False):
        self.trajs = trajs
        self.config = config
        self.horizon = config['horizon']
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.normalize_actions = normalize_actions
        
        # Compute action statistics for normalization
        if normalize_actions:
            all_actions = np.concatenate([t['context_actions'] for t in trajs], axis=0)
            self.action_mean = np.mean(all_actions, axis=0)
            self.action_std = np.std(all_actions, axis=0) + 1e-8
        else:
            self.action_mean = np.zeros(self.action_dim)
            self.action_std = np.ones(self.action_dim)
        
        self.zeros = np.zeros(
            config['state_dim'] ** 2 + config['action_dim'] + 1
        )
        self.zeros = torch.tensor(self.zeros, dtype=torch.float32)
    
    def __len__(self):
        return len(self.trajs)
    
    def __getitem__(self, idx):
        traj = self.trajs[idx]
        
        # Normalize actions if needed
        context_actions = traj['context_actions']
        optimal_action = traj['optimal_action']
        
        if self.normalize_actions:
            context_actions = (context_actions - self.action_mean) / self.action_std
            optimal_action = (optimal_action - self.action_mean) / self.action_std
        
        # Zeros tensor for padding (model uses this to prepend to sequences)
        
        # zeros_dim = max(self.state_dim, self.action_dim)
        # zeros = torch.zeros(zeros_dim, dtype=torch.float32)
        
        return {
            'query_states': torch.tensor(traj['query_state'], dtype=torch.float32),  # Model expects 'query_states'
            'optimal_actions': torch.tensor(optimal_action, dtype=torch.float32),
            'context_states': torch.tensor(traj['context_states'], dtype=torch.float32),
            'context_actions': torch.tensor(context_actions, dtype=torch.float32),
            'context_next_states': torch.tensor(traj['context_next_states'], dtype=torch.float32),
            'context_rewards': torch.tensor(traj['context_rewards'][:, None], dtype=torch.float32),
            'zeros': self.zeros,
        }
    
    def get_action_stats(self):
        return {'mean': self.action_mean, 'std': self.action_std}


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, train_loader, optimizer, continuous_action=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        pred = model(batch)
        true_actions = batch['optimal_actions']
        
        if continuous_action:
            # Gaussian negative log likelihood
            true_actions = true_actions.unsqueeze(1).repeat(1, pred.mean.shape[1], 1)
            loss = -pred.log_prob(true_actions).sum(-1).mean()
        else:
            # Cross entropy
            true_actions = true_actions.unsqueeze(1).repeat(1, pred.shape[1], 1)
            true_actions = true_actions.reshape(-1, true_actions.shape[-1])
            pred = pred.reshape(-1, pred.shape[-1])
            loss = F.cross_entropy(pred, true_actions, reduction='mean')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def eval_epoch(model, test_loader, continuous_action=True):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            pred = model(batch)
            true_actions = batch['optimal_actions']
            
            if continuous_action:
                true_actions = true_actions.unsqueeze(1).repeat(1, pred.mean.shape[1], 1)
                loss = -pred.log_prob(true_actions).sum(-1).mean()
            else:
                true_actions = true_actions.unsqueeze(1).repeat(1, pred.shape[1], 1)
                true_actions = true_actions.reshape(-1, true_actions.shape[-1])
                pred = pred.reshape(-1, pred.shape[-1])
                loss = F.cross_entropy(pred, true_actions, reduction='mean')
            
            total_loss += loss.item()
    
    return total_loss / len(test_loader)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="DPT Training for Ant (Continuous Actions)")
    
    # Environment
    parser.add_argument('--env', type=str, default='ant')
    parser.add_argument('--num_goals', type=int, default=100)
    parser.add_argument('--n_envs', type=int, default=100)
    parser.add_argument('--env_horizon', type=int, default=20)
    parser.add_argument('--radius', type=float, default=2.0)
    
    # Data
    parser.add_argument('--horizon', type=int, default=200, help='Context horizon (H)')
    parser.add_argument('--rollin_type', type=str, default='expert', choices=['expert', 'uniform'])
    parser.add_argument('--normalize_actions', action='store_true')
    
    # Model
    parser.add_argument('--n_embd', type=int, default=64)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    
    # Evaluation
    parser.add_argument('--eval_episodes', type=int, default=40)
    parser.add_argument('--eval_every', type=int, default=20)
    
    # Logging
    parser.add_argument('--save_dir', type=str, default='./results/ant_dpt')
    parser.add_argument('--log_wandb', action='store_true')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create save directory
    exp_name = f"ant-dpt-seed{args.seed}"
    save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to {save_dir}")
    
    # Initialize wandb
    if args.log_wandb:
        import wandb
        wandb.init(project='dpt-ant', name=exp_name, config=vars(args))
    
    # Create environments
    print("Creating environments...")
    from envs.ant_env import create_ant_envs
    train_envs, test_envs, eval_envs = create_ant_envs(
        num_goals=args.num_goals,
        dataset_size=100,
        n_envs=args.n_envs,
        horizon=args.env_horizon,
        radius=args.radius,
        seed=args.seed,
    )
    
    state_dim = train_envs[0].state_dim
    action_dim = train_envs[0].action_dim
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Collect data
    print("Collecting training data...")
    train_trajs = collect_dpt_data_ant(train_envs, args.horizon, args.rollin_type)
    print(f"Collected {len(train_trajs)} training trajectories")
    
    print("Collecting test data...")
    test_trajs = collect_dpt_data_ant(test_envs, args.horizon, args.rollin_type)
    print(f"Collected {len(test_trajs)} test trajectories")
    
    # Create datasets
    config = {
        'horizon': args.horizon,
        'state_dim': state_dim,
        'action_dim': action_dim,
    }
    train_dataset = DPTDataset(train_trajs, config, normalize_actions=args.normalize_actions)
    test_dataset = DPTDataset(test_trajs, config, normalize_actions=args.normalize_actions)
    action_stats = train_dataset.get_action_stats()
    
    # Save action stats
    with open(os.path.join(save_dir, 'action_stats.pkl'), 'wb') as f:
        pickle.dump(action_stats, f)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    # Create model
    model_config = {
        'horizon': args.horizon,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': args.n_layer,
        'n_embd': args.n_embd,
        'n_head': args.n_head,
        'dropout': args.dropout,
        'continuous_action': True,
        'test': False,
    }
    model = Transformer(model_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Save model config
    with open(os.path.join(save_dir, 'model_args.pkl'), 'wb') as f:
        pickle.dump(model_config, f)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Training loop
    train_losses = []
    test_losses = []
    
    for epoch in range(args.num_epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, continuous_action=True)
        train_losses.append(train_loss)
        
        # Test
        test_loss = eval_epoch(model, test_loader, continuous_action=True)
        test_losses.append(test_loss)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{args.num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Log to wandb
        if args.log_wandb:
            wandb.log({
                'train_loss': train_loss,
                'test_loss': test_loss,
                'epoch': epoch + 1,
            })
        
        # Save checkpoint
        if (epoch + 1) % args.eval_every == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch{epoch+1}.pt'))
            
            # Evaluate using eval_ant_dpt
            print(f"Evaluating at epoch {epoch+1}...")
            eval_dir = os.path.join(save_dir, f'eval_epoch{epoch+1}')
            eval_results = evaluate_and_save(
                eval_envs, model, eval_dir,
                n_episodes=args.eval_episodes,
                context_horizon=args.horizon,
                env_horizon=args.env_horizon,
                action_stats=action_stats if args.normalize_actions else None,
                title_suffix=f" - Epoch {epoch+1}"
            )
            
            if args.log_wandb:
                wandb.log({
                    'eval_final_return': eval_results['mean_returns'][-1],
                    'eval_mean_return': np.mean(eval_results['mean_returns']),
                })
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pt'))

    # Evaluate using eval_ant_dpt
    print(f"Evaluating final model...")
    eval_dir = os.path.join(save_dir, f'eval_epoch{epoch+1}')
    eval_results = evaluate_and_save(
        eval_envs, model, eval_dir,
        n_episodes=args.eval_episodes,
        context_horizon=args.horizon,
        env_horizon=args.env_horizon,
        action_stats=action_stats if args.normalize_actions else None,
        title_suffix=f"Final Evaluation"
    )
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Close environments
    for env_list in [train_envs, test_envs, eval_envs]:
        for env in env_list:
            if hasattr(env, 'close'):
                env.close()
    
    if args.log_wandb:
        wandb.finish()
    
    print(f"\nTraining complete! Results saved to {save_dir}")


if __name__ == '__main__':
    main()

