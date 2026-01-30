"""
DPT Evaluation Functions for Ant Environment (Continuous Actions).

Provides evaluation functions that can be imported and used at the end of training.
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AntTransformerController:
    """Transformer controller for continuous action Ant environment."""
    
    def __init__(self, model, batch_size, action_stats=None):
        self.model = model
        self.state_dim = model.config['state_dim']
        self.action_dim = model.config['action_dim']
        self.horizon = model.horizon
        self.batch_size = batch_size
        self.action_stats = action_stats
        
        # Zeros tensor for padding
        zeros_dim = self.state_dim ** 2 + self.action_dim + 1
        self.zeros = torch.zeros(batch_size, zeros_dim).float().to(device)
        
        self.batch = None
    
    def set_batch(self, batch):
        """Set the context batch."""
        self.batch = batch
    
    def act(self, state):
        """Get action from model given current state and context."""
        self.batch['zeros'] = self.zeros
        
        states = torch.tensor(np.array(state)).float().to(device)
        if len(states.shape) == 1:
            states = states[None, :]
        self.batch['query_states'] = states
        
        with torch.no_grad():
            action_dist = self.model(self.batch)
            # Model returns (batch, seq_len, action_dim), take last timestep
            actions = action_dist.mean[:, -1, :].cpu().numpy()
        
        # Denormalize if needed
        if self.action_stats is not None:
            actions = actions * self.action_stats['std'] + self.action_stats['mean']
        
        # Clip actions
        actions = np.clip(actions, -1, 1)
        
        return actions


def deploy_online_vec(vec_env, controller, n_episodes, context_horizon, env_horizon):
    """
    Deploy controller on vectorized environment with context accumulation.
    
    Args:
        vec_env: AntVecEnv instance
        controller: AntTransformerController
        n_episodes: Number of evaluation episodes
        context_horizon: Context window size (H)
        env_horizon: Steps per episode
    
    Returns:
        episode_returns: (n_envs, n_episodes) array
        trajectories: dict with states, actions, rewards, dones
    """
    n_envs = vec_env.num_envs
    state_dim = vec_env.state_dim
    action_dim = vec_env.action_dim
    
    ctx_rollouts = context_horizon // env_horizon
    
    # Initialize context buffers
    context_states = torch.zeros(
        (n_envs, ctx_rollouts, env_horizon, state_dim)).float().to(device)
    context_actions = torch.zeros(
        (n_envs, ctx_rollouts, env_horizon, action_dim)).float().to(device)
    context_next_states = torch.zeros(
        (n_envs, ctx_rollouts, env_horizon, state_dim)).float().to(device)
    context_rewards = torch.zeros(
        (n_envs, ctx_rollouts, env_horizon, 1)).float().to(device)
    
    trajectories = {
        'states': [],
        'actions': [],
        'rewards': [],
        'dones': [],
    }
    episode_returns = []
    
    for ep in tqdm.tqdm(range(n_episodes), desc="Evaluating"):
        # Build context batch
        if ep < ctx_rollouts:
            batch = {
                'context_states': context_states[:, :ep, :, :].reshape(n_envs, -1, state_dim),
                'context_actions': context_actions[:, :ep, :, :].reshape(n_envs, -1, action_dim),
                'context_next_states': context_next_states[:, :ep, :, :].reshape(n_envs, -1, state_dim),
                'context_rewards': context_rewards[:, :ep, :, :].reshape(n_envs, -1, 1),
            }
        else:
            batch = {
                'context_states': context_states.reshape(n_envs, -1, state_dim),
                'context_actions': context_actions.reshape(n_envs, -1, action_dim),
                'context_next_states': context_next_states.reshape(n_envs, -1, state_dim),
                'context_rewards': context_rewards.reshape(n_envs, -1, 1),
            }
        controller.set_batch(batch)
        
        # Run episode
        state = vec_env.reset()
        ep_states = []
        ep_actions = []
        ep_rewards = []
        ep_dones = []
        
        for t in range(env_horizon):
            action = controller.act(state)
            next_state, reward, done, _ = vec_env.step(action)
            
            ep_states.append(state.copy())
            ep_actions.append(action.copy())
            ep_rewards.append(reward.copy())
            ep_dones.append(done.copy())
            
            state = next_state
        
        # Stack episode data
        ep_states = np.stack(ep_states, axis=1)
        ep_actions = np.stack(ep_actions, axis=1)
        ep_rewards = np.stack(ep_rewards, axis=1)
        ep_dones = np.stack(ep_dones, axis=1)
        
        trajectories['states'].append(ep_states)
        trajectories['actions'].append(ep_actions)
        trajectories['rewards'].append(ep_rewards)
        trajectories['dones'].append(ep_dones)
        
        episode_returns.append(np.sum(ep_rewards, axis=1))
        
        # Update context (sliding window)
        if ep < ctx_rollouts:
            context_states[:, ep, :, :] = torch.tensor(ep_states, dtype=torch.float32).to(device)
            context_actions[:, ep, :, :] = torch.tensor(ep_actions, dtype=torch.float32).to(device)
            context_next_states[:, ep, :, :] = torch.tensor(ep_states, dtype=torch.float32).to(device)
            context_rewards[:, ep, :, :] = torch.tensor(ep_rewards[:, :, None], dtype=torch.float32).to(device)
        else:
            # Shift and append
            context_states = torch.cat([
                context_states[:, 1:, :, :],
                torch.tensor(ep_states[:, None, :, :], dtype=torch.float32).to(device)
            ], dim=1)
            context_actions = torch.cat([
                context_actions[:, 1:, :, :],
                torch.tensor(ep_actions[:, None, :, :], dtype=torch.float32).to(device)
            ], dim=1)
            context_next_states = torch.cat([
                context_next_states[:, 1:, :, :],
                torch.tensor(ep_states[:, None, :, :], dtype=torch.float32).to(device)
            ], dim=1)
            context_rewards = torch.cat([
                context_rewards[:, 1:, :, :],
                torch.tensor(ep_rewards[:, None, :, None], dtype=torch.float32).to(device)
            ], dim=1)
    
    # Concatenate all episodes
    trajectories['states'] = np.concatenate(trajectories['states'], axis=1)
    trajectories['actions'] = np.concatenate(trajectories['actions'], axis=1)
    trajectories['rewards'] = np.concatenate(trajectories['rewards'], axis=1)
    trajectories['dones'] = np.concatenate(trajectories['dones'], axis=1)
    
    episode_returns = np.array(episode_returns).T
    
    return episode_returns, trajectories


def plot_returns(mean_returns, std_returns, save_path, title="DPT Evaluation Returns"):
    """Plot episode returns with confidence bands."""
    plt.figure(figsize=(10, 6))
    episodes = np.arange(len(mean_returns))
    plt.plot(episodes, mean_returns, label='Mean Return', linewidth=2)
    plt.fill_between(
        episodes, 
        mean_returns - std_returns, 
        mean_returns + std_returns, 
        alpha=0.2, 
        label='Standard Error'
    )
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_and_save(eval_envs, model, save_dir, n_episodes=40, 
                      context_horizon=200, env_horizon=20, action_stats=None,
                      title_suffix=""):
    """
    Evaluate model on Ant environments and save results.
    
    Args:
        eval_envs: List of AntVecEnv instances (expects single env)
        model: Trained Transformer model
        save_dir: Directory to save results
        n_episodes: Number of evaluation episodes
        context_horizon: Context window size (H)
        env_horizon: Steps per episode
        action_stats: Optional dict with 'mean' and 'std' for action denormalization
        title_suffix: Optional suffix for plot title
    
    Returns:
        dict with trajectories, episode_returns, mean_returns, std_returns
    """
    model.eval()
    
    assert len(eval_envs) == 1, "Expected single AntVecEnv for evaluation"
    env = eval_envs[0]
    n_envs = env.num_envs
    
    print(f"\nEvaluating on {n_envs} environments for {n_episodes} episodes...")
    
    # Create controller
    controller = AntTransformerController(model, n_envs, action_stats)
    
    # Run evaluation
    episode_returns, trajectories = deploy_online_vec(
        env, controller, n_episodes, context_horizon, env_horizon
    )
    
    mean_returns = np.mean(episode_returns, axis=0)
    std_returns = scipy.stats.sem(episode_returns, axis=0)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Save trajectories
    with open(os.path.join(save_dir, 'eval_trajs.pkl'), 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"Saved trajectories to {os.path.join(save_dir, 'eval_trajs.pkl')}")
    
    # Save returns
    np.savez(
        os.path.join(save_dir, 'eval_returns.npz'),
        episode_returns=episode_returns,
        mean_returns=mean_returns,
        std_returns=std_returns,
    )
    print(f"Saved returns to {os.path.join(save_dir, 'eval_returns.npz')}")
    
    # Plot
    plot_path = os.path.join(save_dir, 'eval_returns.png')
    plot_returns(
        mean_returns, std_returns, plot_path,
        title=f"Ant DPT Evaluation{title_suffix}"
    )
    print(f"Saved plot to {plot_path}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Evaluation Summary")
    print(f"{'='*50}")
    print(f"Num episodes: {n_episodes}")
    print(f"Num envs: {n_envs}")
    print(f"Final return: {mean_returns[-1]:.4f} ± {std_returns[-1]:.4f}")
    print(f"Mean return (all eps): {np.mean(mean_returns):.4f}")
    print(f"Max return: {np.max(mean_returns):.4f}")
    print(f"{'='*50}")
    
    return {
        'trajectories': trajectories,
        'episode_returns': episode_returns,
        'mean_returns': mean_returns,
        'std_returns': std_returns,
    }
