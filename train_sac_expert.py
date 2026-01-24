#!/usr/bin/env python3
"""
Train SAC expert for Ant goal-reaching.

Uses AntGoalEnv with:
- 29-dim observation (27-dim base ant state + 2-dim goal)
- Goals sampled from [-2, 2] x [-2, 2] (4x4 square)
- Shaped reward: dense + sparse - ctrl - contact
- 30 step episodes, no early termination

Logs to wandb.
"""

import argparse
import os
from datetime import datetime

import numpy as np
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

# Add envs to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.ant_env import AntGoalEnv, make_ant_goal_env


class WandbCallback(BaseCallback):
    """Callback for logging metrics to wandb."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log training metrics periodically
        if self.n_calls % 1000 == 0:
            # Get info from the buffer
            if len(self.model.ep_info_buffer) > 0:
                ep_rew_mean = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                ep_len_mean = np.mean([ep['l'] for ep in self.model.ep_info_buffer])
                wandb.log({
                    'train/ep_reward_mean': ep_rew_mean,
                    'train/ep_length_mean': ep_len_mean,
                    'train/timesteps': self.num_timesteps,
                }, step=self.num_timesteps)
        return True
    
    def _on_rollout_end(self) -> None:
        # Log actor/critic losses if available
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            logs = self.model.logger.name_to_value
            if logs:
                wandb_logs = {}
                for key, value in logs.items():
                    wandb_logs[f'train/{key}'] = value
                if wandb_logs:
                    wandb.log(wandb_logs, step=self.num_timesteps)


class SaveBestCallback(BaseCallback):
    """Save model when eval reward improves."""
    
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        return True


def create_vec_env(n_envs, horizon, goal_range):
    """Create vectorized environment using SubprocVecEnv."""
    env_fns = [make_ant_goal_env(horizon=horizon, goal_range=goal_range) for _ in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    return vec_env


def main():
    parser = argparse.ArgumentParser(description='Train SAC expert for Ant goal-reaching')
    
    # Environment
    parser.add_argument('--n_envs', type=int, default=8, help='Number of parallel environments')
    parser.add_argument('--horizon', type=int, default=30, help='Steps per episode')
    parser.add_argument('--goal_range', type=float, default=2.0, help='Goal range (goals sampled from [-range, range])')
    
    # Training
    parser.add_argument('--total_timesteps', type=int, default=1_000_000, help='Total training timesteps')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--buffer_size', type=int, default=1_000_000, help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--learning_starts', type=int, default=10000, help='Steps before learning starts')
    parser.add_argument('--tau', type=float, default=0.005, help='Target network update rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    
    # Evaluation
    parser.add_argument('--eval_freq', type=int, default=10000, help='Evaluation frequency (timesteps)')
    parser.add_argument('--n_eval_episodes', type=int, default=20, help='Number of evaluation episodes')
    
    # Logging/Saving
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Generate experiment name
    if args.exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.exp_name = f'ant_sac_expert_{timestamp}'
    
    # Initialize wandb
    wandb.init(
        project='ant-goal-expert',
        name=args.exp_name,
        config=vars(args),
    )
    
    # Set seeds
    np.random.seed(args.seed)
    
    # Create environments
    print(f"Creating {args.n_envs} training environments...")
    train_env = create_vec_env(args.n_envs, args.horizon, args.goal_range)
    
    print(f"Creating {args.n_envs} evaluation environments...")
    eval_env = create_vec_env(args.n_envs, args.horizon, args.goal_range)
    
    print(f"Observation space: {train_env.observation_space}")
    print(f"Action space: {train_env.action_space}")
    
    # Create save directory
    save_path = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_path, exist_ok=True)
    
    # Create SAC model
    print("Creating SAC model...")
    model = SAC(
        'MlpPolicy',
        train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        tau=args.tau,
        gamma=args.gamma,
        verbose=1,
        seed=args.seed,
        tensorboard_log=os.path.join(save_path, 'tb_logs'),
    )
    
    # Callbacks
    wandb_callback = WandbCallback()
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=args.eval_freq // args.n_envs,  # Adjust for vec env
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    
    # Train
    print(f"Starting training for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[wandb_callback, eval_callback],
        progress_bar=True,
    )
    
    # Save final model
    final_model_path = os.path.join(save_path, 'final_model')
    model.save(final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Log final model to wandb
    wandb.save(f"{final_model_path}.zip")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    wandb.finish()
    
    print("Training complete!")


if __name__ == '__main__':
    main()

