"""
Visualize Ant agent behavior.

Creates GIFs comparing trained policy vs expert policy.
Uses create_ant_envs to get evaluation goals and shows all goals in the background
with the current goal highlighted to demonstrate exploration behavior.
"""

import argparse
import imageio
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pathlib import Path
import sys
import os

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.ant_env import AntEnv, EXPERT_PATH, create_ant_envs
from models import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AntVisEnv(gym.Env):
    """
    Ant visualization environment that wraps AntEnv.
    
    - Uses AntEnv internally
    - Supports multi-episode rollouts with trajectory tracking
    - Provides expert actions via opt_action (state + goal concatenated)
    - Plots all evaluation goals in background with current goal highlighted
    
    Observation for student: 29-dim = 2 (xy torso) + 13 (qpos) + 14 (qvel)
    Expert obs: 31-dim = 29-dim state + 2-dim goal
    """
    
    metadata = {"render_modes": ["rgb_array"]}
    
    def __init__(self, goal, num_meta_episodes=5, max_steps=20, threshold=0.2, 
                 all_goals=None, expert_model=None):
        super().__init__()
        
        self.goal = np.array(goal, dtype=np.float32)
        self.num_meta_episodes = num_meta_episodes
        self.max_steps = max_steps
        self.threshold = threshold
        
        # Store all goals for background plotting
        self.all_goals = all_goals if all_goals is not None else [self.goal]
        
        # Create internal AntEnv
        self._env = AntEnv(goal=self.goal, horizon=self.max_steps)
        
        # Use AntEnv's spaces
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        
        self.action_dim = self._env.action_dim  # 8
        
        # Load expert for opt_action (expects 31-dim: state + goal)
        if expert_model is None:
            print(f"Loading SAC expert from {EXPERT_PATH}")
            self._expert = SAC.load(EXPERT_PATH)
        else:
            self._expert = expert_model
        
        self.meta_episode_count = 0
        self.elapsed_steps = 0
        self.current_xy_segments = []
        
    def _get_xy(self):
        """Get ant's current XY position."""
        return self._env._env.wrapped_env.get_xy()
    
    def opt_action(self, obs):
        """Get expert action (like in AntVecEnv).
        
        Args:
            obs: Student observation (29-dim)
        
        Returns:
            Expert action (8-dim)
        """
        # Build expert obs: student obs (29-dim) + goal (2-dim) = 31-dim
        expert_obs = np.concatenate([obs, self.goal]).astype(np.float32)
        action, _ = self._expert.predict(expert_obs, deterministic=True)
        return action.astype(np.float32)
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        obs, info = self._env.reset(seed=seed)
        
        self.meta_episode_count = 0
        self.elapsed_steps = 0
        self.current_xy_segments = [[self._get_xy().copy()]]
        
        return obs, {"goal": self.goal.copy()}
    
    def step(self, action):
        """Step environment."""
        obs, reward, terminated, truncated, info = self._env.step(action)
        
        xy = self._get_xy()
        self.elapsed_steps += 1
        
        # Track trajectory
        self.current_xy_segments[-1].append(xy.copy())
        
        # Check if episode ended (truncated by horizon)
        if truncated:
            self.meta_episode_count += 1
            if self.meta_episode_count < self.num_meta_episodes:
                # Reset for next meta-episode
                obs, _ = self._env.reset()
                self.elapsed_steps = 0
                self.current_xy_segments.append([self._get_xy().copy()])
                truncated = False
        
        # Final termination when all meta-episodes done
        final_terminated = self.meta_episode_count >= self.num_meta_episodes
        final_truncated = final_terminated
        
        info["goal"] = self.goal.copy()
        
        return obs, reward, final_terminated, final_truncated, info
    
    def render(self, mode='rgb_array', policy_name="Policy"):
        """Render with trajectory plot showing all goals.
        
        Uses clean MuJoCo render (no text overlays) and thin trajectory lines
        with a Blues color gradient across episodes (matching paper style).
        """
        xy = self._get_xy()
        target = self.goal
        frame = self._env.render()
        frame = np.ascontiguousarray(frame)
        
        # No text overlays on MuJoCo frame - keep it clean
        
        # 2D trajectory plot with all goals (publication quality)
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Full frame with all spines visible (paper style)
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(0.8)
            ax.spines[spine].set_color('black')
        
        ax.tick_params(colors='black', labelsize=9)
        ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3, color='#cccccc')
        ax.set_axisbelow(True)
        
        # Plot ALL goals in background (faded gray)
        all_goals = np.array(self.all_goals)
        for g in all_goals:
            # Skip current goal (will be plotted separately)
            if np.allclose(g, target, atol=1e-3):
                continue
            # Background goal circles (light gray, faded)
            bg_circle = plt.Circle((g[0], g[1]), self.threshold,
                                   facecolor='#e0e0e0', edgecolor='#b0b0b0',
                                   linewidth=0.8, alpha=0.5, zorder=1)
            ax.add_patch(bg_circle)
            # Small dot for background goals
            ax.scatter(g[0], g[1], marker='o', s=15, color='#b0b0b0',
                      edgecolors='#888888', linewidths=0.3, alpha=0.6, zorder=1)
        
        # Current goal (highlighted in green)
        goal_circle = plt.Circle((target[0], target[1]), self.threshold,
                                  facecolor='#c8f7c5', edgecolor='#2e8b57',
                                  linewidth=1.5, alpha=0.7, zorder=2)
        ax.add_patch(goal_circle)
        
        # Blues colormap for trajectory gradient (paper style)
        cmap = plt.cm.Blues
        n_segments = max(self.num_meta_episodes, len(self.current_xy_segments))
        
        # Plot trajectories with thin lines and color gradient
        for seg_idx, segment in enumerate(self.current_xy_segments):
            if len(segment) < 2:
                continue
            traj = np.array(segment)
            
            # Color from Blues gradient (darker for later episodes)
            # Shift range to avoid very light colors: 0.3 to 1.0
            color_val = 0.3 + 0.7 * (seg_idx + 1) / n_segments
            color = cmap(color_val)
            
            # Thin line trajectory (paper style)
            ax.plot(traj[:, 0], traj[:, 1], linestyle='-', linewidth=1.5, 
                    color=color, alpha=0.9, zorder=3 + seg_idx)
        
        # Current goal marker (star)
        ax.scatter(target[0], target[1], marker='*', s=300, color='#2e8b57',
                   edgecolors='#1a5c38', linewidths=1.2, zorder=10)
        
        # Current ant position marker (small red diamond)
        ax.scatter(xy[0], xy[1], marker='D', s=60, color='#e53e3e',
                   edgecolors='white', linewidths=1.0, zorder=11)
        
        # Starting position marker (hollow circle at origin)
        ax.scatter(0, 0, marker='o', s=50, facecolors='none',
                   edgecolors='black', linewidths=1.2, zorder=9)
        
        # Dynamic axis limits based on all goals
        x_min = min(all_goals[:, 0].min(), -0.5) - 0.5
        x_max = max(all_goals[:, 0].max(), 0.5) + 0.5
        y_min = min(all_goals[:, 1].min(), -0.5) - 0.5
        y_max = max(all_goals[:, 1].max(), 0.5) + 0.5
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X Position', fontsize=10)
        ax.set_ylabel('Y Position', fontsize=10)
        ax.set_title(f'{policy_name} | Goal {self._get_goal_idx()+1}/{len(self.all_goals)}', 
                    fontsize=11, fontweight='bold')
        
        # Add colorbar for episode progression
        norm = Normalize(vmin=1, vmax=n_segments)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
        cbar.set_label('Episode', fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        plot_img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        plt.close(fig)
        
        # Resize plot to match frame height
        plot_h = frame.shape[0]
        plot_w = int(plot_img.shape[1] * (plot_h / plot_img.shape[0]))
        plot_img = cv2.resize(plot_img, (plot_w, plot_h))
        
        # Concatenate
        combined = np.concatenate([frame, plot_img], axis=1)
        return combined
    
    def _get_goal_idx(self):
        """Get index of current goal in all_goals list."""
        for i, g in enumerate(self.all_goals):
            if np.allclose(g, self.goal, atol=1e-3):
                return i
        return 0
    
    def close(self):
        self._env.close()


class TransformerPolicyVis:
    """Policy wrapper for visualization using transformer."""
    
    def __init__(self, model, action_stats=None, temp=0.1):
        self.model = model
        self.temp = temp
        self.context_states = []
        self.context_actions = []
        self.context_rewards = []
        self.context_dones = []
        
        if action_stats is not None:
            self.action_mean = action_stats.get('mean', None)
            self.action_std = action_stats.get('std', None)
        else:
            self.action_mean = None
            self.action_std = None
    
    def reset(self):
        self.context_states = []
        self.context_actions = []
        self.context_rewards = []
        self.context_dones = []
    
    def denormalize_action(self, action):
        if self.action_mean is not None and self.action_std is not None:
            return action * self.action_std + self.action_mean
        return action
    
    @torch.no_grad()
    def get_action(self, obs):
        """Get action from transformer."""
        self.model.eval()
        obs = np.array(obs)
        current_state = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        
        if len(self.context_states) < 1:
            dist = self.model.get_action(current_state, None, None, None, None)
        else:
            states = torch.from_numpy(np.stack(self.context_states, axis=0)).float().unsqueeze(0).to(device)
            actions = torch.from_numpy(np.stack(self.context_actions, axis=0)).float().unsqueeze(0).to(device)
            rewards = torch.from_numpy(np.array(self.context_rewards)).float().unsqueeze(0).to(device)
            dones = torch.from_numpy(np.array(self.context_dones)).float().unsqueeze(0).to(device)
            
            # Trim to model horizon
            if states.shape[1] > self.model.horizon - 1:
                states = states[:, -(self.model.horizon - 1):]
                actions = actions[:, -(self.model.horizon - 1):]
                rewards = rewards[:, -(self.model.horizon - 1):]
                dones = dones[:, -(self.model.horizon - 1):]
            
            dist = self.model.get_action(current_state, states, actions, rewards, dones)
        
        # Sample action
        if self.temp < 1.0:
            action = dist.mean
        else:
            action = dist.sample()
        
        action = action.cpu().numpy()[0]
        return self.denormalize_action(action)
    
    def update_context(self, state, action, reward, done):
        """Update context buffers."""
        # Normalize action for context if we have stats
        if self.action_mean is not None:
            norm_action = (action - self.action_mean) / self.action_std
        else:
            norm_action = action
        
        self.context_states.append(state)
        self.context_actions.append(norm_action)
        self.context_rewards.append(reward)
        self.context_dones.append(float(done))


def rollout_expert(env, policy_name, num_meta_episodes):
    """Roll out the expert using env.opt_action()."""
    frames = []
    total_reward = 0
    
    obs, info = env.reset()
    
    done = False
    while not done:
        frame = env.render(policy_name=policy_name)
        frames.append(frame)
        # breakpoint()
        # Get expert action using opt_action (like in data collection)
        action = env.opt_action(obs)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
    
    frames.append(env.render(policy_name=policy_name))
    return frames, total_reward


def rollout_policy(env, policy, policy_name, num_meta_episodes):
    """Roll out a learned policy."""
    frames = []
    total_reward = 0
    
    obs, info = env.reset()
    
    if hasattr(policy, 'reset'):
        policy.reset()
    
    done = False
    prev_obs = obs.copy()
    
    while not done:
        frame = env.render(policy_name=policy_name)
        frames.append(frame)
        
        # Get action from learned policy
        action = policy.get_action(obs)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Update context for transformer policy
        if hasattr(policy, 'update_context'):
            episode_done = env.elapsed_steps == 0  # Just reset
            policy.update_context(prev_obs, action, reward, episode_done)
        
        prev_obs = obs.copy()
    
    frames.append(env.render(policy_name=policy_name))
    return frames, total_reward


def load_model(checkpoint_path, config_override=None):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    config = checkpoint.get('config', {})
    if config_override:
        config.update(config_override)
    
    # Create model
    model = get_model(
        model_type='decision_transformer',
        horizon=config.get('horizon', 200),
        state_dim=config.get('state_dim', 29),  # Student sees 29-dim
        action_dim=config.get('action_dim', 8),
        continuous_action=config.get('continuous_action', True),
        gmm_heads=config.get('gmm_heads', 1)
    )
    
    # Load weights
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Get action stats if available
    action_stats = checkpoint.get('action_stats', None)
    
    return model, config, action_stats


def get_unique_goals(goals_array):
    """Get unique goals from an array of goals."""
    unique = []
    for g in goals_array:
        is_duplicate = False
        for u in unique:
            if np.allclose(g, u, atol=1e-3):
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(g)
    return np.array(unique)


def main():
    parser = argparse.ArgumentParser(description="Visualize Ant agent behavior")
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                        help='Output directory for GIFs')
    parser.add_argument('--num_meta_episodes', type=int, default=5,
                        help='Number of meta-episodes per rollout')
    parser.add_argument('--max_steps', type=int, default=20,
                        help='Max steps per meta-episode')
    parser.add_argument('--num_goals', type=int, default=100,
                        help='Number of goals to generate for create_ant_envs')
    parser.add_argument('--num_vis_goals', type=int, default=5,
                        help='Number of goals to actually visualize (subset of eval goals)')
    parser.add_argument('--radius', type=float, default=2.0,
                        help='Goal radius')
    parser.add_argument('--fps', type=int, default=10,
                        help='FPS for output GIF')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--expert_only', action='store_true',
                        help='Only visualize expert')
    parser.add_argument('--model_only', action='store_true',
                        help='Only visualize learned model')
    parser.add_argument('--n_envs', type=int, default=100,
                        help='Number of parallel envs for create_ant_envs')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trained model
    model, config, action_stats = None, None, None
    if not args.expert_only:
        if args.model_path is None:
            print("Warning: No model path provided, only running expert")
            args.expert_only = True
        else:
            print(f"Loading model from {args.model_path}")
            model, config, action_stats = load_model(args.model_path)
            print(f"Model config: horizon={config.get('horizon')}, action_stats={action_stats is not None}")
    
    # Create environments using create_ant_envs to get evaluation goals
    print(f"\nCreating environments with {args.num_goals} goals...")
    train_envs, test_envs, eval_envs = create_ant_envs(
        num_goals=args.num_goals,
        dataset_size=args.n_envs,
        n_envs=args.n_envs,
        horizon=args.max_steps,
        radius=args.radius,
        seed=args.seed,
    )
    
    # Get unique evaluation goals from the eval environment
    eval_env = eval_envs[0]
    all_eval_goals = get_unique_goals(eval_env._goals)
    expert_model = eval_env._expert  # Reuse loaded expert
    
    print(f"Found {len(all_eval_goals)} unique evaluation goals")
    print(f"All eval goals:\n{all_eval_goals}")
    
    # Select subset of goals to visualize (evenly spaced)
    if len(all_eval_goals) <= args.num_vis_goals:
        vis_goals = all_eval_goals
    else:
        indices = np.linspace(0, len(all_eval_goals) - 1, args.num_vis_goals, dtype=int)
        vis_goals = all_eval_goals[indices]
    
    print(f"\nVisualizing {len(vis_goals)} goals (subset of {len(all_eval_goals)} eval goals)")
    print(f"Using expert from: {EXPERT_PATH}")
    
    for goal_idx, goal in enumerate(vis_goals):
        print(f"\n=== Goal {goal_idx + 1}/{len(vis_goals)}: ({goal[0]:.2f}, {goal[1]:.2f}) ===")
        
        # Expert rollout - uses opt_action
        if not args.model_only:
            print("Rolling out expert (using opt_action)...")
            env_expert = AntVisEnv(
                goal, args.num_meta_episodes, args.max_steps,
                all_goals=all_eval_goals, expert_model=expert_model
            )
            frames_expert, reward_expert = rollout_expert(
                env_expert, "Expert (SAC)", args.num_meta_episodes
            )
            env_expert.close()
            
            gif_path = output_dir / f"expert_goal{goal_idx}.gif"
            imageio.mimwrite(str(gif_path), frames_expert, fps=args.fps)
            print(f"Expert total reward: {reward_expert:.1f}, saved to {gif_path}")
        
        # Model rollout
        if not args.expert_only and model is not None:
            print("Rolling out learned model...")
            env_model = AntVisEnv(
                goal, args.num_meta_episodes, args.max_steps,
                all_goals=all_eval_goals, expert_model=expert_model
            )
            policy = TransformerPolicyVis(model, action_stats=action_stats, temp=0.1)
            frames_model, reward_model = rollout_policy(
                env_model, policy, "Learned DT", args.num_meta_episodes
            )
            env_model.close()
            
            gif_path = output_dir / f"model_goal{goal_idx}.gif"
            imageio.mimwrite(str(gif_path), frames_model, fps=args.fps)
            print(f"Model total reward: {reward_model:.1f}, saved to {gif_path}")
        
        # Side-by-side comparison
        if not args.expert_only and not args.model_only and model is not None:
            print("Creating side-by-side comparison...")
            env_expert = AntVisEnv(
                goal, args.num_meta_episodes, args.max_steps,
                all_goals=all_eval_goals, expert_model=expert_model
            )
            env_model = AntVisEnv(
                goal, args.num_meta_episodes, args.max_steps,
                all_goals=all_eval_goals, expert_model=expert_model
            )
            policy = TransformerPolicyVis(model, action_stats=action_stats, temp=0.1)
            
            # Reset both
            obs_e, info_e = env_expert.reset()
            obs_m, info_m = env_model.reset()
            policy.reset()
            
            frames_combined = []
            done_e = done_m = False
            prev_obs_m = obs_m.copy()
            
            max_steps = args.num_meta_episodes * args.max_steps + 10
            step = 0
            
            while step < max_steps and not (done_e and done_m):
                # Render both
                frame_e = env_expert.render(policy_name="Expert") if not done_e else frames_combined[-1][:, :frames_combined[-1].shape[1]//2]
                frame_m = env_model.render(policy_name="Learned DT") if not done_m else frames_combined[-1][:, frames_combined[-1].shape[1]//2:]
                
                # Combine
                combined = np.concatenate([frame_e, frame_m], axis=1)
                frames_combined.append(combined)
                
                # Step expert (uses opt_action)
                if not done_e:
                    action_e = env_expert.opt_action(obs_e)
                    obs_e, _, term_e, trunc_e, info_e = env_expert.step(action_e)
                    done_e = term_e or trunc_e
                
                # Step model
                if not done_m:
                    action_m = policy.get_action(obs_m)
                    ep_done_m = env_model.elapsed_steps == 0
                    obs_m, rew_m, term_m, trunc_m, info_m = env_model.step(action_m)
                    policy.update_context(prev_obs_m, action_m, rew_m, ep_done_m)
                    prev_obs_m = obs_m.copy()
                    done_m = term_m or trunc_m
                
                step += 1
            
            env_expert.close()
            env_model.close()
            
            gif_path = output_dir / f"comparison_goal{goal_idx}.gif"
            imageio.mimwrite(str(gif_path), frames_combined, fps=args.fps)
            print(f"Comparison saved to {gif_path}")
    
    # Clean up environments
    for env in train_envs + test_envs + eval_envs:
        if hasattr(env, 'close'):
            env.close()
    
    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
