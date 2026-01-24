"""
Ant Navigation Environment for Meta-Learning.

Wraps the d4rl AntMaze environment for goal-conditioned navigation.
The student sees: state (without xy position)
The expert sees: full state + goal position (privileged information)

Uses stable_baselines3 SubprocVecEnv for parallel MuJoCo execution.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

# Path to the pretrained SAC expert (trained on AntGoalEnv with 31-dim obs: 29-dim state + 2-dim goal)
EXPERT_PATH = "/checkpoint/siro/sriyash/dpt-code/models/ant_sac_expert_20260124_041101/best_model.zip"


class AntEnv(gym.Env):
    """
    Single Ant navigation environment (gymnasium compatible).
    
    Observation for student: 29-dim state = 2 (xy torso) + 13 (qpos) + 14 (qvel)
        - This is the d4rl AntMazeEnv obs (31 dims) without the last 2 dims (goal_direction)
    Expert obs: 29-dim state + 2-dim goal = 31 dims (privileged)
    
    Args:
        goal: 2D goal position
        horizon: Steps per episode
    """
    
    metadata = {"render_modes": ["rgb_array"]}
    
    def __init__(self, goal, horizon):
        super().__init__()
        
        # Lazy import to avoid loading d4rl on every import
        from d4rl.locomotion.ant import AntMazeEnv
        from d4rl.locomotion.wrappers import NormalizedBoxEnv
        
        EMPTY_MAZE = [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 'r', 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1]
        ]
        
        base_env = AntMazeEnv(
            maze_map=EMPTY_MAZE,
            maze_size_scaling=4.0,
            reward_type='sparse',
        )
        self._env = NormalizedBoxEnv(base_env)
        
        self.goal = np.array(goal, dtype=np.float32)
        self.horizon = horizon
        
        self.state_dim = 29
        self.action_dim = 8
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )
        
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        """Reset environment and return student observation (without goal_direction)."""
        super().reset(seed=seed)
        
        # Set target_goal BEFORE reset so _goal is set correctly in reset_model
        self._env.wrapped_env.target_goal = self.goal
        self._env.reset()
        self._env.wrapped_env.set_xy([0, 0])
        # Also set _goal directly to ensure _get_obs() computes correct goal_direction
        self._env.wrapped_env._goal = self.goal
        
        obs = self._env.wrapped_env._get_obs()[:-2].astype(np.float32)
        self.current_step = 0
        return obs, {"goal": self.goal.copy()}
    
    def step(self, action):
        """Step environment."""
        obs, _, _, info = self._env.step(action)
        obs = obs[:-2].astype(np.float32)
        
        # Compute reward based on distance to goal
        xy = self._env.wrapped_env.get_xy()
        dist = np.linalg.norm(xy - self.goal)
        reward = float(dist < 0.5)  # Sparse reward
        
        self.current_step += 1
        terminated = False  # We use truncation for horizon
        truncated = self.current_step >= self.horizon
        
        info["goal"] = self.goal.copy()
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        return self._env.render(mode='rgb_array')
    
    def close(self):
        self._env.close()


def make_ant_env(goal, horizon):
    """Factory function for creating AntEnv (required for SubprocVecEnv)."""
    def _init():
        return AntEnv(goal=goal, horizon=horizon)
    return _init


class AntGoalEnv(gym.Env):
    """
    Ant goal-reaching environment for training SAC expert.
    
    Observation: 31-dim = 29-dim base ant state + 2-dim goal (concatenated)
        - Base ant state: 2 (xy torso) + 13 (qpos) + 14 (qvel) = 29 dims
        - Goal: 2 dims (x, y target position)
    Goal: Sampled uniformly from [-2, 2] x [-2, 2] (4x4 square)
    Reward: Dense reward to goal + sparse bonus - control cost - contact cost
    Episode: 30 steps, no early termination (learns to stay at goal)
    
    Args:
        horizon: Steps per episode (default 30)
        goal_range: Range for goal sampling (default 2.0, so [-2, 2])
        sparse_threshold: Distance threshold for sparse reward (default 0.5)
        sparse_bonus: Bonus reward when within threshold (default 10.0)
    """
    
    metadata = {"render_modes": ["rgb_array"]}
    
    def __init__(self, horizon=30, goal_range=2.0, sparse_threshold=0.5, sparse_bonus=10.0):
        super().__init__()
        
        # Lazy import to avoid loading d4rl on every import
        from d4rl.locomotion.ant import AntMazeEnv
        from d4rl.locomotion.wrappers import NormalizedBoxEnv
        
        EMPTY_MAZE = [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 'r', 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1]
        ]
        
        base_env = AntMazeEnv(
            maze_map=EMPTY_MAZE,
            maze_size_scaling=4.0,
            reward_type='sparse',
        )
        self._env = NormalizedBoxEnv(base_env)
        
        self.horizon = horizon
        self.goal_range = goal_range
        self.sparse_threshold = sparse_threshold
        self.sparse_bonus = sparse_bonus

        # Base obs: 2 (xy torso) + 13 (qpos) + 14 (qvel) = 29 dims
        self.base_obs_dim = 29
        self.goal_dim = 2
        self.obs_dim = self.base_obs_dim + self.goal_dim  # 31
        self.action_dim = self._env.action_space.shape[0]  # 8
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )
        
        self.goal = None
        self.current_step = 0
        
    def _sample_goal(self):
        """Sample goal uniformly from 4x4 square."""
        return np.random.uniform(-self.goal_range, self.goal_range, size=2).astype(np.float32)
    
    def _get_xy(self):
        """Get ant's current xy position."""
        return self._env.wrapped_env.get_xy()
    
    def _compute_reward(self, action):
        """
        Compute shaped reward:
        - Dense: -0.1 * distance to goal
        - Sparse: +2 if within 0.2
        - Control cost: -0.5 * sum(action^2)
        - Contact cost: -0.5 * 1e-3 * sum(clip(contact_forces)^2)
        """
        xy = self._get_xy()
        dist = np.linalg.norm(xy - self.goal)
        
        # Dense reward (negative distance, scaled)
        dense_reward = -0.1 * dist
        
        # Control cost
        ctrl_cost = 0.5 * np.square(action).sum()
        
        # Contact cost
        cfrc_ext = self._env.wrapped_env.sim.data.cfrc_ext
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(cfrc_ext, -1, 1)))
        
        # Sparse bonus (+2 if within 0.2)
        sparse_reward = 5.0 if dist < 0.2 else 0.0
        
        reward = dense_reward - ctrl_cost - contact_cost + sparse_reward
        
        return reward, {
            'dense_reward': dense_reward,
            'ctrl_cost': -ctrl_cost,
            'contact_cost': -contact_cost,
            'sparse_reward': sparse_reward,
            'dist_to_goal': dist,
        }
    
    def reset(self, seed=None, options=None):
        """Reset environment with new random goal."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        # Sample new goal
        self.goal = self._sample_goal()
        
        # Reset the base environment
        self._env.reset()
        self._env.wrapped_env.set_xy([0, 0])
        
        self.current_step = 0
        
        obs = self._env.wrapped_env._get_obs()[:-2].astype(np.float32)
        obs = np.concatenate([obs, self.goal]).astype(np.float32)
        info = {'goal': self.goal.copy(), 'dist_to_goal': np.linalg.norm(self._get_xy() - self.goal)}
        
        return obs, info
    
    def step(self, action):
        """Step environment."""
        obs, _, _, info = self._env.step(action)
        obs = obs[:-2].astype(np.float32)
        obs = np.concatenate([obs, self.goal]).astype(np.float32)

        reward, reward_info = self._compute_reward(action)
        
        self.current_step += 1
        
        # No early termination - always run full horizon
        terminated = False
        truncated = self.current_step >= self.horizon
        
        info = {
            'goal': self.goal.copy(),
            **reward_info,
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        return self._env.render(mode='rgb_array')
    
    def close(self):
        self._env.close()


def make_ant_goal_env(horizon=30, goal_range=2.0):
    """Factory function for creating AntGoalEnv (required for SubprocVecEnv)."""
    def _init():
        return AntGoalEnv(horizon=horizon, goal_range=goal_range)
    return _init


class AntVecEnv:
    """
    Vectorized Ant environment using stable_baselines3 SubprocVecEnv.
    
    Args:
        goals: Array of goals, one per env (n_envs, 2)
        horizon: Steps per episode
        expert_model: Loaded SAC model (shared across all envs)
    """
    
    def __init__(self, goals, horizon, expert_model=None):
        self._goals = np.array(goals, dtype=np.float32)
        self._num_envs = len(goals)
        self.horizon = horizon
        
        # Create SubprocVecEnv with one env per goal
        print(f"Creating SubprocVecEnv with {self._num_envs} environments...")
        env_fns = [make_ant_env(goal, horizon) for goal in goals]
        self._vec_env = SubprocVecEnv(env_fns)
        print(f"SubprocVecEnv created.")
        
        self.observation_space = self._vec_env.observation_space
        self.action_space = self._vec_env.action_space

        self.state_dim = 29
        self.action_dim = 8
        
        # Load or use provided expert model
        if expert_model is None:
            print(f"Loading SAC expert from {EXPERT_PATH}")
            self._expert = SAC.load(EXPERT_PATH)
        else:
            self._expert = expert_model
            
        # Placeholder for compatibility
        self._envs = [None] * self._num_envs
            
    @property
    def num_envs(self):
        return self._num_envs
    
    @property
    def envs(self):
        return self._envs
    
    def sample_action(self):
        """Sample random actions."""
        return np.random.uniform(-1, 1, (self._num_envs, self.action_dim)).astype(np.float32)
    
    def reset(self):
        """Reset all environments."""
        obs = self._vec_env.reset()    
        return obs
    
    def step(self, actions):
        """Step all environments."""
        obs, rewards, dones, infos = self._vec_env.step(actions)
        
        return obs, rewards, dones, infos
    
    def opt_action(self, obs):
        """
        Get expert actions using privileged information (state + goal).
        
        The expert expects 31-dim flat observation:
        - 29-dim state (same as student obs): 2 (xy) + 13 (qpos) + 14 (qvel)
        - 2-dim goal position (privileged info)
        
        Args:
            obs: Student observations (n_envs, 29)
        
        Returns:
            Expert actions (n_envs, action_dim)
        """
        expert_obs = np.concatenate([obs, self._goals], axis=1).astype(np.float32)
        actions, _ = self._expert.predict(expert_obs, deterministic=True)
        return actions.astype(np.float32)
    
    def close(self):
        self._vec_env.close()


def create_ant_envs(num_goals=50, dataset_size=1000, n_envs=100, horizon=20, radius=2.0, seed=42):
    """
    Create Ant environments for training and testing.
    
    Simple approach:
    1. Generate num_goals goals on semicircle
    2. Split 80/20 into train/test goals
    3. Duplicate goals to fill n_envs (e.g., 40 train goals -> duplicate to get 100 envs)
    4. Create single SubprocVecEnv for each split
    
    Args:
        num_goals: Number of distinct goals to generate
        dataset_size: Not used (kept for API compatibility)
        n_envs: Number of parallel environments per batch
        horizon: Steps per episode
        radius: Radius for goal sampling on semicircle
        seed: Random seed for goal generation
    
    Returns:
        train_envs, test_envs, eval_envs: Lists containing single AntVecEnv each
    """
    np.random.seed(seed)
    
    # Generate goals on semicircle
    angles = np.linspace(0, np.pi, num_goals)
    goals = np.array([[radius * np.cos(a), radius * np.sin(a)] for a in angles])
    
    # Shuffle and split 80/20
    np.random.shuffle(goals)
    split_idx = int(0.8 * len(goals))
    train_goals = goals[:split_idx]  # 80% for train
    test_goals = goals[split_idx:]    # 20% for test
    
    print(f"Generated {num_goals} goals: {len(train_goals)} train, {len(test_goals)} test")
    
    # Duplicate goals to fill n_envs
    # For train: repeat train_goals to get n_envs envs
    factor = n_envs // len(goals)
    factor = max(1, factor)
    # train_repeats = max(1, (n_envs + len(train_goals) - 1) // len(train_goals))
    train_goals_expanded = np.tile(train_goals, (factor, 1))[:n_envs]
    
    # For test/eval: repeat test_goals
    # test_repeats = max(1, (n_envs + len(test_goals) - 1) // len(test_goals))
    test_goals_expanded = np.tile(test_goals, (factor, 1))[:n_envs]
    
    print(f"Expanded to {len(train_goals_expanded)} train envs, {len(test_goals_expanded)} test envs")
    
    # Load expert model once (shared across all envs)
    print(f"Loading SAC expert from {EXPERT_PATH}")
    expert_model = SAC.load(EXPERT_PATH)
    
    # Create vectorized environments
    print("Creating train environments...")
    train_env = AntVecEnv(train_goals_expanded, horizon, expert_model)
    
    print("Creating test environments...")
    test_env = AntVecEnv(test_goals_expanded, horizon, expert_model)
    
    print("Creating eval environments...")
    eval_env = AntVecEnv(test_goals_expanded, horizon, expert_model)
    
    # Return as lists (for compatibility with existing code that iterates over env batches)
    num_envs = dataset_size // horizon
    return [train_env] * num_envs, [test_env] * num_envs, [eval_env] * num_envs
