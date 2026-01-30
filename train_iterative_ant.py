"""
Iterative Ant Training with robomimic.

This script trains a transformer policy using DAgger-style iterative data collection
with the following components:
- Training: robomimic's algo_factory and TrainUtils.run_epoch
- Environment: AntVecEnv from envs/ant_env.py
- Data Collection: get_dagger_data style from collect_data.py
- Evaluation: evaluate_policy_on_envs style from eval_policy.py
- Logging: wandb with train_context_accumulator.py directory structure
"""

import torch.multiprocessing as mp

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn", force=True)

import argparse
import gc
import json
import h5py
import numpy as np
import os
import pickle
import psutil
import shutil
import sys
import time
import traceback
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, flush_warnings

import wandb

from envs.ant_env import create_ant_envs


# =============================================================================
# Data Collection (DAgger-style for AntVecEnv)
# =============================================================================

def collect_dagger_data_single_env(env, policy, exploration_horizon, horizon):
    """
    Collect DAgger data from a single AntVecEnv.
    
    Uses policy for first exploration_horizon steps, then expert for the rest.
    Observations are augmented with done and reward from the previous step.
    
    Args:
        env: AntVecEnv instance
        policy: RolloutPolicy for learned policy (can be None for first iter)
        exploration_horizon: Steps to use learned policy
        horizon: Total steps to collect
    
    Returns:
        Dictionary with data arrays for all envs in this VecEnv
    """
    n_envs = env.num_envs
    obs = env.reset()
    
    if policy is not None:
        resets = np.ones(n_envs, dtype=bool)
        policy.start_episode(resets)
    
    states = []
    actions = []
    expert_actions_list = []
    expert_masks = []
    # Store prev_done and prev_reward as part of observation (what the policy actually sees)
    obs_dones = []
    obs_rewards = []
    
    step_counts = np.zeros(n_envs, dtype=np.int32)
    
    # Initialize done and reward to zeros BEFORE the loop starts
    prev_done = np.zeros((n_envs, 1), dtype=np.float32)
    prev_reward = np.zeros((n_envs, 1), dtype=np.float32)
    
    for t in range(horizon):
        # Decide whether to use expert or policy
        use_expert = step_counts >= exploration_horizon
        expert_mask = use_expert.astype(np.uint8)
        
        # Get policy action (if available)
        if policy is not None:
            # Augment observation with previous done and reward
            obs_dict = {
                'state': obs,
                'done': prev_done,
                'reward': prev_reward,
            }
            policy_action = policy(ob=obs_dict, batched_ob=True)
        else:
            policy_action = np.zeros((n_envs, env.action_dim), dtype=np.float32)
        
        # Get expert action (for labels)
        expert_action = env.opt_action(obs)
        
        # Use expert or policy action based on exploration horizon
        action = np.where(use_expert[:, None], expert_action, policy_action)
        
        # Store data - observation includes prev_done and prev_reward (what policy sees)
        states.append(obs.copy())
        obs_dones.append(prev_done.copy())
        obs_rewards.append(prev_reward.copy())
        actions.append(action.copy())
        expert_actions_list.append(expert_action.copy())
        expert_masks.append(expert_mask.copy())
        
        # Step environment
        next_obs, reward, done, info = env.step(action)
        step_counts += 1
        
        # Update prev_done and prev_reward for next iteration
        # For meta-learning: do NOT reset these when sub-episode ends
        # The agent needs to see done=True to know the sub-episode ended
        prev_done = done.astype(np.float32).reshape(-1, 1)
        prev_reward = reward.astype(np.float32).reshape(-1, 1)
        
        # Handle sub-episode resets (for multi-episode collection within horizon)
        # Only reset the environment state, NOT:
        # - prev_done/prev_reward (agent needs to see done=True)
        # - step_counts (tracks cumulative steps for exploration_horizon)
        # - policy context (agent accumulates context across sub-episodes)
        if np.any(done):
            next_obs = env.reset()
            # NOTE: We intentionally do NOT reset step_counts, prev_done/prev_reward, 
            # or call policy.start_episode. For meta-learning, the agent maintains
            # cumulative step count and accumulated context across sub-episodes.
        
        obs = next_obs
    
    # Stack into arrays: (n_envs, horizon, ...)
    data = {
        "states": np.stack(states, axis=1),
        "obs_dones": np.stack(obs_dones, axis=1),      # (n_envs, horizon, 1) - prev_done
        "obs_rewards": np.stack(obs_rewards, axis=1),  # (n_envs, horizon, 1) - prev_reward
        "actions": np.stack(actions, axis=1),
        "expert_actions": np.stack(expert_actions_list, axis=1),
        "expert_masks": np.stack(expert_masks, axis=1),
    }
    
    return data


def collect_dagger_data(envs, policy, exploration_horizon, horizon, dataset_size):
    """
    Collect DAgger data from a list of AntVecEnv instances.
    
    Args:
        envs: List of AntVecEnv instances
        policy: RolloutPolicy for learned policy (can be None for first iter)
        exploration_horizon: Steps to use learned policy
        horizon: Total steps to collect
    
    Returns:
        List of trajectory dicts with states, actions, rewards, dones, expert_actions, expert_mask
    """
    import tqdm
    env = envs[0]
    n = dataset_size  // env.num_envs
    trajs = []
    # for env in tqdm.tqdm(envs, desc="Collecting dagger data"):
    for i in range(n):
        data = collect_dagger_data_single_env(env, policy, exploration_horizon, horizon)
        n_envs = env.num_envs
        # Convert to list of trajectory dicts (one per env in this VecEnv)
        for k in range(n_envs):
            # Get goal
            if hasattr(env, '_goals'):
                goal = env._goals[k].copy()
            else:
                goal = None
            
            traj = {
                'states': data['states'][k],
                'obs_dones': data['obs_dones'][k],      # prev_done as part of observation
                'obs_rewards': data['obs_rewards'][k],  # prev_reward as part of observation
                'actions': data['actions'][k],
                'expert_actions': data['expert_actions'][k],
                'expert_mask': data['expert_masks'][k],
                'goal': goal,
            }
            trajs.append(traj)
    
    return trajs


def trajs_to_hdf5(trajs, hdf5_path):
    """
    Convert trajectory list to HDF5 format expected by robomimic.
    
    Observations include:
    - state: the current state
    - done: done flag from PREVIOUS transition (what policy sees)
    - reward: reward from PREVIOUS transition (what policy sees)
    
    Args:
        trajs: List of trajectory dicts
        hdf5_path: Path to save HDF5 file
    """
    with h5py.File(hdf5_path, 'w') as f:
        # Create data group
        data_grp = f.create_group('data')
        
        total_samples = 0
        for i, traj in enumerate(trajs):
            demo_grp = data_grp.create_group(f'demo_{i}')
            
            # Observations (state, done, reward)
            # done and reward are from the PREVIOUS step (what the policy sees)
            obs_grp = demo_grp.create_group('obs')
            obs_grp.create_dataset('state', data=traj['states'].astype(np.float32))
            obs_grp.create_dataset('done', data=traj['obs_dones'].astype(np.float32))
            obs_grp.create_dataset('reward', data=traj['obs_rewards'].astype(np.float32))
            
            # Actions (expert actions for training)
            demo_grp.create_dataset('actions', data=traj['expert_actions'].astype(np.float32))
            
            # Expert mask from collected data
            expert_mask = traj['expert_mask'].astype(np.uint8)
            demo_grp.create_dataset('expert_mask', data=expert_mask)
            
            # Attributes
            demo_grp.attrs['num_samples'] = len(traj['actions'])
            total_samples += len(traj['actions'])
        
        # Global attributes
        data_grp.attrs['total'] = len(trajs)
        data_grp.attrs['env_args'] = json.dumps({})
        
        # Create empty mask group (required by robomimic)
        f.create_group('mask')
    
    print(f"Saved {len(trajs)} trajectories ({total_samples} samples) to {hdf5_path}")


# =============================================================================
# Evaluation (dpt-code style)
# =============================================================================

def evaluate_policy(model, eval_envs, eval_horizon, env_horizon, save_dir, iteration):
    """
    Evaluate policy using dpt-code style.
    
    Observations are augmented with done and reward from the previous step.
    
    Args:
        model: robomimic model
        eval_envs: List of AntVecEnv for evaluation
        eval_horizon: Total evaluation steps
        env_horizon: Steps per episode
        save_dir: Directory to save results
        iteration: Current iteration
    
    Returns:
        Dictionary with evaluation results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    policy = RolloutPolicy(model)
    
    # Collect evaluation trajectories
    all_rewards = []
    env=eval_envs[0]
    # for env in eval_envs:
    for _ in range(10):
        n_envs = env.num_envs
        obs = env.reset()
        
        resets = np.ones(n_envs, dtype=bool)
        policy.start_episode(resets)
        
        env_rewards = [[] for _ in range(n_envs)]
        
        # Initialize done and reward to zeros BEFORE the loop
        prev_done = np.zeros((n_envs, 1), dtype=np.float32)
        prev_reward = np.zeros((n_envs, 1), dtype=np.float32)
        
        for t in range(eval_horizon):
            # Create observation dict augmented with previous done and reward
            obs_dict = {
                'state': obs,
                'done': prev_done,
                'reward': prev_reward,
            }
            
            action = policy(ob=obs_dict, batched_ob=True)
            next_obs, reward, done, info = env.step(action)
            
            for k in range(n_envs):
                env_rewards[k].append(reward[k])
            
            # Update prev_done and prev_reward for next iteration
            # For meta-learning: do NOT reset these when sub-episode ends
            prev_done = done.astype(np.float32).reshape(-1, 1)
            prev_reward = reward.astype(np.float32).reshape(-1, 1)
            
            # Handle sub-episode resets
            # Only reset the environment, NOT prev_done/prev_reward or policy context
            if np.any(done):
                next_obs = env.reset()
                # NOTE: We intentionally do NOT reset prev_done/prev_reward or call policy.start_episode
                # The agent should see done=True and maintain accumulated context
            
            obs = next_obs
        
        # Stack rewards
        for k in range(n_envs):
            all_rewards.append(np.array(env_rewards[k]))
    
    # Compute episode returns
    rewards = np.stack(all_rewards)  # (B, T)
    episode_returns, mean_returns, std_returns = compute_episode_returns(rewards, env_horizon)
    
    # Save results
    np.savez(
        os.path.join(save_dir, 'eval_returns.npz'),
        episode_returns=episode_returns,
        mean_returns=mean_returns,
        std_returns=std_returns,
    )
    
    # Plot returns
    plot_returns(mean_returns, std_returns, 'ant', os.path.join(save_dir, 'eval_returns.png'))
    
    # Save trajectories pickle
    with open(os.path.join(save_dir, 'eval_results.pkl'), 'wb') as f:
        pickle.dump({
            'iteration': iteration,
            'episode_returns': episode_returns,
            'mean_returns': mean_returns,
            'std_returns': std_returns,
        }, f)
    
    print(f"Iteration {iteration} eval: mean_return={np.mean(mean_returns):.3f}")
    
    return {
        'episode_returns': episode_returns,
        'mean_returns': mean_returns,
        'std_returns': std_returns,
    }


def compute_episode_returns(rewards, env_horizon):
    """
    Compute per-episode returns from trajectory rewards.
    
    Args:
        rewards: (B, T) array of rewards
        env_horizon: Number of steps per episode
    
    Returns:
        episode_returns: (B, num_episodes) array of returns
        mean_returns: (num_episodes,) mean return per episode
        std_returns: (num_episodes,) standard error per episode
    """
    episode_returns = []
    for reward_seq in rewards:
        done_indices = np.arange(0, len(reward_seq) + 1, env_horizon)
        seq_returns = [
            np.sum(reward_seq[done_indices[i]:done_indices[i + 1]]) 
            for i in range(len(done_indices) - 1)
        ]
        episode_returns.append(np.array(seq_returns))
    
    episode_returns = np.stack(episode_returns)  # (B, num_episodes)
    mean_returns = np.mean(episode_returns, axis=0)
    std_returns = np.std(episode_returns, axis=0) / np.sqrt(episode_returns.shape[0])
    
    return episode_returns, mean_returns, std_returns


def plot_returns(mean_returns, std_returns, env_name, save_path):
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
    plt.title(f'Evaluation Returns on {env_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config, device, args):
    """Main training function."""
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    
    # Note: Expert policy is loaded by AntVecEnv internally via EXPERT_PATH
    # and accessed via env.opt_action()
    
    # Create output directory structure (train_context_accumulator style)
    exp_name = f"{args.name}-ant-seed{args.seed}" if args.name else f"iterative-ant-seed{args.seed}"
    base_dir = os.path.join(args.save_dir, f"seed{args.seed}", exp_name)
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"Output directory: {base_dir}")
    
    # Initialize wandb
    if args.log_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=exp_name,
            config=vars(args),
        )
    
    # Save config
    with open(os.path.join(base_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(base_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(vars(args), f)
    
    # Initialize observation utils
    ObsUtils.initialize_obs_utils_with_config(config)
    
    # Create environments (expert is loaded from EXPERT_PATH in ant_env.py)
    print("Creating environments...")
    train_envs, test_envs, eval_envs = create_ant_envs(
        num_goals=args.num_goals,
        dataset_size=args.n_envs,
        n_envs=args.n_envs,
        horizon=args.env_horizon,
        radius=args.radius,
        seed=args.seed,
    )
    
    # Get observation and action dimensions
    sample_env = train_envs[0]
    state_dim = sample_env.state_dim
    action_dim = sample_env.action_dim
    
    obs_shapes = {
        'state': (state_dim,),
        'done': (1,),
        'reward': (1,),
    }
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Setup robomimic config for training
    with config.values_unlocked():
        config.train.num_epochs = args.num_epochs
        config.train.batch_size = args.batch_size
        if "optim_params" in config.algo:
            for k in config.algo.optim_params:
                config.algo.optim_params[k]["num_train_batches"] = config.experiment.epoch_every_n_steps
                config.algo.optim_params[k]["num_epochs"] = config.train.num_epochs * args.num_iterations
    
    # Create model
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=obs_shapes,
        ac_dim=action_dim,
        device=device
    )
    
    print("\n============= Model Summary =============")
    print(model)
    flush_warnings()
    
    # Training loop
    current_exploration_horizon = 0
    all_dataset_paths = []
    total_env_steps = 0
    global_epoch = 0
    
    for iteration in range(args.num_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{args.num_iterations}")
        print(f"Exploration horizon: {current_exploration_horizon}, Env horizon: {args.env_horizon * (iteration + 1)}")
        print(f"{'='*60}")
        
        # Create step directory
        step_dir = os.path.join(base_dir, f"dagger_step_{iteration}")
        os.makedirs(step_dir, exist_ok=True)
        
        # Collect data
        print("Collecting data...")
        policy = RolloutPolicy(model) if iteration > 0 else None
        
        collection_horizon = args.env_horizon * (iteration + 1)
        trajs = collect_dagger_data(
            envs=train_envs,
            policy=policy,
            exploration_horizon=current_exploration_horizon,
            horizon=collection_horizon,
            dataset_size=args.num_episodes,
        )
        
        # Save as HDF5
        hdf5_path = os.path.join(step_dir, f"iter_{iteration}_data.hdf5")
        trajs_to_hdf5(trajs, hdf5_path)
        all_dataset_paths.append(hdf5_path)
        
        total_env_steps += len(trajs) * collection_horizon
        
        # Load data for training
        with config.values_unlocked():
            config.train.data = [{"path": p} for p in all_dataset_paths]
        
        trainset, _ = TrainUtils.load_data_for_training(config, obs_keys=list(obs_shapes.keys()))
        train_sampler = trainset.get_dataset_sampler()
        
        train_loader = DataLoader(
            dataset=trainset,
            sampler=train_sampler,
            batch_size=config.train.batch_size,
            shuffle=(train_sampler is None),
            num_workers=min(config.train.num_data_workers, 8),
            drop_last=True,
            collate_fn=TorchUtils.collate_fn,
            prefetch_factor=4 if config.train.num_data_workers > 0 else None,
        )
        
        # Re-create optimizers for each iteration (fresh start)
        model._create_optimizers()
        
        train_num_steps = config.experiment.epoch_every_n_steps
        
        # Training epochs
        for epoch in range(config.train.num_epochs):
            global_epoch += 1
            
            step_log = TrainUtils.run_epoch(
                model=model,
                data_loader=train_loader,
                epoch=global_epoch,
                num_steps=train_num_steps,
            )
            model.on_epoch_end(global_epoch)
            
            # Log to wandb
            if args.log_wandb:
                log_dict = {
                    f"dagger-{iteration}/train_loss": step_log.get('Loss', 0),
                    f"dagger-{iteration}/epoch": epoch,
                    "global_epoch": global_epoch,
                    "iteration": iteration,
                }
                for k, v in step_log.items():
                    if not k.startswith("Time_"):
                        log_dict[f"train/{k}"] = v
                wandb.log(log_dict)
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss={step_log.get('Loss', 0):.4f}")
        
        # Save model checkpoint
        ckpt_path = os.path.join(step_dir, f"model_epoch_{config.train.num_epochs}.pth")
        TrainUtils.save_model(
            model=model,
            config=config,
            env_meta={},
            shape_meta=obs_shapes,
            ckpt_path=ckpt_path,
            obs_normalization_stats=None,
            action_normalization_stats=None,
        )
        print(f"Saved checkpoint to {ckpt_path}")
        
        # Final evaluation for this iteration
        print("Running final evaluation for iteration...")
        eval_dir = os.path.join(step_dir, "eval")
        eval_horizon = args.eval_episodes * args.env_horizon
        
        eval_results = evaluate_policy(
            model=model,
            eval_envs=eval_envs[:1],  # Use first eval env
            eval_horizon=eval_horizon,
            env_horizon=args.env_horizon,
            save_dir=eval_dir,
            iteration=iteration,
        )
        
        # Log eval results to wandb
        if args.log_wandb:
            eval_log = {
                f"eval/step_{iteration}_mean_return": np.mean(eval_results['mean_returns']),
                f"eval/step_{iteration}_final_return": eval_results['mean_returns'][-1] if len(eval_results['mean_returns']) > 0 else 0,
                "total_env_steps": total_env_steps,
            }
            # Log per-episode returns
            for ep_idx, ret in enumerate(eval_results['mean_returns']):
                eval_log[f"eval/step_{iteration}_ep_{ep_idx}_return"] = ret
            
            # Log return plot as image
            plot_path = os.path.join(eval_dir, 'eval_returns.png')
            if os.path.exists(plot_path):
                eval_log[f"eval/returns_plot_step_{iteration}"] = wandb.Image(plot_path)
            
            wandb.log(eval_log)
        
        # Update exploration horizon for next iteration
        current_exploration_horizon += args.env_horizon
        
        # Memory cleanup
        gc.collect()
        
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        print(f"Memory usage: {mem_usage} MB")
    
    # Save final model
    final_path = os.path.join(base_dir, "final_model.pth")
    TrainUtils.save_model(
        model=model,
        config=config,
        env_meta={},
        shape_meta=obs_shapes,
        ckpt_path=final_path,
        obs_normalization_stats=None,
        action_normalization_stats=None,
    )
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    # Close environments
    for env_list in [train_envs, test_envs, eval_envs]:
        for env in env_list:
            if hasattr(env, 'close'):
                env.close()
    
    if args.log_wandb:
        wandb.finish()


def main(args):
    """Main entry point."""
    # Load config
    ext_cfg = json.load(open(args.config, 'r'))
    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)
        config.train.seed = args.seed
    
    config.lock()
    
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    
    print(f"Using device: {device}")
    print(f"Config: {args.config}")
    
    try:
        train(config, device=device, args=args)
    except Exception as e:
        print(f"Training failed with error:\n{e}\n\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative Ant Training with robomimic")
    
    # Experiment
    parser.add_argument("--name", type=str, default="iterative-ant")
    parser.add_argument("--config", type=str, default="configs/transformer_bs8.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./results/ant_iterative")
    
    # Environment
    parser.add_argument("--num_goals", type=int, default=50)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--env_horizon", type=int, default=20)
    parser.add_argument("--n_envs", type=int, default=100)
    
    # Training
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    
    # Evaluation
    parser.add_argument("--eval_episodes", type=int, default=40)
    
    # Logging
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dpt-sweep")
    parser.add_argument("--wandb_entity", type=str, default="sriyash")
    
    args = parser.parse_args()
    main(args)

