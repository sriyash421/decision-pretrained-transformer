"""Evaluate trained Ant models across seeds."""

import os
import pickle
import numpy as np
import torch

# Must set multiprocessing start method before importing anything that uses it
import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn", force=True)

from models import DecisionTransformer
from envs.ant_env import create_ant_envs
from get_rollout_policy import get_rollout_policy
from eval_policy import evaluate_policy_on_envs


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for seed in [0, 1, 2]:
        print(f"\n{'='*50}")
        print(f"Evaluating seed {seed}")
        print(f"{'='*50}")
        
        # Load model_args
        base_dir = f'./results/ant_goals80/seed{seed}/ant-iterative-ant-seed{seed}'
        with open(os.path.join(base_dir, 'model_args.pkl'), 'rb') as f:
            model_args = pickle.load(f)
        # model_args['continuous_action'] = True
        # model_args['low_noise_eval'] = False
        print(model_args)
        # Load model from dagger_step_9
        model = DecisionTransformer(model_args).to(device)
        ckpt_path = os.path.join(base_dir, 'dagger_step_9', 'model_epoch_180.pth')
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        print(f"Loaded model from {ckpt_path}")
        
        # Create eval envs
        _, _, eval_envs = create_ant_envs(
            num_goals=100, dataset_size=100, n_envs=100,
            horizon=20, radius=2.0, seed=seed, eval_only=True   
        )
        
        # Create policy
        policy = get_rollout_policy(
            'decision_transformer', model=model,
            context_horizon=100, env_horizon=20,
            context_accumulation=False, sliding_window=True
        )
        
        # Evaluate with 40 episodes
        eval_horizon = 40 * 20  # 40 episodes * 20 steps
        save_dir = os.path.join(base_dir, 'dagger_step_9', 'eval_40ep')
        
        results = evaluate_policy_on_envs(
            eval_envs=eval_envs, policy=policy,
            eval_horizon=eval_horizon, env_horizon=20,
            save_dir=save_dir, env_name='ant', plot=True
        )
        
        print(f'Seed {seed}: Final return = {results["mean_returns"][-1]:.3f} ± {results["std_returns"][-1]:.3f}')
        
        # Cleanup
        for env in eval_envs:
            env.close()

    print('\nDone!')


if __name__ == '__main__':
    main()