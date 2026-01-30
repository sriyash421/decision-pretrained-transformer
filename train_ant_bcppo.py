"""
BC+PPO and Advisor training for Ant Navigation.

Two modes:
- BCPPO: Simple BC + PPO with decaying BC coefficient
- Advisor: Learned distance predictor for adaptive weighting

Uses sb3_contrib RecurrentPPO with custom policy and training loop.
"""

import argparse
import os
import pickle
import random
from copy import deepcopy
from collections import defaultdict
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt

import gymnasium
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.torch_layers import FlattenExtractor
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer

from envs.ant_env import AntVecEnv, create_ant_envs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# Custom Rollout Buffer with Expert Actions
# ==============================================================================

class RecurrentRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    lstm_states: Any
    episode_starts: torch.Tensor
    mask: torch.Tensor
    expert_actions: torch.Tensor


class MetaRecurrentRolloutBuffer(RecurrentRolloutBuffer):
    """Rollout buffer that also stores expert actions."""
    
    def reset(self):
        super().reset()
        self.expert_actions = np.zeros_like(self.actions)

    def add(self, *args, expert_action=None, lstm_states=None, **kwargs):
        self.expert_actions[self.pos] = expert_action
        super().add(*args, lstm_states=lstm_states, **kwargs)

    def _get_samples(self, batch_inds, env_change, env=None):
        base = super()._get_samples(batch_inds, env_change, env=env)
        padded_batch_size = base.observations.shape[0]

        expert_actions = self.expert_actions.reshape(-1, *self.actions.shape[1:])
        expert_action = self.pad(expert_actions[batch_inds]).reshape(
            (padded_batch_size, *self.actions.shape[1:])
        )
        return RecurrentRolloutBufferSamples(
            observations=base.observations,
            actions=base.actions,
            old_values=base.old_values,
            old_log_prob=base.old_log_prob,
            advantages=base.advantages,
            returns=base.returns,
            lstm_states=base.lstm_states,
            episode_starts=base.episode_starts,
            mask=base.mask,
            expert_actions=expert_action,
        )


# ==============================================================================
# Advisor Policy (with auxiliary imitation network + distance predictor)
# ==============================================================================

class AdvisorPolicy(RecurrentActorCriticPolicy):
    """
    Policy for Advisor method with:
    - Main policy network (for RL)
    - Auxiliary policy network (for pure imitation)
    - Distance predictor (for advisor weighting)
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.Tanh,
        ortho_init=True,
        use_sde=False,
        log_std_init=0.0,
        full_std=True,
        use_expln=False,
        squash_output=False,
        features_extractor_class=FlattenExtractor,
        features_extractor_kwargs=None,
        share_features_extractor=True,
        normalize_images=True,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=None,
        lstm_hidden_size=256,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,
        lstm_kwargs=None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=n_lstm_layers,
            shared_lstm=shared_lstm,
            enable_critic_lstm=enable_critic_lstm,
            lstm_kwargs=lstm_kwargs,
        )
        
        # Auxiliary policy for imitation learning (copy of main policy)
        self.aux_policy_net = deepcopy(self.mlp_extractor.policy_net)
        self.aux_action_net = deepcopy(self.action_net)
        
        # For continuous actions, also copy log_std
        if hasattr(self, 'log_std'):
            self.aux_log_std = deepcopy(self.log_std)
        
        # Distance predictor: predicts distance from observations
        obs_dim = int(np.prod(self.observation_space.shape))
        self.distance_predictor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        
        # Reinitialize optimizer to include new parameters
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def forward_aux_expert(self, obs, lstm_states, episode_starts):
        """Forward pass through auxiliary imitation network."""
        features = self.extract_features(obs)
        aux_pi_features = features if self.share_features_extractor else features[0]
        latent_aux_pi, _ = self._process_sequence(
            aux_pi_features, lstm_states.pi, episode_starts, self.lstm_actor
        )
        
        latent_aux = self.aux_policy_net(latent_aux_pi)
        mean_actions = self.aux_action_net(latent_aux)
        
        # For continuous actions, return distribution
        if isinstance(self.action_space, spaces.Box):
            if hasattr(self, 'aux_log_std'):
                log_std = self.aux_log_std
            else:
                log_std = self.log_std
            return self.action_dist.proba_distribution(mean_actions, log_std)
        else:
            return self.action_dist.proba_distribution(action_logits=mean_actions)


# ==============================================================================
# Advisor PPO (BCPPO + Advisor modes)
# ==============================================================================

class AdvisorPPO(RecurrentPPO):
    """
    RecurrentPPO with two training modes:
    - use_bcppo=True: Simple BC + PPO with decaying BC coefficient
    - use_bcppo=False: Advisor with learned distance predictor
    """
    
    policy_aliases = {
        "AdvisorPolicy": AdvisorPolicy,
    }

    def __init__(
        self,
        *args,
        use_bcppo: bool = False,
        bc_decay: float = 0.995,
        advisor_alpha: float = 4.0,
        advisor_beta: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_bcppo = use_bcppo
        self.bc_decay = bc_decay
        self.bc_loss_coeff = 1.0  # Initial BC coefficient
        
        # Advisor hyperparameters
        self.advisor_alpha = advisor_alpha  # Weight scaling: w = exp(-alpha * distance)
        self.advisor_beta = advisor_beta    # Distance power

    def _setup_model(self) -> None:
        super()._setup_model()

        buffer_cls = MetaRecurrentRolloutBuffer
        lstm = self.policy.lstm_actor
        hidden_state_buffer_shape = (
            self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size
        )

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """Collect rollouts with expert actions stored."""
        assert self._last_obs is not None, "No previous observation"
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        lstm_states = deepcopy(self._last_lstm_states)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = torch.tensor(
                    self._last_episode_starts, dtype=torch.float32, device=self.device
                )
                actions, values, log_probs, lstm_states = self.policy(
                    obs_tensor, lstm_states, episode_starts
                )

            actions = actions.cpu().numpy()
            clipped_actions = actions
            
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            
            # Get expert actions from infos
            expert_action = np.array([info["expert_action"] for info in infos])

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # Handle timeout bootstrapping
            for idx, done_ in enumerate(dones):
                if (
                    done_
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with torch.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx : idx + 1, :].contiguous(),
                            lstm_states.vf[1][:, idx : idx + 1, :].contiguous(),
                        )
                        episode_starts = torch.tensor(
                            [False], dtype=torch.float32, device=self.device
                        )
                        terminal_value = self.policy.predict_values(
                            terminal_obs, terminal_lstm_state, episode_starts
                        )[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
                expert_action=expert_action,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

        with torch.no_grad():
            episode_starts = torch.tensor(
                dones, dtype=torch.float32, device=self.device
            )
            values = self.policy.predict_values(
                obs_as_tensor(new_obs, self.device), lstm_states.vf, episode_starts
            )

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """Update policy with BCPPO or Advisor loss."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        il_losses, rl_losses, advisor_ws = [], [], []
        bc_losses, prediction_losses = [], []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                mask = rollout_data.mask > 1e-8

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )

                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (
                        advantages[mask].std() + 1e-8
                    )

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2)

                # Expert actions (continuous)
                expert_actions = rollout_data.expert_actions
                if isinstance(self.action_space, spaces.Discrete):
                    expert_actions = expert_actions[:, 0].long()
                
                _, expert_log_prob, _ = self.policy.evaluate_actions(
                    rollout_data.observations,
                    expert_actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )

                if self.use_bcppo:
                    # ========== BCPPO Mode ==========
                    # BC loss: negative log-likelihood of expert actions
                    if len(expert_log_prob.shape) > 1:
                        expert_log_prob_sum = expert_log_prob.sum(-1)
                    else:
                        expert_log_prob_sum = expert_log_prob
                    bc_loss = -torch.mean(expert_log_prob_sum[mask])
                    rl_loss = policy_loss[mask].mean()
                    
                    loss = (1 - self.bc_loss_coeff) * rl_loss + self.bc_loss_coeff * bc_loss
                    
                    if entropy is None:
                        entropy_loss = -torch.mean(-log_prob[mask])
                    else:
                        entropy_loss = -torch.mean(entropy[mask])
                    
                    loss += self.ent_coef * entropy_loss
                    
                    if self.clip_range_vf is None:
                        values_pred = values
                    else:
                        values_pred = rollout_data.old_values + torch.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    value_loss = torch.mean(((rollout_data.returns - values_pred) ** 2)[mask])
                    loss += self.vf_coef * value_loss
                    
                    il_loss = bc_loss
                    w = torch.tensor(self.bc_loss_coeff)
                    bc_losses.append(bc_loss.item())
                    
                else:
                    # ========== Advisor Mode ==========
                    aux_distribution = self.policy.forward_aux_expert(
                        rollout_data.observations,
                        rollout_data.lstm_states,
                        rollout_data.episode_starts,
                    )
                    aux_log_prob = aux_distribution.log_prob(expert_actions)
                    if len(aux_log_prob.shape) > 1:
                        aux_log_prob = aux_log_prob.sum(-1)
                    aux_imitation_loss = -torch.mean(aux_log_prob[mask])
                    
                    distance_target = ((-aux_log_prob) ** self.advisor_beta).detach()
                    predicted_distance = self.policy.distance_predictor(
                        rollout_data.observations
                    ).squeeze(-1)
                    prediction_loss = F.mse_loss(
                        predicted_distance[mask], distance_target[mask]
                    )
                    
                    w = torch.exp(-self.advisor_alpha * predicted_distance).clamp(0.0, 1.0).detach()
                    
                    expert_log_prob_sum = expert_log_prob
                    if len(expert_log_prob.shape) > 1:
                        expert_log_prob_sum = expert_log_prob.sum(-1)
                    
                    il_loss = (-(w * expert_log_prob_sum)[mask]).mean()
                    rl_loss = (((1 - w) * policy_loss)[mask]).mean()
                    
                    if entropy is None:
                        entropy_loss = -torch.mean(-log_prob[mask])
                    else:
                        entropy_loss = -torch.mean(entropy[mask])
                    
                    if self.clip_range_vf is None:
                        values_pred = values
                    else:
                        values_pred = rollout_data.old_values + torch.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    value_loss = torch.mean(((rollout_data.returns - values_pred) ** 2)[mask])
                    
                    loss = rl_loss + il_loss + aux_imitation_loss + prediction_loss
                    loss += self.ent_coef * entropy_loss + self.vf_coef * value_loss
                    
                    prediction_losses.append(prediction_loss.item())

                pg_losses.append(policy_loss.mean().item())
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1) > clip_range).float()[mask]
                ).item()
                clip_fractions.append(clip_fraction)
                il_losses.append(il_loss.item())
                rl_losses.append(rl_loss.item())
                advisor_ws.append(w.mean().item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean(
                        ((torch.exp(log_ratio) - 1) - log_ratio)[mask]
                    ).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping due to KL: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        # Decay BC coefficient for BCPPO
        if self.use_bcppo:
            self.bc_loss_coeff *= self.bc_decay

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        # Logging
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/il_loss", np.mean(il_losses))
        self.logger.record("train/rl_loss", np.mean(rl_losses))
        self.logger.record("train/advisor_w", np.mean(advisor_ws))
        self.logger.record("train/explained_variance", explained_var)

        if self.use_bcppo:
            self.logger.record("train/bc_loss", np.mean(bc_losses))
            self.logger.record("train/bc_loss_coeff", self.bc_loss_coeff)
        else:
            self.logger.record("train/prediction_loss", np.mean(prediction_losses))

        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)


# ==============================================================================
# Meta Vectorized Environment
# ==============================================================================

class AntMetaVecEnv(VecEnv):
    """
    Vectorized meta-environment for Ant BC+PPO/Advisor.
    
    - Wraps AntVecEnv
    - Augments observations with (prev_reward, prev_done)
    - Stores expert actions in info dict
    """
    
    def __init__(self, ant_vec_env, num_meta_episodes):
        self._ant_env = ant_vec_env
        self.num_meta_episodes = num_meta_episodes
        self.env_horizon = ant_vec_env.horizon
        self.meta_horizon = num_meta_episodes * self.env_horizon
        
        # Augmented observation: [obs, prev_reward, prev_done]
        base_obs_shape = ant_vec_env.observation_space.shape
        aug_obs_dim = base_obs_shape[0] + 2
        observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(aug_obs_dim,), dtype=np.float32
        )
        
        action_space = ant_vec_env.action_space
        
        super().__init__(ant_vec_env.num_envs, observation_space, action_space)
        
        self.current_episodes = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.prev_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.prev_dones = np.zeros(self.num_envs, dtype=np.float32)
        self._last_obs = None

    def _get_augmented_obs(self, obs):
        return np.concatenate([
            obs,
            self.prev_rewards[:, None],
            self.prev_dones[:, None]
        ], axis=1).astype(np.float32)

    def reset(self):
        self.current_episodes = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.prev_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.prev_dones = np.zeros(self.num_envs, dtype=np.float32)
        
        obs = self._ant_env.reset()
        self._last_obs = obs
        return self._get_augmented_obs(obs)

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        actions = self._actions
        
        # Get expert actions BEFORE step
        expert_actions = self._ant_env.opt_action(self._last_obs)
        
        obs, rewards, dones, infos = self._ant_env.step(actions)
        self.episode_steps += 1
        
        internal_dones = np.zeros(self.num_envs, dtype=np.float32)
        meta_dones = np.zeros(self.num_envs, dtype=bool)
        
        for i in range(self.num_envs):
            infos[i]["expert_action"] = expert_actions[i]
            
            if dones[i] or self.episode_steps[i] >= self.env_horizon:
                self.current_episodes[i] += 1
                self.episode_steps[i] = 0
                internal_dones[i] = 1.0
            
            if self.current_episodes[i] >= self.num_meta_episodes:
                meta_dones[i] = True
                infos[i]['terminal_observation'] = np.concatenate([
                    obs[i], [rewards[i], internal_dones[i]]
                ]).astype(np.float32)
        
        self.prev_rewards = rewards.copy()
        self.prev_dones = internal_dones.copy()
        self._last_obs = obs
        
        return self._get_augmented_obs(obs), rewards, meta_dones, infos

    def close(self):
        self._ant_env.close()

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return []

    def get_attr(self, attr_name, indices=None):
        return [None] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        pass


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate(model, env, num_episodes, deterministic=True):
    """Evaluate policy on environment."""
    obs = env.reset()
    episode_returns = []
    current_returns = np.zeros(env.num_envs)
    episodes_done = np.zeros(env.num_envs, dtype=int)
    
    # Get LSTM states
    lstm_states = None
    episode_starts = np.ones(env.num_envs, dtype=bool)
    
    while min(episodes_done) < num_episodes:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=deterministic
        )
        obs, rewards, dones, infos = env.step(action)
        current_returns += rewards
        episode_starts = dones
        
        for i in range(env.num_envs):
            if dones[i]:
                episode_returns.append(current_returns[i])
                current_returns[i] = 0
                episodes_done[i] += 1
    
    return episode_returns


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BC+PPO / Advisor for Ant")
    
    # Experiment
    parser.add_argument("--exp_name", type=str, default="ant-bcppo")
    parser.add_argument("--seed", type=int, default=0)
    
    # Method selection
    parser.add_argument("--use_bcppo", action="store_true", 
                        help="Use BCPPO mode (default: Advisor mode)")
    
    # Environment
    parser.add_argument("--num_goals", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--num_meta_episodes", type=int, default=5)
    parser.add_argument("--eval_episodes", type=int, default=20)
    
    # Training
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    
    # BCPPO specific
    parser.add_argument("--bc_decay", type=float, default=0.995)
    
    # Advisor specific
    parser.add_argument("--advisor_alpha", type=float, default=4.0)
    parser.add_argument("--advisor_beta", type=float, default=0.1)
    
    # LSTM
    parser.add_argument("--lstm_hidden_size", type=int, default=256)
    
    # Evaluation
    parser.add_argument("--eval_freq", type=int, default=50000)
    
    # Logging
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dpt-sweep")
    parser.add_argument("--wandb_entity", type=str, default="sriyash")
    parser.add_argument("--save_dir", type=str, default="./results/ant_bcppo")
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Determine method name
    method_name = "bcppo" if args.use_bcppo else "advisor"
    
    # Save directory
    save_dir = os.path.join(args.save_dir, f"{args.exp_name}-{method_name}-seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize wandb
    if args.log_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"{args.exp_name}-{method_name}-seed{args.seed}",
        )
    
    print(f"Training {method_name.upper()} on Ant...")
    print(f"  num_goals={args.num_goals}, horizon={args.horizon}, n_envs={args.n_envs}")
    
    # Create environments
    print("Creating Ant environments...")
    train_envs, test_envs, eval_envs = create_ant_envs(
        num_goals=args.num_goals,
        dataset_size=1000,
        n_envs=args.n_envs,
        horizon=args.horizon,
        seed=args.seed,
    )
    
    train_ant_env = train_envs[0]
    eval_ant_env = eval_envs[0] if eval_envs else train_envs[0]
    
    # Wrap in meta-env
    train_env = AntMetaVecEnv(train_ant_env, args.num_meta_episodes)
    eval_env = AntMetaVecEnv(eval_ant_env, args.eval_episodes)
    
    print(f"  Observation space: {train_env.observation_space}")
    print(f"  Action space: {train_env.action_space}")
    print(f"  Meta horizon: {train_env.meta_horizon}")
    
    # Create model
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=nn.Tanh,
        lstm_hidden_size=args.lstm_hidden_size,
        n_lstm_layers=1,
        share_features_extractor=True,
    )
    
    model = AdvisorPPO(
        AdvisorPolicy,
        train_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=args.seed,
        policy_kwargs=policy_kwargs,
        use_bcppo=args.use_bcppo,
        bc_decay=args.bc_decay,
        advisor_alpha=args.advisor_alpha,
        advisor_beta=args.advisor_beta,
    )
    
    print(f"Model created. Training for {args.total_timesteps} timesteps...")
    
    # Training loop with evaluation
    timesteps_done = 0
    eval_results = []
    
    while timesteps_done < args.total_timesteps:
        steps_to_train = min(args.eval_freq, args.total_timesteps - timesteps_done)
        
        model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)
        timesteps_done += steps_to_train
        
        # Evaluate
        print(f"\nEvaluating at {timesteps_done} timesteps...")
        episode_returns = evaluate(model, eval_env, args.eval_episodes)
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        
        eval_results.append({
            "timesteps": timesteps_done,
            "mean_return": mean_return,
            "std_return": std_return,
        })
        
        print(f"  Mean return: {mean_return:.3f} ± {std_return:.3f}")
        
        if args.log_wandb:
            wandb.log({
                "eval/mean_return": mean_return,
                "eval/std_return": std_return,
                "timesteps": timesteps_done,
                "train/bc_coeff": model.bc_loss_coeff if args.use_bcppo else 0,
            })
    
    # Save final model and results
    model.save(os.path.join(save_dir, "final_model"))
    
    with open(os.path.join(save_dir, "eval_results.pkl"), "wb") as f:
        pickle.dump(eval_results, f)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    timesteps = [r["timesteps"] for r in eval_results]
    means = [r["mean_return"] for r in eval_results]
    stds = [r["std_return"] for r in eval_results]
    ax.plot(timesteps, means, label='Mean Return', linewidth=2)
    ax.fill_between(timesteps, np.array(means) - np.array(stds), 
                    np.array(means) + np.array(stds), alpha=0.2)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Return')
    ax.set_title(f'{method_name.upper()} Ant - {args.exp_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "eval_returns.png"), dpi=150, bbox_inches='tight')
    
    if args.log_wandb:
        wandb.log({"eval/returns_plot": wandb.Image(fig)})
    
    plt.close(fig)
    
    print(f"\nTraining complete! Results saved to {save_dir}")
    
    train_env.close()
    eval_env.close()
    
    if args.log_wandb:
        wandb.finish()
