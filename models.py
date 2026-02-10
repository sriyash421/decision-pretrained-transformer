import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
transformers.set_seed(0)
from transformers import GPT2Config, GPT2Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.distributions import TransformedDistribution, TanhTransform, Normal, Independent

# Constants (following robomimic exactly)
MEAN_CLAMP = 9.0     # mean_limits=(-9.0, 9.0)

def get_model(model_type, horizon, state_dim, action_dim, continuous_action, gmm_heads=1):
    n_embd = 256
    n_head = 4
    n_layer = 4
    dropout = 0.1
    shuffle = True
    config = {
        'horizon': horizon,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'shuffle': shuffle,
        'dropout': dropout,
        'test': False,
        'store_gpu': True,
        'continuous_action': continuous_action,
        'gmm_heads': gmm_heads,
    }
    if model_type == "decision_transformer":
        model = DecisionTransformer(config).to(device)
    elif model_type == "transformer":
        model = Transformer(config).to(device)
    elif model_type == "mlp":
        model = MLP(config).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_type}")
    return model

class Transformer(nn.Module):
    """Transformer class."""

    def __init__(self, config):
        super(Transformer, self).__init__()

        self.config = config
        self.test = config['test']
        self.horizon = self.config['horizon']
        self.n_embd = self.config['n_embd']
        self.n_layer = self.config['n_layer']
        self.n_head = self.config['n_head']
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']
        self.dropout = self.config['dropout']

        config = GPT2Config(
            n_positions=4 * (1 + self.horizon),
            n_ctx=4 * (1 + self.horizon),
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=4,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(config)

        self.embed_transition = nn.Linear(
            2 * self.state_dim + self.action_dim + 1, self.n_embd)
        self.continuous_action = self.config['continuous_action']
        if self.continuous_action:
            self.pred_action_means = nn.Linear(self.n_embd, self.action_dim)
            self.pred_action_log_stds = nn.Linear(self.n_embd, self.action_dim)
            # Robomimic defaults for Gaussian policy
            self.init_std = self.config.get('init_std', 0.3)
            self.std_limits = (0.007, 7.5)  # (min, max)
        else:
            self.pred_actions = nn.Linear(self.n_embd, self.action_dim)

    def forward(self, x):
        query_states = x['query_states'][:, None, :]
        zeros = x['zeros'][:, None, :]

        state_seq = torch.cat([query_states, x['context_states']], dim=1)
        action_seq = torch.cat(
            [zeros[:, :, :self.action_dim], x['context_actions']], dim=1)
        next_state_seq = torch.cat(
            [zeros[:, :, :self.state_dim], x['context_next_states']], dim=1)
        reward_seq = torch.cat([zeros[:, :, :1], x['context_rewards']], dim=1)

        seq = torch.cat(
            [state_seq, action_seq, next_state_seq, reward_seq], dim=2)
        stacked_inputs = self.embed_transition(seq)
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        if self.continuous_action:
            action_means = self.pred_action_means(
                transformer_outputs['last_hidden_state'])
            action_std_raw = self.pred_action_log_stds(
                transformer_outputs['last_hidden_state'])
            if self.test:
                action_means = action_means[:, -1, :]
                action_std_raw = action_std_raw[:, -1, :]
            # Scaled softplus exactly like robomimic
            action_stds = F.softplus(action_std_raw)
            action_stds = action_stds * (self.init_std / F.softplus(torch.zeros(1, device=action_std_raw.device)))
            action_stds = torch.clamp(action_stds, min=self.std_limits[0], max=self.std_limits[1])
            # Clamp and squash mean
            action_means = torch.clamp(action_means, -MEAN_CLAMP, MEAN_CLAMP)
            action_means = torch.tanh(action_means)
            dist = Independent(Normal(action_means, action_stds), 1)
            return dist
        else:
            preds = self.pred_actions(transformer_outputs['last_hidden_state'])

            if self.test:
                return preds[:, -1, :]
            return preds[:, 1:, :]


class ImageTransformer(Transformer):
    """Transformer class for image-based data."""

    def __init__(self, config):
        super().__init__(config)
        self.im_embd = 8

        size = self.config['image_size']
        size = (size - 3) // 2 + 1
        size = (size - 3) // 1 + 1

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Flatten(start_dim=1),
            nn.Linear(int(16 * size * size), self.im_embd),
            nn.ReLU(),
        )

        new_dim = self.im_embd + self.state_dim + self.action_dim + 1
        self.embed_transition = torch.nn.Linear(new_dim, self.n_embd)
        self.embed_ln = nn.LayerNorm(self.n_embd)

    def forward(self, x):
        query_images = x['query_images'][:, None, :]
        query_states = x['query_states'][:, None, :]
        context_images = x['context_images']
        context_states = x['context_states']
        context_actions = x['context_actions']
        context_rewards = x['context_rewards']

        if len(context_rewards.shape) == 2:
            context_rewards = context_rewards[:, :, None]

        batch_size = query_states.shape[0]

        image_seq = torch.cat([query_images, context_images], dim=1)
        image_seq = image_seq.view(-1, *image_seq.size()[2:])

        image_enc_seq = self.image_encoder(image_seq)
        image_enc_seq = image_enc_seq.view(batch_size, -1, self.im_embd)

        context_states = torch.cat([query_states, context_states], dim=1)
        context_actions = torch.cat([
            torch.zeros(batch_size, 1, self.action_dim).to(device),
            context_actions,
        ], dim=1)
        context_rewards = torch.cat([
            torch.zeros(batch_size, 1, 1).to(device),
            context_rewards,
        ], dim=1)

        stacked_inputs = torch.cat([
            image_enc_seq,
            context_states,
            context_actions,
            context_rewards,
        ], dim=2)
        stacked_inputs = self.embed_transition(stacked_inputs)
        stacked_inputs = self.embed_ln(stacked_inputs)

        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        preds = self.pred_actions(transformer_outputs['last_hidden_state'])

        if self.test:
            return preds[:, -1, :]
        return preds[:, 1:, :]


class MLP(nn.Module):
    """MLP class."""

    def __init__(self, config):
        super(MLP, self).__init__()

        self.config = config
        self.horizon = config['horizon']
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.model = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        if self.config['continuous_action']:
            self.pred_action_means = nn.Linear(256, self.action_dim)
            self.pred_action_log_stds = nn.Linear(256, self.action_dim)
            # Robomimic defaults for Gaussian policy
            self.init_std = config.get('init_std', 0.3)
            self.std_limits = (0.007, 7.5)  # (min, max)
        else:
            self.pred_actions = nn.Linear(256, self.action_dim)
    
    def forward(self, x, sample_time=False):
        assert not sample_time, "MLP does not support sampling time."
        query_states = x
        query_states = query_states.view(-1, self.state_dim)
        query_states = query_states.to(device)
        if self.config['continuous_action']:
            action_means = self.pred_action_means(self.model(query_states))
            action_std_raw = self.pred_action_log_stds(self.model(query_states))
            # Scaled softplus exactly like robomimic
            action_stds = F.softplus(action_std_raw)
            action_stds = action_stds * (self.init_std / F.softplus(torch.zeros(1, device=action_std_raw.device)))
            action_stds = torch.clamp(action_stds, min=self.std_limits[0], max=self.std_limits[1])
            # Clamp and squash mean
            action_means = torch.clamp(action_means, -MEAN_CLAMP, MEAN_CLAMP)
            action_means = torch.tanh(action_means)
            dist = Independent(Normal(action_means, action_stds), 1)
            return dist
        else:
            pred_actions = self.model(query_states)
            return pred_actions


class AsymmetricCritic(nn.Module):
    """
    Asymmetric Critic for AAWR (Asymmetric Advantage Weighted Regression).
    
    Uses privileged goal information during training to compute high-quality
    Q and V estimates. The policy only sees (state, reward, done) but the
    critic sees (state, reward, done, goal).
    
    Components:
    - Q-network: (state, action, goal) -> Q value
    - V-network: (state, goal) -> V value
    """

    def __init__(self, state_dim, action_dim, goal_dim, hidden_dim=256, n_layers=3):
        """
        Args:
            state_dim: Dimension of state/observation
            action_dim: Dimension of action (for discrete, this is num_actions)
            goal_dim: Dimension of privileged goal information
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
        """
        super(AsymmetricCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.hidden_dim = hidden_dim
        
        # Q-network: (state, action, goal) -> Q value
        q_layers = []
        q_input_dim = state_dim + action_dim + goal_dim
        for i in range(n_layers):
            q_layers.append(nn.Linear(q_input_dim if i == 0 else hidden_dim, hidden_dim))
            q_layers.append(nn.ReLU())
        q_layers.append(nn.Linear(hidden_dim, 1))
        self.q_network = nn.Sequential(*q_layers)
        
        # V-network: (state, goal) -> V value
        v_layers = []
        v_input_dim = state_dim + goal_dim
        for i in range(n_layers):
            v_layers.append(nn.Linear(v_input_dim if i == 0 else hidden_dim, hidden_dim))
            v_layers.append(nn.ReLU())
        v_layers.append(nn.Linear(hidden_dim, 1))
        self.v_network = nn.Sequential(*v_layers)
        
        # Target Q-network for stable training
        self.q_target = None
        
    def init_target(self):
        """Initialize target Q-network as a copy of Q-network."""
        import copy
        self.q_target = copy.deepcopy(self.q_network)
        for param in self.q_target.parameters():
            param.requires_grad = False
            
    def update_target(self, tau=0.005):
        """Soft update target Q-network."""
        if self.q_target is None:
            self.init_target()
            return
        for target_param, param in zip(self.q_target.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def q_value(self, state, action, goal):
        """
        Compute Q-value for (state, action, goal).
        
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim) - one-hot for discrete
            goal: (batch_size, goal_dim)
        
        Returns:
            q: (batch_size, 1)
        """
        x = torch.cat([state, action, goal], dim=-1)
        return self.q_network(x)
    
    def q_value_target(self, state, action, goal):
        """Compute Q-value using target network."""
        if self.q_target is None:
            return self.q_value(state, action, goal)
        x = torch.cat([state, action, goal], dim=-1)
        return self.q_target(x)
    
    def v_value(self, state, goal):
        """
        Compute V-value for (state, goal).
        
        Args:
            state: (batch_size, state_dim)
            goal: (batch_size, goal_dim)
        
        Returns:
            v: (batch_size, 1)
        """
        x = torch.cat([state, goal], dim=-1)
        return self.v_network(x)
    
    def advantage(self, state, action, goal):
        """
        Compute advantage A = Q(s,a,g) - V(s,g).
        
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)
            goal: (batch_size, goal_dim)
        
        Returns:
            advantage: (batch_size, 1)
        """
        q = self.q_value(state, action, goal)
        v = self.v_value(state, goal)
        return q - v

class DecisionTransformer(nn.Module):
    """Decision Transformer class."""

    def __init__(self, config):
        super(DecisionTransformer, self).__init__()

        self.config = config
        self.test = config['test']
        self.horizon = self.config['horizon']
        self.n_embd = self.config['n_embd']
        self.n_layer = self.config['n_layer']
        self.n_head = self.config['n_head']
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']
        self.dropout = self.config['dropout']

        gpt_config = GPT2Config(
            n_positions=self.horizon,
            n_ctx=self.horizon,
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(gpt_config)

        self.embed_transition = nn.Linear(
            self.state_dim + self.action_dim + 2, self.n_embd)
        self.embed_ln = nn.LayerNorm(self.n_embd)
        self.continuous_action = self.config['continuous_action']
        self.gmm_heads = self.config['gmm_heads']
        if self.gmm_heads > 1:
            assert self.continuous_action, "GMM only supported for continuous action spaces."
            print("Using GMM with", self.gmm_heads, "heads")
            # self.action_dim = self.action_dim * self.gmm_heads
            self.pred_action_weights = nn.Linear(self.n_embd, self.gmm_heads)
            self.pred_action_means = nn.Linear(self.n_embd, self.action_dim * self.gmm_heads)
            self.pred_action_log_stds = nn.Linear(self.n_embd, self.action_dim * self.gmm_heads)
        elif self.continuous_action:
            self.pred_action_means = nn.Linear(self.n_embd, self.action_dim)
            self.pred_action_log_stds = nn.Linear(self.n_embd, self.action_dim)
        else:
            self.pred_actions = nn.Linear(self.n_embd, self.action_dim)
        self.action_proj = nn.Linear(self.action_dim, self.n_embd)
        self.tanh_action = config.get('tanh_action', False)
        self.low_noise_eval = config.get('low_noise_eval', False)
        
        # Robomimic defaults for Gaussian policy
        self.init_std = config.get('init_std', 0.3)
        self.std_limits = (
            config.get('std_min', 0.007),  # min
            config.get('std_max', 7.5),    # max
        )
        
    def forward(self, x, query_actions=None, sample_time=False):
        states = x['states']
        actions = x['actions']
        rewards = x['rewards']
        dones = x['dones']
        input_actions = torch.cat([
            torch.zeros(states.shape[0], 1, self.action_dim).to(device),
            actions[:, :-1, :],
        ], dim=1)
        input_rewards = torch.cat([
            torch.zeros(states.shape[0], 1).to(device),
            rewards[:, :-1],
        ], dim=1)
        input_dones = torch.cat([
            torch.zeros(states.shape[0], 1).to(device),
            dones[:, :-1],
        ], dim=1)
        position_ids = None
        if sample_time:
            # assert False, "Sampling time not supported."
            position_ids = torch.arange(
                states.shape[1], device=states.device).unsqueeze(0).expand(
                    states.shape[0], -1)
            initial_timestep = torch.randint(0, self.horizon - states.shape[1], (states.shape[0],), device=states.device)
            position_ids = position_ids + initial_timestep.unsqueeze(1)

        inputs_ = torch.cat([states, input_actions, input_rewards.unsqueeze(-1), input_dones.unsqueeze(-1)], dim=2)
        inputs = self.embed_transition(inputs_)
        inputs = self.embed_ln(inputs)

        transformer_outputs = self.transformer(inputs_embeds=inputs, position_ids=position_ids)

        # query_actions = x['actions'] if query_actions is None else query_actions
        # query_actions = query_actions.to(device)
        # query_actions = self.action_proj(query_actions)
        # value_pred_inputs = torch.cat([transformer_outputs['last_hidden_state'], query_actions], dim=-1)
        # value_preds = self.pred_values(value_pred_inputs) # B x T x 1
        value_preds = None

        if self.gmm_heads > 1:
            hidden = transformer_outputs['last_hidden_state']  # [B, T, H]
            action_weights = self.pred_action_weights(hidden)
            action_means = self.pred_action_means(hidden)
            action_std_raw = self.pred_action_log_stds(hidden)
            B, T, _ = action_means.shape
            K = self.gmm_heads
            D = self.action_dim
            action_means = action_means.reshape(B, T, K, D)
            action_std_raw = action_std_raw.reshape(B, T, K, D)
            
            # Scaled softplus exactly like robomimic
            action_stds = F.softplus(action_std_raw)
            action_stds = action_stds * (self.init_std / F.softplus(torch.zeros(1, device=action_std_raw.device)))
            action_stds = torch.clamp(action_stds, min=self.std_limits[0], max=self.std_limits[1])
            
            if not self.training and self.low_noise_eval:
                action_stds = torch.ones_like(action_stds) * 1e-4
            
            # Clamp and tanh the mean
            action_means = torch.clamp(action_means, -MEAN_CLAMP, MEAN_CLAMP)
            if not self.tanh_action:
                action_means = torch.tanh(action_means)

            mixture = torch.distributions.Categorical(logits=action_weights)
            components = Independent(
                Normal(loc=action_means, scale=action_stds),  # batch: [B, T, K], event: D
                1
            )
            dist = torch.distributions.MixtureSameFamily(mixture, components)
            if self.tanh_action:
                dist = TransformedDistribution(dist, TanhTransform())
            return dist, value_preds
        if self.continuous_action:
            action_means = self.pred_action_means(
                transformer_outputs['last_hidden_state'])
            action_std_raw = self.pred_action_log_stds(
                transformer_outputs['last_hidden_state'])
            
            # Scaled softplus exactly like robomimic
            action_stds = F.softplus(action_std_raw)
            action_stds = action_stds * (self.init_std / F.softplus(torch.zeros(1, device=action_std_raw.device)))
            action_stds = torch.clamp(action_stds, min=self.std_limits[0], max=self.std_limits[1])
            
            # Low noise eval (like robomimic)
            if not self.training and self.low_noise_eval:
                action_stds = torch.ones_like(action_stds) * 1e-4
            
            # Clamp and tanh the mean to ensure actions in [-1, 1]
            action_means = torch.clamp(action_means, -MEAN_CLAMP, MEAN_CLAMP)
            if not self.tanh_action:
                action_means = torch.tanh(action_means)
            
            # Use Independent wrapper for proper log_prob computation
            dist = Independent(Normal(action_means, action_stds), 1)
            
            if self.tanh_action:
                dist = TransformedDistribution(dist, TanhTransform())
            return dist, value_preds
        preds = self.pred_actions(transformer_outputs['last_hidden_state']) # B x T x A
        return preds, value_preds
    
    def get_action(self, current_state, states, actions, rewards, dones, return_transformer_outputs=False):
        # return_value = False
        # return self.debug_mlp(current_state)  # B x D -> B x A
        if states is None: # current_state is B x D
            input_states = current_state.unsqueeze(1)
            input_actions = torch.zeros(current_state.shape[0], 1, self.action_dim).to(device)
            input_rewards = torch.zeros(current_state.shape[0], 1).to(device)
            input_dones = torch.zeros(current_state.shape[0], 1).to(device)
        else: # states is B x T x D, actions is B x T x A, rewards is B x T, dones is B x T, current_state is B x D
            input_states = torch.cat([
                states, current_state.unsqueeze(1)
            ], dim=1) # B x (T+1) x D
            input_actions = torch.cat([
                torch.zeros(states.shape[0], 1, self.action_dim).to(device),
                actions,
            ], dim=1) # B x (T+1) x A
            input_rewards = torch.cat([
                torch.zeros(states.shape[0], 1).to(device),
                rewards,
            ], dim=1) # B x (T+1)
            input_dones = torch.cat([
                torch.zeros(states.shape[0], 1).to(device),
                dones,
            ], dim=1) # B x (T+1)

            input_states = input_states[:, -self.horizon:, :]  # Keep only the last horizon states
            input_actions = input_actions[:, -self.horizon:, :]
            input_rewards = input_rewards[:, -self.horizon:]
            input_dones = input_dones[:, -self.horizon:]
        
        inputs = torch.cat([input_states, input_actions, input_rewards.unsqueeze(-1), input_dones.unsqueeze(-1)], dim=2)
        inputs = self.embed_transition(inputs)
        inputs = self.embed_ln(inputs)
        transformer_outputs = self.transformer(inputs_embeds=inputs)

        if self.gmm_heads > 1:
            hidden = transformer_outputs['last_hidden_state']
            action_weights = self.pred_action_weights(hidden)[:, -1, :]
            action_means = self.pred_action_means(hidden)[:, -1, :]
            action_std_raw = self.pred_action_log_stds(hidden)[:, -1, :]
            B, _ = action_means.shape
            K = self.gmm_heads
            D = self.action_dim
            action_means = action_means.reshape(B, K, D)
            action_std_raw = action_std_raw.reshape(B, K, D)
            
            # Scaled softplus exactly like robomimic
            action_stds = F.softplus(action_std_raw)
            action_stds = action_stds * (self.init_std / F.softplus(torch.zeros(1, device=action_std_raw.device)))
            action_stds = torch.clamp(action_stds, min=self.std_limits[0], max=self.std_limits[1])
            
            if not self.training and self.low_noise_eval:
                action_stds = torch.ones_like(action_stds) * 1e-4
            
            # Clamp and tanh the mean
            action_means = torch.clamp(action_means, -MEAN_CLAMP, MEAN_CLAMP)
            if not self.tanh_action:
                action_means = torch.tanh(action_means)
            
            mixture = torch.distributions.Categorical(logits=action_weights)
            components = Independent(
                Normal(loc=action_means, scale=action_stds),  # batch: [B, K], event: D
                1
            )
            dist = torch.distributions.MixtureSameFamily(mixture, components)
            if self.tanh_action:
                dist = TransformedDistribution(dist, TanhTransform())
            if return_transformer_outputs:
                return dist, transformer_outputs['last_hidden_state'][:, -1, :]
            return dist
        if self.continuous_action:
            action_means = self.pred_action_means(
                transformer_outputs['last_hidden_state'])[:, -1, :]
            action_std_raw = self.pred_action_log_stds(
                transformer_outputs['last_hidden_state'])[:, -1, :]
            
            # Scaled softplus exactly like robomimic
            action_stds = F.softplus(action_std_raw)
            action_stds = action_stds * (self.init_std / F.softplus(torch.zeros(1, device=action_std_raw.device)))
            action_stds = torch.clamp(action_stds, min=self.std_limits[0], max=self.std_limits[1])
            
            # Low noise eval
            if not self.training and self.low_noise_eval:
                action_stds = torch.ones_like(action_stds) * 1e-4
            
            # Clamp and tanh the mean
            action_means = torch.clamp(action_means, -MEAN_CLAMP, MEAN_CLAMP)
            if not self.tanh_action:
                action_means = torch.tanh(action_means)
            
            # Use Independent wrapper
            dist = Independent(Normal(action_means, action_stds), 1)
            
            if self.tanh_action:
                dist = TransformedDistribution(dist, TanhTransform())
            if return_transformer_outputs:
                return dist, transformer_outputs['last_hidden_state'][:, -1, :]
            return dist
        preds = self.pred_actions(transformer_outputs['last_hidden_state']) # B x (T+1) x A
        if return_transformer_outputs:
            return preds[:, -1, :], transformer_outputs['last_hidden_state'][:, -1, :]
        return preds[:, -1, :]  # Return the last action


class DecisionTransformerCnn(nn.Module):
    """Decision Transformer with image/grid encoder for observations.

    Observations arrive as (H, W, C) uint8/float from procgen / gymnasium.

    Encoder selection (automatic, based on spatial size):
    * **H ≥ 32** (e.g. 64×64 RGB): Atari-style CNN
      Conv(k=8,s=4) → Conv(k=4,s=2) → Conv(k=3,s=1) → Linear.
    * **H < 32** (e.g. 7×7 binary grid): lightweight MLP
      Flatten → Linear → ReLU → Linear → ReLU.

    Config must contain:
        obs: tuple (H, W, C) – raw observation shape
        n_embd: int – hidden dimension (encoder output dim)
        ... (all keys required by DecisionTransformer)
    """

    def __init__(self, config):
        super().__init__()
        gpt2_config = GPT2Config(
            n_positions=config['horizon'],
            n_ctx=config['horizon'],
            n_embd=config['n_embd'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            resid_pdrop=config['dropout'],
            embd_pdrop=config['dropout'],
            attn_pdrop=config['dropout'],
            use_cache=False,
        )
        self.transformer = GPT2Model(gpt2_config)
        obs_shape = config['obs']  # (H, W, C), (15x15x3)
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(obs_shape[2], 64, kernel_size=4, stride=2),  # (H/2, W/2, 64)
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=4, stride=2),  # (H/4, W/4, 32)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, config['n_embd']),  # (n_embd)
            # nn.Linear(obs_shape[0]//4 * obs_shape[1]//4 * 32, config['n_embd']),  # (n_embd)
            # nn.ReLU(),
        )
        # state_dim = np.prod(obs_shape)
        state_dim = config['n_embd']
        action_dim = 4
        n_embd = config['n_embd']
        self.embed_transition = nn.Linear(
            state_dim + action_dim + 2, n_embd)
        self.embed_ln = nn.LayerNorm(n_embd)
        self.pred_actions = nn.Linear(n_embd, action_dim)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_embd = n_embd

    def forward(self, x, **kwargs):
        B,T = x['states'].shape[0], x['states'].shape[1]
        # states = x['states'].view(B, T, self.state_dim)
        states = x['states'].view(B * T, *x['states'].shape[2:])  # (B*T, H, W, C)
        states = states.permute(0, 3, 1, 2)
        states = self.cnn_encoder(states)  # (B*T, n_embd)
        states = states.view(B, T, self.n_embd)  # (B, T, n_embd)
        actions = x['actions']
        rewards = x['rewards']
        dones = x['dones']
        input_actions = torch.cat([
            torch.zeros(states.shape[0], 1, self.action_dim).to(device),
            actions[:, :-1, :],
        ], dim=1)
        input_rewards = torch.cat([
            torch.zeros(states.shape[0], 1).to(device),
            rewards[:, :-1],
        ], dim=1)
        input_dones = torch.cat([
            torch.zeros(states.shape[0], 1).to(device),
            dones[:, :-1],
        ], dim=1)
        inputs_ = torch.cat([states, input_actions, input_rewards.unsqueeze(-1), input_dones.unsqueeze(-1)], dim=2)
        inputs = self.embed_transition(inputs_)
        inputs = self.embed_ln(inputs)

        transformer_outputs = self.transformer(inputs_embeds=inputs)
        preds = self.pred_actions(transformer_outputs['last_hidden_state']) # B x T x A
        return preds

    def get_action(self, states, actions, rewards, dones, attention_mask):
        # current_state: [B, H, W, C] -> [B, hidden_dim]
        B,T = states.shape[0], states.shape[1]
        # states = states.view(B, T, self.state_dim)
        states = states.view(B * T, *states.shape[2:])  # (B*T, H, W, C)
        states = states.permute(0, 3, 1, 2)
        states = self.cnn_encoder(states)  # (B*T, n_embd)
        states = states.view(B, T, self.n_embd)  # (B, T, n_embd)
        inputs_ = torch.cat([states, actions, rewards.unsqueeze(-1), dones.unsqueeze(-1)], dim=2)
        inputs = self.embed_transition(inputs_)
        inputs = self.embed_ln(inputs)
        transformer_outputs = self.transformer(inputs_embeds=inputs, attention_mask=attention_mask)
        preds = self.pred_actions(transformer_outputs['last_hidden_state']) # B x T x A
        seq_len = attention_mask.sum(dim=1).long()  # [B]
        last_preds = preds[torch.arange(B), seq_len - 1]  # [B, A]
        return last_preds