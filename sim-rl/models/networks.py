"""Shared actor/critic network definitions used by both training and testing."""

import torch
import torch.nn as nn
from tensordict import TensorDict

ACTIVATION_MAP = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}


class SimpleActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256, 128), activation="elu"):
        super().__init__()
        act_fn = ACTIVATION_MAP[activation]
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act_fn()]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.is_recurrent = False
        self.output_distribution_params = ()
        self.output_entropy = torch.tensor(0.0)

    def forward(self, obs_td, stochastic_output=False, **kwargs):
        obs = obs_td["obs"] if isinstance(obs_td, TensorDict) else obs_td
        mean = self.net(obs)
        std = self.log_std.exp().expand_as(mean)
        self.output_distribution_params = (mean, std)
        if stochastic_output:
            dist = torch.distributions.Normal(mean, std)
            self.output_entropy = dist.entropy().sum(-1)
            return dist.sample()
        self.output_entropy = torch.zeros(mean.shape[0], device=mean.device)
        return mean

    def get_output_log_prob(self, actions):
        mean, std = self.output_distribution_params
        return torch.distributions.Normal(mean, std).log_prob(actions).sum(-1)

    def get_kl_divergence(self, old_params, new_params):
        old_dist = torch.distributions.Normal(*old_params)
        new_dist = torch.distributions.Normal(*new_params)
        return torch.distributions.kl_divergence(old_dist, new_dist).sum(-1)

    def get_hidden_state(self): return None
    def reset(self, dones=None): pass
    def update_normalization(self, obs): pass


class SimpleCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dims=(256, 256, 128), activation="elu"):
        super().__init__()
        act_fn = ACTIVATION_MAP[activation]
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act_fn()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.is_recurrent = False

    def forward(self, obs_td, **kwargs):
        obs = obs_td["obs"] if isinstance(obs_td, TensorDict) else obs_td
        return self.net(obs)

    def get_hidden_state(self): return None
    def reset(self, dones=None): pass
    def update_normalization(self, obs): pass
