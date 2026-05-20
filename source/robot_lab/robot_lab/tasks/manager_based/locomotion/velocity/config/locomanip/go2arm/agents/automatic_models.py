# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "selu":
        return nn.SELU()
    if name == "relu":
        return nn.ReLU()
    if name == "crelu":
        return nn.ReLU()
    if name == "lrelu":
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Invalid activation function '{name}'.")


class DogActorCriticCfg:
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"
    adaptation_module_branch_hidden_dims = [256, 128]
    use_decoder = False


class DogActorCritic(nn.Module):
    """Dog policy used by upstream Roboduet `auto_train`."""

    is_recurrent = False

    def __init__(
        self,
        num_obs: int,
        num_privileged_obs: int,
        num_obs_history: int,
        num_actions: int,
        init_noise_std: float = 1.0,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        adaptation_module_branch_hidden_dims: list[int] | None = None,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(
                "DogActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )

        self.decoder = False
        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs
        actor_hidden_dims = actor_hidden_dims or DogActorCriticCfg.actor_hidden_dims
        critic_hidden_dims = critic_hidden_dims or DogActorCriticCfg.critic_hidden_dims
        adaptation_module_branch_hidden_dims = (
            adaptation_module_branch_hidden_dims or DogActorCriticCfg.adaptation_module_branch_hidden_dims
        )
        act = _activation(activation)

        adaptation_layers: list[nn.Module] = [
            nn.Linear(self.num_obs_history, adaptation_module_branch_hidden_dims[0]),
            act,
        ]
        for idx, hidden_dim in enumerate(adaptation_module_branch_hidden_dims):
            if idx == len(adaptation_module_branch_hidden_dims) - 1:
                adaptation_layers.append(nn.Linear(hidden_dim, self.num_privileged_obs))
            else:
                adaptation_layers.append(nn.Linear(hidden_dim, adaptation_module_branch_hidden_dims[idx + 1]))
                adaptation_layers.append(act)
        self.adaptation_module = nn.Sequential(*adaptation_layers)

        actor_layers: list[nn.Module] = [
            nn.Linear(self.num_privileged_obs + self.num_obs_history, actor_hidden_dims[0]),
            act,
        ]
        for idx, hidden_dim in enumerate(actor_hidden_dims):
            if idx == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(hidden_dim, num_actions))
            else:
                actor_layers.append(nn.Linear(hidden_dim, actor_hidden_dims[idx + 1]))
                actor_layers.append(act)
        self.actor_body = nn.Sequential(*actor_layers)

        critic_layers: list[nn.Module] = [
            nn.Linear(self.num_privileged_obs + self.num_obs_history, critic_hidden_dims[0]),
            act,
        ]
        for idx, hidden_dim in enumerate(critic_hidden_dims):
            if idx == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(hidden_dim, 1))
            else:
                critic_layers.append(nn.Linear(hidden_dim, critic_hidden_dims[idx + 1]))
                critic_layers.append(act)
        self.critic_body = nn.Sequential(*critic_layers)

        self.std = nn.Parameter(float(init_noise_std) * torch.ones(num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        del dones

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observation_history: torch.Tensor):
        latent = self.adaptation_module(observation_history)
        mean = self.actor_body(torch.cat((observation_history, latent), dim=-1))
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observation_history: torch.Tensor, **kwargs):
        del kwargs
        self.update_distribution(observation_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info: dict | None = None):
        return self.act_teacher(ob["obs_history"], ob["privileged_obs"], policy_info=policy_info)

    def act_inference(self, ob, policy_info: dict | None = None):
        if isinstance(ob, dict):
            return self.act_student(ob["obs_history"], policy_info=policy_info)
        return self.act_student(ob, policy_info=policy_info)

    def act_student(self, observation_history: torch.Tensor, policy_info: dict | None = None):
        latent = self.adaptation_module(observation_history)
        actions_mean = self.actor_body(torch.cat((observation_history, latent), dim=-1))
        if policy_info is not None:
            policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean

    def act_teacher(
        self,
        observation_history: torch.Tensor,
        privileged_info: torch.Tensor,
        policy_info: dict | None = None,
    ):
        actions_mean = self.actor_body(torch.cat((observation_history, privileged_info), dim=-1))
        if policy_info is not None:
            policy_info["latents"] = privileged_info
        return actions_mean

    def evaluate(self, observation_history: torch.Tensor, privileged_observations: torch.Tensor, **kwargs):
        del kwargs
        return self.critic_body(torch.cat((observation_history, privileged_observations), dim=-1))

    def get_student_latent(self, observation_history: torch.Tensor):
        return self.adaptation_module(observation_history)


class ArmActorCriticCfg:
    init_noise_std = 0.1
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"
    adaptation_module_branch_hidden_dims = [256, 128]
    use_decoder = False


class ArmActorCritic(nn.Module):
    """Arm policy used by upstream Roboduet `auto_train`."""

    is_recurrent = False

    def __init__(
        self,
        num_obs: int,
        num_privileged_obs: int,
        num_obs_history: int,
        num_actions: int,
        init_noise_std: float = 0.1,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        adaptation_module_branch_hidden_dims: list[int] | None = None,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(
                "ArmActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )

        self.decoder = False
        self.num_obs = num_obs
        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs
        actor_hidden_dims = actor_hidden_dims or ArmActorCriticCfg.actor_hidden_dims
        critic_hidden_dims = critic_hidden_dims or ArmActorCriticCfg.critic_hidden_dims
        adaptation_module_branch_hidden_dims = (
            adaptation_module_branch_hidden_dims or ArmActorCriticCfg.adaptation_module_branch_hidden_dims
        )
        act = _activation(activation)

        adaptation_layers: list[nn.Module] = [
            nn.Linear(self.num_obs_history, adaptation_module_branch_hidden_dims[0]),
            act,
        ]
        for idx, hidden_dim in enumerate(adaptation_module_branch_hidden_dims):
            if idx == len(adaptation_module_branch_hidden_dims) - 1:
                adaptation_layers.append(nn.Linear(hidden_dim, self.num_privileged_obs))
            else:
                adaptation_layers.append(nn.Linear(hidden_dim, adaptation_module_branch_hidden_dims[idx + 1]))
                adaptation_layers.append(act)
        self.adaptation_module = nn.Sequential(*adaptation_layers)

        self.actor_history_encoder = nn.Sequential(
            nn.Linear(self.num_obs_history - self.num_obs, actor_hidden_dims[0]),
            act,
            nn.Linear(actor_hidden_dims[0], actor_hidden_dims[1]),
            act,
            nn.Linear(actor_hidden_dims[1], actor_hidden_dims[2]),
        )
        actor_layers: list[nn.Module] = [
            nn.Linear(self.num_obs + self.num_privileged_obs + actor_hidden_dims[2], actor_hidden_dims[0]),
            act,
        ]
        for idx, hidden_dim in enumerate(actor_hidden_dims):
            if idx == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(hidden_dim, num_actions))
            else:
                actor_layers.append(nn.Linear(hidden_dim, actor_hidden_dims[idx + 1]))
                actor_layers.append(act)
        self.actor_body = nn.Sequential(*actor_layers)

        self.critic_history_encoder = nn.Sequential(
            nn.Linear(self.num_obs_history - self.num_obs, critic_hidden_dims[0]),
            act,
            nn.Linear(critic_hidden_dims[0], critic_hidden_dims[1]),
            act,
            nn.Linear(critic_hidden_dims[1], critic_hidden_dims[2]),
        )
        critic_layers: list[nn.Module] = [
            nn.Linear(self.num_obs + self.num_privileged_obs + critic_hidden_dims[2], critic_hidden_dims[0]),
            act,
        ]
        for idx, hidden_dim in enumerate(critic_hidden_dims):
            if idx == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(hidden_dim, 1))
            else:
                critic_layers.append(nn.Linear(hidden_dim, critic_hidden_dims[idx + 1]))
                critic_layers.append(act)
        self.critic_body = nn.Sequential(*critic_layers)

        self.std = nn.Parameter(float(init_noise_std) * torch.ones(num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        del dones

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observation_history: torch.Tensor):
        obs = observation_history[..., -self.num_obs :]
        latent = self.adaptation_module(observation_history)
        history_latent = self.actor_history_encoder(observation_history[..., : -self.num_obs])
        mean = self.actor_body(torch.cat((obs, latent, history_latent), dim=-1))
        mean[..., -2:] = torch.tanh(mean[..., -2:])
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observation_history: torch.Tensor, **kwargs):
        del kwargs
        self.update_distribution(observation_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info: dict | None = None):
        return self.act_teacher(ob["obs_history"], ob["privileged_obs"], policy_info=policy_info)

    def act_inference(self, ob, policy_info: dict | None = None):
        if isinstance(ob, dict):
            return self.act_student(ob["obs_history"], policy_info=policy_info)
        return self.act_student(ob, policy_info=policy_info)

    def act_student(self, observation_history: torch.Tensor, policy_info: dict | None = None):
        obs = observation_history[..., -self.num_obs :]
        latent = self.adaptation_module(observation_history)
        history_latent = self.actor_history_encoder(observation_history[..., : -self.num_obs])
        actions_mean = self.actor_body(torch.cat((obs, latent, history_latent), dim=-1))
        actions_mean[..., -2:] = torch.tanh(actions_mean[..., -2:])
        if policy_info is not None:
            policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean

    def act_teacher(
        self,
        observation_history: torch.Tensor,
        privileged_info: torch.Tensor,
        policy_info: dict | None = None,
    ):
        obs = observation_history[..., -self.num_obs :]
        history_latent = self.actor_history_encoder(observation_history[..., : -self.num_obs])
        actions_mean = self.actor_body(torch.cat((obs, privileged_info, history_latent), dim=-1))
        actions_mean[..., -2:] = torch.tanh(actions_mean[..., -2:])
        if policy_info is not None:
            policy_info["latents"] = privileged_info
        return actions_mean

    def evaluate(self, observation_history: torch.Tensor, privileged_observations: torch.Tensor, **kwargs):
        del kwargs
        obs = observation_history[..., -self.num_obs :]
        history_latent = self.critic_history_encoder(observation_history[..., : -self.num_obs])
        return self.critic_body(torch.cat((obs, privileged_observations, history_latent), dim=-1))

    def get_student_latent(self, observation_history: torch.Tensor):
        return self.adaptation_module(observation_history)
