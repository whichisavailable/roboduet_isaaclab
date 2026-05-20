# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .automatic_rollout_storage import AutomaticRolloutStorage
from .callable_resolver import resolve_callable


class AutomaticPPO:
    """Per-policy PPO aligned with upstream Roboduet `auto_train`."""

    def __init__(
        self,
        actor_critic,
        device: str = "cpu",
        *,
        value_loss_coef: float = 1.0,
        use_clipped_value_loss: bool = True,
        clip_param: float = 0.2,
        entropy_coef: float = 0.01,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        learning_rate: float = 5.0e-4,
        adaptation_module_learning_rate: float = 5.0e-4,
        num_adaptation_module_substeps: int = 1,
        schedule: str = "adaptive",
        gamma: float = 0.99,
        lam: float = 0.95,
        desired_kl: float = 0.01,
        max_grad_norm: float = 1.0,
        selective_adaptation_module_loss: bool = False,
        symmetry_callable=None,
        symmetry_mirror_callable=None,
        symmetry_loss_coef: float = 1.0,
    ):
        self.device = device
        self.actor_critic = actor_critic.to(device)
        self.storage = None
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.adaptation_module_optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=adaptation_module_learning_rate
        )
        self.transition = AutomaticRolloutStorage.Transition()

        self.value_loss_coef = value_loss_coef
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.learning_rate = learning_rate
        self.schedule = schedule
        self.gamma = gamma
        self.lam = lam
        self.desired_kl = desired_kl
        self.max_grad_norm = max_grad_norm
        self.num_adaptation_module_substeps = num_adaptation_module_substeps
        self.selective_adaptation_module_loss = selective_adaptation_module_loss
        self.symmetry = resolve_callable(symmetry_callable) if symmetry_callable else None
        self.symmetry_mirror_action = resolve_callable(symmetry_mirror_callable) if symmetry_mirror_callable else None
        self.symmetry_enabled = self.symmetry is not None
        if self.symmetry_enabled and self.symmetry_mirror_action is None:
            raise ValueError("symmetry_mirror_callable is required when symmetry_callable is enabled.")
        self.symmetry_loss_coef = float(symmetry_loss_coef)
        self.last_symmetry_loss = 0.0

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: list[int],
        privileged_obs_shape: list[int],
        obs_history_shape: list[int],
        action_shape: list[int],
        action_distribution_shape: list[int],
    ):
        self.storage = AutomaticRolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            privileged_obs_shape,
            obs_history_shape,
            action_shape,
            action_distribution_shape,
            self.device,
        )
        self._zero_env_bins = torch.zeros(num_envs, 1, dtype=torch.long, device=self.device)

    def train_mode(self):
        self.actor_critic.train()

    def eval_mode(self):
        self.actor_critic.eval()

    def act(self, obs: torch.Tensor, privileged_obs: torch.Tensor, obs_history: torch.Tensor):
        self.transition.actions = self.actor_critic.act(obs_history).detach()
        self.transition.values = self.actor_critic.evaluate(obs_history, privileged_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_histories = obs_history
        return self.transition.actions

    def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor, infos: dict):
        self.transition.rewards = rewards
        self.transition.dones = dones
        if "time_outs" in infos:
            self.transition.rewards = rewards.clone()
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.transition.env_bins = self._zero_env_bins
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs: torch.Tensor, last_critic_privileged_obs: torch.Tensor):
        last_values = self.actor_critic.evaluate(last_critic_obs, last_critic_privileged_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, un_adapt: bool = False):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_adaptation_module_loss = 0.0
        mean_adaptation_module_test_loss = 0.0
        mean_symmetry_loss = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            _obs_batch,
            _critic_obs_batch,
            privileged_obs_batch,
            obs_history_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            masks_batch,
            _env_bins_batch,
        ) in generator:
            symmetry_data = None
            batch_target_values = target_values_batch
            batch_advantages = advantages_batch
            batch_returns = returns_batch
            batch_old_actions_log_prob = old_actions_log_prob_batch
            batch_old_mu = old_mu_batch
            batch_old_sigma = old_sigma_batch
            batch_privileged_obs = privileged_obs_batch
            batch_obs_history = obs_history_batch
            batch_actions = actions_batch

            if self.symmetry_enabled:
                symmetry_data = self.symmetry(
                    privileged_obs_batch=privileged_obs_batch,
                    obs_history_batch=obs_history_batch,
                    actions_batch=actions_batch,
                    old_mu_batch=old_mu_batch,
                    old_sigma_batch=old_sigma_batch,
                )
                batch_target_values = torch.cat((target_values_batch, target_values_batch), dim=0)
                batch_advantages = torch.cat((advantages_batch, advantages_batch), dim=0)
                batch_returns = torch.cat((returns_batch, returns_batch), dim=0)
                batch_old_actions_log_prob = torch.cat(
                    (old_actions_log_prob_batch, symmetry_data["mirrored_old_actions_log_prob"]), dim=0
                )
                batch_old_mu = torch.cat((old_mu_batch, symmetry_data["mirrored_old_mu"]), dim=0)
                batch_old_sigma = torch.cat((old_sigma_batch, symmetry_data["mirrored_old_sigma"]), dim=0)
                batch_privileged_obs = torch.cat(
                    (privileged_obs_batch, symmetry_data["mirrored_privileged_obs"]), dim=0
                )
                batch_obs_history = torch.cat((obs_history_batch, symmetry_data["mirrored_obs_history"]), dim=0)
                batch_actions = torch.cat((actions_batch, symmetry_data["mirrored_actions"]), dim=0)

            self.actor_critic.act(batch_obs_history, masks=masks_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(batch_actions)
            value_batch = self.actor_critic.evaluate(batch_obs_history, batch_privileged_obs, masks=masks_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / batch_old_sigma + 1.0e-5)
                        + (torch.square(batch_old_sigma) + torch.square(batch_old_mu - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(batch_old_actions_log_prob))
            surrogate = -torch.squeeze(batch_advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch_advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch_target_values + (value_batch - batch_target_values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - batch_returns).pow(2)
                value_losses_clipped = (value_clipped - batch_returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch_returns - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            symmetry_loss = torch.zeros((), device=self.device)
            if self.symmetry_enabled and symmetry_data is not None:
                original_count = obs_history_batch.shape[0]
                mirrored_mu_target = self.symmetry_mirror_action(mu_batch[:original_count])
                symmetry_loss = F.mse_loss(mu_batch[original_count:], mirrored_mu_target)
                loss = loss + self.symmetry_loss_coef * symmetry_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_symmetry_loss += symmetry_loss.item()

            if not un_adapt:
                data_size = batch_privileged_obs.shape[0]
                num_train = int(data_size // 5 * 4)
                for _ in range(self.num_adaptation_module_substeps):
                    adaptation_pred = self.actor_critic.adaptation_module(batch_obs_history)
                    with torch.no_grad():
                        adaptation_target = batch_privileged_obs

                    selection_indices = torch.linspace(
                        0,
                        adaptation_pred.shape[1] - 1,
                        steps=adaptation_pred.shape[1],
                        dtype=torch.long,
                        device=adaptation_pred.device,
                    )
                    if self.selective_adaptation_module_loss:
                        selection_indices = torch.tensor([0], dtype=torch.long, device=adaptation_pred.device)

                    adaptation_loss = F.mse_loss(
                        adaptation_pred[:num_train, selection_indices],
                        adaptation_target[:num_train, selection_indices],
                    )
                    adaptation_test_loss = F.mse_loss(
                        adaptation_pred[num_train:, selection_indices],
                        adaptation_target[num_train:, selection_indices],
                    )

                    self.adaptation_module_optimizer.zero_grad()
                    adaptation_loss.backward()
                    self.adaptation_module_optimizer.step()

                    mean_adaptation_module_loss += adaptation_loss.item()
                    mean_adaptation_module_test_loss += adaptation_test_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_adaptation_module_loss /= max(num_updates * self.num_adaptation_module_substeps, 1)
        mean_adaptation_module_test_loss /= max(num_updates * self.num_adaptation_module_substeps, 1)
        mean_symmetry_loss /= num_updates
        self.last_symmetry_loss = mean_symmetry_loss
        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_adaptation_module_loss,
            0.0,
            0.0,
            mean_adaptation_module_test_loss,
            0.0,
            0.0,
        )
