import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from humanoid.algo.ppo.ppo import PPO

from .amp_discriminator import AMPDiscriminator
from .amp_replay_buffer import AMPReplayBuffer
from .amp_rollout_storage import AMPRolloutStorage


class AMPPPO(PPO):
    def __init__(
        self,
        actor_critic,
        amp_obs_dim,
        amp_reward_coef=1.0,
        amp_task_reward_lerp=0.3,
        amp_discr_hidden_dims=None,
        amp_discr_learning_rate=1e-4,
        amp_discr_batch_size=4096,
        amp_replay_buffer_size=200000,
        amp_grad_penalty_coef=10.0,
        device="cpu",
        **kwargs,
    ):
        super().__init__(actor_critic, device=device, **kwargs)
        self.amp_obs_dim = amp_obs_dim
        self.amp_reward_coef = amp_reward_coef
        self.amp_task_reward_lerp = amp_task_reward_lerp
        self.amp_discr_batch_size = amp_discr_batch_size
        self.amp_grad_penalty_coef = amp_grad_penalty_coef
        self.discriminator = AMPDiscriminator(
            amp_obs_dim, hidden_dims=amp_discr_hidden_dims
        ).to(self.device)
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=amp_discr_learning_rate
        )
        self.replay_buffer = AMPReplayBuffer(
            amp_replay_buffer_size, amp_obs_dim, device=self.device
        )
        self.motion_loader = None
        self.transition = AMPRolloutStorage.Transition()
        self.last_update_stats = {
            "value_loss": 0.0,
            "surrogate_loss": 0.0,
            "discriminator_loss": 0.0,
            "amp_reward": 0.0,
        }

    def set_motion_loader(self, motion_loader):
        self.motion_loader = motion_loader

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ):
        self.storage = AMPRolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            [self.amp_obs_dim],
            device=self.device,
        )

    def predict_amp_reward(self, amp_obs, next_amp_obs):
        with torch.inference_mode():
            logits = self.discriminator(amp_obs, next_amp_obs)
            prob = torch.sigmoid(logits)
            reward = -torch.log(torch.clamp(1.0 - prob, min=1e-4))
        return reward.squeeze(-1)

    def process_env_step(self, rewards, dones, infos, amp_obs, next_amp_obs):
        task_rewards = rewards.clone()
        amp_rewards = self.predict_amp_reward(amp_obs, next_amp_obs)
        mixed_rewards = (1.0 - self.amp_task_reward_lerp) * task_rewards + (
            self.amp_task_reward_lerp * self.amp_reward_coef * amp_rewards
        )

        self.transition.rewards = mixed_rewards.clone()
        self.transition.dones = dones
        self.transition.amp_observations = amp_obs
        self.transition.next_amp_observations = next_amp_obs

        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        self.storage.add_transitions(self.transition)
        self.replay_buffer.insert(amp_obs, next_amp_obs)
        self.transition.clear()
        self.actor_critic.reset(dones)
        self.last_update_stats["amp_reward"] = amp_rewards.mean().item()

    def _compute_discriminator_loss(self):
        if self.motion_loader is None or len(self.motion_loader) == 0 or len(self.replay_buffer) == 0:
            return torch.tensor(0.0, device=self.device)

        batch_size = min(self.amp_discr_batch_size, len(self.replay_buffer), len(self.motion_loader))
        policy_amp_obs, policy_next_amp_obs = self.replay_buffer.sample(batch_size)
        expert_amp_obs, expert_next_amp_obs = self.motion_loader.sample(batch_size)

        policy_logits = self.discriminator(policy_amp_obs, policy_next_amp_obs)
        expert_logits = self.discriminator(expert_amp_obs, expert_next_amp_obs)

        policy_loss = F.binary_cross_entropy_with_logits(
            policy_logits, torch.zeros_like(policy_logits)
        )
        expert_loss = F.binary_cross_entropy_with_logits(
            expert_logits, torch.ones_like(expert_logits)
        )

        expert_inputs = torch.cat((expert_amp_obs, expert_next_amp_obs), dim=-1).detach()
        expert_inputs.requires_grad_(True)
        expert_scores = self.discriminator.model(expert_inputs)
        gradients = torch.autograd.grad(
            outputs=expert_scores.sum(),
            inputs=expert_inputs,
            create_graph=True,
        )[0]
        grad_penalty = gradients.pow(2).sum(dim=-1).mean()

        return policy_loss + expert_loss + self.amp_grad_penalty_coef * grad_penalty

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            self.actor_critic.act(
                obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
            )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        discriminator_loss = self._compute_discriminator_loss()
        if discriminator_loss.requires_grad:
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()
            discriminator_loss_value = discriminator_loss.item()
        else:
            discriminator_loss_value = float(discriminator_loss.item())

        self.last_update_stats = {
            "value_loss": mean_value_loss,
            "surrogate_loss": mean_surrogate_loss,
            "discriminator_loss": discriminator_loss_value,
            "amp_reward": self.last_update_stats.get("amp_reward", 0.0),
        }
        self.storage.clear()
        return self.last_update_stats
