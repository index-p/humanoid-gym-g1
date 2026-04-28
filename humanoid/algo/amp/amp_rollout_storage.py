import torch

from humanoid.algo.ppo.rollout_storage import RolloutStorage


class AMPRolloutStorage(RolloutStorage):
    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.amp_observations = None
            self.next_amp_observations = None

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        amp_obs_shape,
        device="cpu",
    ):
        super().__init__(
            num_envs,
            num_transitions_per_env,
            obs_shape,
            privileged_obs_shape,
            actions_shape,
            device=device,
        )
        self.amp_observations = torch.zeros(
            num_transitions_per_env, num_envs, *amp_obs_shape, device=self.device
        )
        self.next_amp_observations = torch.zeros(
            num_transitions_per_env, num_envs, *amp_obs_shape, device=self.device
        )

    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        super().add_transitions(transition)
        self.amp_observations[self.step - 1].copy_(transition.amp_observations)
        self.next_amp_observations[self.step - 1].copy_(transition.next_amp_observations)

    def amp_mini_batch_generator(self, num_mini_batches, num_epochs=1):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)
        amp_obs = self.amp_observations.flatten(0, 1)
        next_amp_obs = self.next_amp_observations.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                yield amp_obs[batch_idx], next_amp_obs[batch_idx]
