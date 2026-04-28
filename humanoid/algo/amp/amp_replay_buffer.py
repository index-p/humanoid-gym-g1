import torch


class AMPReplayBuffer:
    def __init__(self, buffer_size, amp_obs_dim, device="cpu"):
        self.buffer_size = int(buffer_size)
        self.device = device
        self.amp_obs = torch.zeros(self.buffer_size, amp_obs_dim, device=device)
        self.next_amp_obs = torch.zeros(self.buffer_size, amp_obs_dim, device=device)
        self.position = 0
        self.full = False

    def insert(self, amp_obs, next_amp_obs):
        if amp_obs.numel() == 0:
            return

        count = amp_obs.shape[0]
        if count >= self.buffer_size:
            self.amp_obs.copy_(amp_obs[-self.buffer_size:])
            self.next_amp_obs.copy_(next_amp_obs[-self.buffer_size:])
            self.position = 0
            self.full = True
            return

        end = self.position + count
        if end <= self.buffer_size:
            self.amp_obs[self.position:end].copy_(amp_obs)
            self.next_amp_obs[self.position:end].copy_(next_amp_obs)
        else:
            split = self.buffer_size - self.position
            self.amp_obs[self.position:].copy_(amp_obs[:split])
            self.next_amp_obs[self.position:].copy_(next_amp_obs[:split])
            remainder = count - split
            self.amp_obs[:remainder].copy_(amp_obs[split:])
            self.next_amp_obs[:remainder].copy_(next_amp_obs[split:])
            self.full = True
        self.position = end % self.buffer_size
        if self.position == 0:
            self.full = True

    def sample(self, batch_size):
        size = len(self)
        if size == 0:
            raise ValueError("AMP replay buffer is empty")
        batch_size = min(batch_size, size)
        indices = torch.randint(0, size, (batch_size,), device=self.device)
        return self.amp_obs[indices], self.next_amp_obs[indices]

    def __len__(self):
        return self.buffer_size if self.full else self.position
