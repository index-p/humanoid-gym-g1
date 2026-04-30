import torch


class AMPRunningNormalizer:
    def __init__(self, obs_dim, epsilon=1e-5, device="cpu"):
        self.obs_dim = int(obs_dim)
        self.epsilon = float(epsilon)
        self.device = device
        self.count = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.mean = torch.zeros(self.obs_dim, dtype=torch.float32, device=device)
        self.var = torch.ones(self.obs_dim, dtype=torch.float32, device=device)

    def update(self, values):
        if values is None or values.numel() == 0:
            return
        values = values.detach().reshape(-1, self.obs_dim).to(self.device)
        batch_count = float(values.shape[0])
        if batch_count == 0:
            return

        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)

        if self.count.item() == 0:
            self.mean.copy_(batch_mean)
            self.var.copy_(torch.clamp(batch_var, min=self.epsilon))
            self.count.fill_(batch_count)
            return

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / total_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        correction = delta.pow(2) * (self.count * batch_count / total_count)
        new_var = (m_a + m_b + correction) / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(torch.clamp(new_var, min=self.epsilon))
        self.count.copy_(total_count)

    def normalize(self, values):
        if values is None or values.numel() == 0 or self.count.item() == 0:
            return values
        return (values - self.mean) / torch.sqrt(self.var + self.epsilon)

    def state_dict(self):
        return {
            "count": self.count.clone(),
            "mean": self.mean.clone(),
            "var": self.var.clone(),
            "epsilon": self.epsilon,
            "obs_dim": self.obs_dim,
        }

    def load_state_dict(self, state_dict):
        self.count.copy_(state_dict["count"].to(self.device))
        self.mean.copy_(state_dict["mean"].to(self.device))
        self.var.copy_(state_dict["var"].to(self.device))
        self.epsilon = float(state_dict.get("epsilon", self.epsilon))
