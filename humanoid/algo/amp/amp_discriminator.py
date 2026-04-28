import torch
import torch.nn as nn


class AMPDiscriminator(nn.Module):
    def __init__(self, amp_obs_dim, hidden_dims=None, activation=nn.ReLU()):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers = []
        input_dim = amp_obs_dim * 2
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, amp_obs, next_amp_obs):
        features = torch.cat((amp_obs, next_amp_obs), dim=-1)
        return self.model(features)
