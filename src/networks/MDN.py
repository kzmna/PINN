import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MDN(nn.Module):
    def __init__(self, n_components=3):
        super().__init__()

        hidden_dim = 100
        self.n_components = n_components

        self.layers = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.pi = nn.Linear(hidden_dim, n_components)
        self.mu = nn.Linear(hidden_dim, n_components)
        self.sigma = nn.Linear(hidden_dim, n_components)

    def forward(self, s_pred):
        h = self.layers(s_pred)

        pi = F.softmax(self.pi(h), dim=-1)
        mu = self.mu(h)
        sigma = torch.exp(self.sigma(h))

        # sigma = torch.clamp(sigma, min=1e-4, max=10)
        sigma = F.softplus(self.sigma(h)) + 1e-6

        return pi, mu, sigma

    @torch.no_grad()
    def predict(self, S_pred):
        pi, mu, sigma = self.forward(S_pred)

        mean = torch.sum(pi * mu, dim=1)
        var = torch.sum(pi * (sigma**2 + mu**2), dim=1) - mean**2

        std = torch.sqrt(var)

        lower = mean - 1.96 * std
        upper = mean + 1.96 * std

        return mean, lower, upper