import torch
import torch.nn as nn
import torch.nn.functional as F


class MDN(nn.Module):
    def __init__(self, in_dim=3, n_components=1, hidden_dim=128):
        """
        """
        super().__init__()

        self.n_components = n_components

        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # self.pi = nn.Linear(hidden_dim, n_components)
        self.mu = nn.Linear(hidden_dim, n_components)
        self.log_sigma = nn.Linear(hidden_dim, n_components)

    def forward(self, x):
        """
        x: (batch, in_dim)
        """
        h = self.layers(x)

        # means
        mu = self.mu(h)  # (B, K)

        # std
        log_sigma = self.log_sigma(h)
        sigma = F.softplus(log_sigma) + 1e-4

        return mu.squeeze(-1), sigma.squeeze(-1)

    @torch.no_grad()
    def predict(self, x):
        self.eval()

        mu, sigma = self.forward(x)

        lower = mu - 1.96 * sigma
        upper = mu + 1.96 * sigma

        return mu, lower, upper