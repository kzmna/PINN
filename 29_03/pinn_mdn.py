import torch
import torch.nn as nn
import torch.nn.functional as F

class PINN_MDN(nn.Module):
    def __init__(self, n_components=3):
        super().__init__()

        self.n_components = n_components

        self.net = nn.Sequential(
            nn.Linear(2, 100),
            nn.Tanh(),

            nn.Linear(100, 100),
            nn.Tanh(),

            nn.Linear(100, 100),
            nn.Tanh(),

            nn.Linear(100, 100),
            nn.Tanh(),
        )

        # MDN для S
        self.fc_mu = nn.Linear(100, n_components)
        self.fc_sigma = nn.Linear(100, n_components)
        self.fc_pi = nn.Linear(100, n_components)

        # скорость
        self.fc_v = nn.Linear(100, 1)

    def forward(self, t, z):
        x = torch.cat([t, z], dim=1)
        h = self.net(x)

        mu = self.fc_mu(h)
        sigma = torch.exp(self.fc_sigma(h))  # > 0
        pi = F.softmax(self.fc_pi(h), dim=1)

        v_m = self.fc_v(h)

        return mu, sigma, pi, v_m