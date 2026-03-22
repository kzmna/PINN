import torch
import torch.nn as nn

class MDN_PINN(nn.Module):
    def __init__(
        self,
        epochs,
        lr,
        n_units, # сколько нейронов на слое
        n_K=3,
        # output_dim=3,
        S_max = 500, # м2/кг
        t_max=1800, # с
        L_max=10, # м
    ):

        super().__init__()
        
        self.epochs = epochs
        self.lr = lr
        self.n_units = n_units
        self.n_K=n_K

        # self.output_dim = output_dim
        self.S_max = S_max
        self.t_max = t_max
        self.L_max = L_max
        self.v_m = nn.Parameter(torch.tensor(1.0))

        self.layers = nn.Sequential(
            nn.Linear(2, self.n_units),
            nn.Tanh(),
            nn.Linear(self.n_units, self.n_units),
            nn.Tanh(),
            nn.Linear(self.n_units, self.n_units),
            nn.Tanh(),
            nn.Linear(self.n_units, self.n_units),
            nn.Tanh(),
        )
        self.pi = nn.Linear(n_units, n_K)
        self.mu = nn.Linear(n_units, n_K)
        self.sifma = nn.Linear(n_units, n_K)

def forward(self, t, z):
    x = torch.cat([t, z], dim=1)

    h = self.layers(x)

    pi = torch.softmax(self.pi(h), dim=1)
    mu = self.mu
    sigma = torch.exp(self.sigma(h))

    return pi, mu, sigma