import torch
import torch.nn as nn
import torch.nn.functional as F

class PINN(nn.Module):
    def __init__(
        self,
        epochs,
        lr,
        n_units, # сколько нейронов на слое
        # n_K=3,
        # output_dim=3,
        S_max = 500, # м2/кг
        t_max=1800, # с
        L_max=10, # м
    ):

        super().__init__()
        
        self.epochs = epochs
        self.lr = lr
        self.n_units = n_units
        # self.n_K=n_K

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
            nn.Linear(self.n_units, 2)
        )

    def forward(self, t, z):
        x = torch.cat([t, z], dim=1) # склеивание по столбцам, мб нужно torch.stack?

        out = self.layers(x)

        S = out[:, 0:1] # 1 столбец данных
        v_m = out[:, 1:2] # 2 столбец

        return S, v_m

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