import torch
import torch.nn as nn
import torch.nn.functional as F

class PINN(nn.Module):
    def __init__(
        self,
        epochs=10000,
        lr=0.001,
        n_units=100, # сколько нейронов на слое
        S_max = 500, # м2/кг
        t_max=1800, # с
        L_max=10, # м
    ):

        super().__init__()
        
        self.epochs = epochs
        self.lr = lr
        self.n_units = n_units

        self.S_max = S_max
        self.t_max = t_max
        self.L_max = L_max
        self.v_m = nn.Parameter(torch.tensor(1.0))
        self.k = nn.Parameter(torch.tensor(.0))

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

    def loss(self):
        pass
        # TODO

    def forward(self, t, z):
        x = torch.cat([t, z], dim=1) # склеивание по столбцам, мб нужно torch.stack?

        out = self.layers(x)

        S = out[:, 0:1] # 1 столбец данных
        v_m = out[:, 1:2] # 2 столбец

        return S, v_m

    def fit(self):
        pass
        # TODO
    
    def predict(self):
        pass
        # TODO

class PINN_MDN(nn.Module):
    def __init__(
        self, 
        n_components=3,
        n_units=100,
        epochs=10000,
        ):
        super().__init__()

        self.n_components = n_components
        self.n_units=n_units
        self.epochs=epochs

        self.layers = nn.Sequential(
            nn.Linear(2, n_units),
            nn.Tanh(),

            nn.Linear(n_units, n_units),
            nn.Tanh(),

            nn.Linear(n_units, n_units),
            nn.Tanh(),

            nn.Linear(n_units, n_units),
            nn.Tanh(),
        )

        # MDN для S
        self.fc_mu = nn.Linear(100, n_components)
        self.fc_sigma = nn.Linear(100, n_components)
        self.fc_pi = nn.Linear(100, n_components)

    def forward(self, t, z):
        x = torch.cat([t, z], dim=1)
        h = self.layers(x)

        mu = self.fc_mu(h)
        sigma = torch.exp(self.fc_sigma(h))  # > 0
        pi = F.softmax(self.fc_pi(h), dim=1)

        return mu, sigma, pi

    def loss(self):
        pass
        # TODO
    def fit(self):
        pass
        # TODO
        
    def predict(self):
        pass
        # TODO