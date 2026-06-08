import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PINN_v3(nn.Module):
    def __init__(
        self,
        S_max = 500,
        n_units=100, # сколько нейронов на слое
        k = 0.0015
    ):

        super().__init__()
        
        self.n_units = n_units
        self.S_max = S_max
        self.log_k = nn.Parameter(torch.tensor(-6.0))
        self.log_D = nn.Parameter(torch.tensor(-7.0))

        # обучаемые параметры
        # self.v_m = nn.Parameter(torch.tensor(0.5))
        # self.k = nn.Parameter(torch.tensor(3e-4))
        self.k = k

        self.layers = nn.Sequential(
            nn.Linear(3, self.n_units),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(self.n_units, 2)
        )

    def forward(self, t, z, v):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if z.dim() == 1:
            z = z.unsqueeze(1)
        if v.dim() == 1:
            v = v.unsqueeze(1)

        x = torch.cat([t, z, v], dim=1) # склеивание по столбцам
        out = self.layers(x)

        S = out[:, 0:1]
        eps = torch.sigmoid(out[:, 1:2])

        return S, eps

    def pde_residual(self, t, z, v):

        t = t.clone().detach().requires_grad_(True)
        z = z.clone().detach().requires_grad_(True)

        S, eps = self.forward(t, z, v)

        k = torch.exp(self.log_k)
        D = torch.exp(self.log_D)

        # ===== S =====

        S_t = torch.autograd.grad(
            S, t,
            grad_outputs=torch.ones_like(S),
            create_graph=True
        )[0]

        S_z = torch.autograd.grad(
            S, z,
            grad_outputs=torch.ones_like(S),
            create_graph=True
        )[0]

        S_zz = torch.autograd.grad(
            S_z, z,
            grad_outputs=torch.ones_like(S_z),
            create_graph=True
        )[0]

        # ===== eps =====

        eps_t = torch.autograd.grad(
            eps, t,
            grad_outputs=torch.ones_like(eps),
            create_graph=True
        )[0]

        eps_z = torch.autograd.grad(
            eps, z,
            grad_outputs=torch.ones_like(eps),
            create_graph=True
        )[0]

        eps_zz = torch.autograd.grad(
            eps_z, z,
            grad_outputs=torch.ones_like(eps_z),
            create_graph=True
        )[0]

        residual_eps = (
            eps_t
            + v * eps_z
            - D * eps_zz
        )

        residual_S = (
            S_t
            + v * S_z
            - D * S_zz
            - k * (self.S_max - S)
        )

        return residual_S, residual_eps
    
    @torch.no_grad()
    def predict(self, t, z, v):
        """
        Аргументы:
            t, z : тензоры или массивы numpy формы (n_points, 1) или (n_points,)

        Возвращает:
            S_pred : тензор на том же устройстве, форма (n_points, 1)
        """
        self.eval()

        # Обеспечиваем размерность
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if z.dim() == 1:
            z = z.unsqueeze(1)
        if v.dim() == 1:
            v = v.unsqueeze(1)

        S = self.forward(t, z, v)

        return S

