import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PINN_S(nn.Module):
    def __init__(
        self,
        S_max=500,
        n_units=100
    ):
        super().__init__()

        self.S_max = S_max

        # обучаемые физические параметры
        self.log_k = nn.Parameter(torch.tensor(-6.0))
        self.log_D = nn.Parameter(torch.tensor(-7.0))

        self.layers = nn.Sequential(
            nn.Linear(3, n_units),
            nn.Tanh(),

            nn.Linear(n_units, n_units),
            nn.Tanh(),

            nn.Linear(n_units, n_units),
            nn.Tanh(),

            nn.Linear(n_units, n_units),
            nn.Tanh(),

            nn.Linear(n_units, 1)
        )

    def forward(self, t, z, v):

        if t.dim() == 1:
            t = t.unsqueeze(1)

        if z.dim() == 1:
            z = z.unsqueeze(1)

        if v.dim() == 1:
            v = v.unsqueeze(1)

        x = torch.cat([t, z, v], dim=1)

        S = self.layers(x)

        return S
    
    def pde_residual(self, t, z, v):

        t = t.clone().detach().requires_grad_(True)
        z = z.clone().detach().requires_grad_(True)

        S = self.forward(t, z, v)

        k = torch.exp(self.log_k)
        D = torch.exp(self.log_D)

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

        residual = (
            S_t
            + v * S_z
            - D * S_zz
            - k * (self.S_max - S)
        )

        return residual

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