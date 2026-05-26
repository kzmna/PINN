import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PINN_k(nn.Module):
    def __init__(
        self,
        S_max = 750,
        n_units=100, # сколько нейронов на слое
    ):

        super().__init__()
        
        self.n_units = n_units
        self.S_max = S_max

        # обучаемые параметры
        # self.v_m = nn.Parameter(torch.tensor(0.5))
        # self.k = nn.Parameter(torch.tensor(3e-4))

        self.layers = nn.Sequential(
            nn.Linear(4, self.n_units),
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
            nn.Linear(self.n_units, 1)
        )

    def forward(self, t, z, v, k):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if z.dim() == 1:
            z = z.unsqueeze(1)
        if v.dim() == 1:
            v = v.unsqueeze(1)
        if k.dim() == 1:
            k = k.unsqueeze(1)

        x = torch.cat([t, z, v, k], dim=1) # склеивание по столбцам
        S = self.layers(x)

        return S

    def pde_residual(self, t, z, v, k):
        """
            ∂S/∂t + v * ∂S/∂z - k*(S_max - S) = 0
        """
        t = t.clone().detach().requires_grad_(True)
        z = z.clone().detach().requires_grad_(True)

        S = self.forward(t, z, v, k)

        # Первые производные S по t и z
        S_t = torch.autograd.grad(S, t, grad_outputs=torch.ones_like(S),
                                   create_graph=True)[0]
        S_z = torch.autograd.grad(S, z, grad_outputs=torch.ones_like(S),
                                   create_graph=True)[0]
        residual = 1/29406750.0 * S_t  + v * 1/4.5 * S_z - k * (self.S_max - S)

        return residual
    
    @torch.no_grad()
    def predict(self, t, z, v, k):
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
        if k.dim() == 1:
            k = k.unsqueeze(1)

        S = self.forward(t, z, v, k)

        return S

