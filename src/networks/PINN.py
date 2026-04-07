import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

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

        # обучаемые параметры
        # self.v_m = nn.Parameter(torch.tensor(0.0))
        self.v_m = torch.tensor(0.7)
        # self.k = nn.Parameter(torch.tensor(0.0))
        self.k = torch.tensor(0.16)

        # TODO: либо интегрировать константы, либо добавить MSE
        # TODO: нарисовать график разброса предсказаний PINN

        self.layers = nn.Sequential(
            nn.Linear(2, self.n_units),
            nn.Tanh(),
            #nn.Linear(self.n_units, self.n_units),
            #nn.Tanh(),
            #nn.Linear(self.n_units, self.n_units),
            #nn.Tanh(),
            #nn.Linear(self.n_units, self.n_units),
            #nn.Tanh(),
            nn.Linear(self.n_units, 1)
        )

    def forward(self, t, z):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if z.dim() == 1:
            z = z.unsqueeze(1)

        x = torch.cat([t, z], dim=1) # склеивание по столбцам, мб нужно torch.stack?
        S = self.layers(x)

        return S

    def pde_residual(self, t, z):
        """
            ∂S/∂t + v_m * ∂S/∂z - k*(S_max - S) = 0
        """
        t.requires_grad_(True)
        z.requires_grad_(True)

        S = self.forward(t, z)

        # Первые производные S по t и z
        S_t = torch.autograd.grad(S, t, grad_outputs=torch.ones_like(S),
                                   create_graph=True)[0]
        S_z = torch.autograd.grad(S, z, grad_outputs=torch.ones_like(S),
                                   create_graph=True)[0]

        residual = S_t + self.v_m * S_z - self.k * (self.S_max - S)

        return residual
    
    @torch.no_grad()
    def predict(self, t, z):
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

        S = self.forward(t, z)

        return S

