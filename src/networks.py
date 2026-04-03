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

    def pde_residual(self, t, z):
        """
        Вычисляет невязку уравнения:
            ∂S/∂t + v_m * ∂S/∂z - k*(S_max - S) = 0
        Использует autograd для вычисления производных.
        """
        t.requires_grad_(True)
        z.requires_grad_(True)
        S = self.forward(t, z)

        # Первые производные S по t и z
        S_t = torch.autograd.grad(S, t, grad_outputs=torch.ones_like(S),
                                   create_graph=True, retain_graph=True)[0]
        S_z = torch.autograd.grad(S, z, grad_outputs=torch.ones_like(S),
                                   create_graph=True, retain_graph=True)[0]

        residual = S_t + self.v_m * S_z - self.k * (self.S_max - S)
        return residual


    def loss(self, t_pde, z_pde, t_bc, z_bc, S_bc, t_ic, z_ic, S_ic,
             lambda_pde=1.0, lambda_bc=1.0, lambda_ic=1.0):
             """
        Вычисляет полную функцию потерь.
        
        Аргументы:
            t_pde, z_pde : коллокационные точки для PDE (тензоры)
            t_bc, z_bc, S_bc : точки на границе (z=0), где задано S(0,t)=S_bc
            t_ic, z_ic, S_ic : начальные условия (t=0)
            lambda_* : веса компонент потерь
        """
        # 1. Потеря PDE
        r = self.pde_residual(t_pde, z_pde)
        loss_pde = torch.mean(r**2)

        # 2. Граничное условие на входе (z=0)
        S_pred_bc, _ = self.forward(t_bc, z_bc)
        loss_bc = F.mse_loss(S_pred_bc, S_bc)

        # 3. Начальное условие (t=0)
        S_pred_ic, _ = self.forward(t_ic, z_ic)
        loss_ic = F.mse_loss(S_pred_ic, S_ic)

        total_loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic

        return total_loss
     

    def forward(self, t, z):

        t_norm = t / self.t_max
        z_norm = z / self.L_max

        x = torch.cat([t_norm, z_norm], dim=1) # склеивание по столбцам, мб нужно torch.stack?

        out = self.layers(x)

        S = out[:, 0:1] # 1 столбец данных
        # v_m = out[:, 1:2] # 2 столбец

        return S

    def fit(self):
        
        pass
        # TODO: дописать метод
    
    def predict(self):
        pass
        # TODO: дописать метод

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