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
        self.v_m = nn.Parameter(torch.tensor(1.0))
        self.k = nn.Parameter(torch.tensor(0.0))

        self.layers = nn.Sequential(
            nn.Linear(2, self.n_units),
            nn.Tanh(),
            nn.Linear(self.n_units, self.n_units),
            nn.Tanh(),
            nn.Linear(self.n_units, self.n_units),
            nn.Tanh(),
            nn.Linear(self.n_units, self.n_units),
            nn.Tanh(),
            nn.Linear(self.n_units, 1)
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
                                   create_graph=True)[0]
        S_z = torch.autograd.grad(S, z, grad_outputs=torch.ones_like(S),
                                   create_graph=True)[0]

        residual = S_t + self.v_m * S_z - self.k * (self.S_max - S)

        return residual


    def loss(self, t_pde, z_pde, t_bc, z_bc, S_bc, t_ic, z_ic, S_ic,
             lambda_pde=1.0, lambda_bc=1.0, lambda_ic=1.0):
        """Вычисляет полную функцию потерь.
        
        Аргументы:
            t_pde, z_pde : коллокационные точки для PDE (тензоры)
            t_bc, z_bc, S_bc : точки на границе (z=0), где задано S(0,t)=S_bc
            t_ic, z_ic, S_ic : начальные условия (t=0)
            lambda_* : веса компонент потерь
        """
        # 1. Потеря PDE
        r = self.pde_residual(t_pde, z_pde)#
        loss_pde = torch.mean(r**2)

        # 2. Граничное условие на входе (z=0)
        S_pred_bc = self.forward(t_bc, z_bc)
        loss_bc = F.mse_loss(S_pred_bc, S_bc)

        # 3. Начальное условие (t=0)
        S_pred_ic = self.forward(t_ic, z_ic)
        loss_ic = F.mse_loss(S_pred_ic, S_ic)

        total_loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic

        return total_loss
     

    def forward(self, t, z):

        x = torch.cat([t, z], dim=1) # склеивание по столбцам, мб нужно torch.stack?

        out = self.layers(x)

        S = out[:, 0:1] # 1 столбец данных
        # v_m = out[:, 1:2] # 2 столбец

        return S

    def fit(self, t_pde, z_pde, t_bc, z_bc, S_bc, t_ic, z_ic, S_ic,
            epochs=None, lr=None, 
            lambda_pde=1.0, lambda_bc=1.0, lambda_ic=1.0, log_freq=100):
        """
            Аргументы:
            t_pde, z_pde : тензоры коллокационных точек для PDE (n_pde, 1)
            t_bc, z_bc, S_bc : граничные условия (n_bc, 1)
            t_ic, z_ic, S_ic : начальные условия (n_ic, 1)
            epochs : количество эпох
            lr : скорость обучения
            lambda_* : веса компонент потерь
            log_freq : частота печати
        """
        if epochs == None and lr == None:
            epochs = self.epochs
            lr = self.lr
        optimizer = optim.Adam(self.parameters(), lr=lr)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.5)

        best_loss = float('inf')
        loss_history = []
        

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            total_loss = self.loss(
                t_pde, z_pde, t_bc, z_bc, S_bc, t_ic, z_ic, S_ic,
                lambda_pde, lambda_bc, lambda_ic
            )
            total_loss.backward()
            optimizer.step()
            loss_history.append(total_loss.item())
            # scheduler.step(total_loss)
            if epoch % log_freq == 0 or epoch == epochs-1:
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {total_loss.item():.6f} ")

        return loss_history
    
    def predict(self, t, z):
        """
        Аргументы:
            t, z : тензоры или массивы numpy формы (n_points, 1) или (n_points,)
            batch_size : размер батча для предсказания (чтобы не перегружать память)

        Возвращает:
            S_pred : тензор на том же устройстве, форма (n_points, 1)
        """
        self.eval()

        # Приводим к тензорам, если на вход подали numpy
        # if isinstance(t, np.ndarray):
        #     t = torch.from_numpy(t).float()
        # if isinstance(z, np.ndarray):
        #     z = torch.from_numpy(z).float()

        # Обеспечиваем размерность (n_points, 1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if z.dim() == 1:
            z = z.unsqueeze(1)

        n_points = t.size(0)

        with torch.no_grad():
            S = self.forward(t, z)

        return S

class PINN_MDN(nn.Module):
    def __init__(
        self, 
        n_components=3,
        n_units=100,
        epochs=10000,
        lr=1e-3,
        t_max=1800,
        L_max=10,
        S_max=500,
        device='cpu'
        ):
        super().__init__()

        self.n_components = n_components
        self.n_units=n_units
        self.epochs=epochs
        self.lr = lr
        self.t_max = t_max
        self.L_max = L_max
        self.S_max = S_max
        self.device = device

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

    def loss(self, t, z, S_true):
        """
        Вычисляет отрицательное логарифмическое правдоподобие (NLL) для смеси гауссиан.
        
        Аргументы:
            t, z: входные координаты (batch, 1)
            S_true: истинные значения удельной поверхности (batch, 1)
        
        Возвращает:
            nll: среднее отрицательное логарифмическое правдоподобие
        """

        mu, sigma, pi = self.forward(t, z)
        
        # Расширяем S_true для broadcasting
        S_true_expanded = S_true.unsqueeze(1)  # (batch, 1, 1)
        
        # Вычисляем логарифм плотности для каждой компоненты
        # N(S_true | mu, sigma^2)
        log_prob = -0.5 * ((S_true - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * np.log(2 * np.pi)
        
        # Логарифм смеси: log( sum_i pi_i * exp(log_prob_i) )
        log_pi = torch.log(pi + 1e-8)
        log_mixture = torch.logsumexp(log_pi + log_prob, dim=1)  # (batch,)
        
        # Отрицательное логарифмическое правдоподобие
        nll = -log_mixture.mean()
        
        return nll

    def fit(self, t_train, z_train, S_train, epochs=None, lr=None, 
            verbose=True, log_freq=100):
        """
        Обучение MDN.
        
        Аргументы:
            t_train, z_train, S_train: обучающие данные
            epochs: количество эпох (если None, используется self.epochs)
            lr: скорость обучения (если None, используется self.lr)
            log_freq: частота печати
            patience: ранняя остановка
        """
        if epochs is None:
            epochs = self.epochs
        if lr is None:
            lr = self.lr
        
        # Перемещаем данные на устройство
        t_train = t_train.to(self.device)
        z_train = z_train.to(self.device)
        S_train = S_train.to(self.device)
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
        
        best_loss = float('inf')
        patience_counter = 0
        loss_history = {'train': [], 'val': []}
        
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            num_batches = 0
        
            loss = self.loss(t_train, z_train, S_train)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            train_loss = epoch_loss
            loss_history['train'].append(train_loss)
            
            if verbose and (epoch % log_freq == 0 or epoch == epochs-1):
                print(f"Epoch {epoch:4d}/{epochs} | Train Loss: {train_loss:.6f}{val_str}")

            return loss_history
        
    def predict(self, t, z):
        """
        Предсказывает удельную поверхность S с доверительными интервалами.
        
        Аргументы:
            t, z: координаты (numpy или torch)
        
        Возвращает:
            (mu, sigma, pi) если return_distribution=True
        """
        self.eval()
        
        # Приводим к тензорам
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()
        
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if z.dim() == 1:
            z = z.unsqueeze(1)
        
        t = t.to(self.device)
        z = z.to(self.device)
        n_points = t.size(0)
        
        all_mu = []
        all_sigma = []
        all_pi = []
        
        with torch.no_grad():
            for i in range(0, n_points, batch_size):
                t_batch = t[i:i+batch_size]
                z_batch = z[i:i+batch_size]
                mu, sigma, pi = self.forward(t_batch, z_batch)
                all_mu.append(mu.cpu())
                all_sigma.append(sigma.cpu())
                all_pi.append(pi.cpu())
        
        mu = torch.cat(all_mu, dim=0)
        sigma = torch.cat(all_sigma, dim=0)
        pi = torch.cat(all_pi, dim=0)

        return mu, sigma, pi