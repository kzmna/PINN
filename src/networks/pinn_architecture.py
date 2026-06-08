import torch
import torch.nn as nn

class PINN_Mill(nn.Module):
    """
    Вход:
    z       - координата
    Fm      - расход твердого
    Fl      - расход жидкости

    Выход:
    eps - вязкость пульпы
    S   - удельная поверхность
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.Tanh(),

            nn.Linear(128, 128),
            nn.Tanh(),

            nn.Linear(128, 128),
            nn.Tanh(),

            nn.Linear(128, 128),
            nn.Tanh(),

            nn.Linear(128, 128),
            nn.Tanh(),

            nn.Linear(128, 2)
        )

    def forward(self, t, z, Fm, Fl):
        x = torch.cat(
            [t, z, Fm, Fl],
            dim=1
        )

        y = self.net(x)

        eps_raw = y[:, 0:1]
        S_raw   = y[:, 1:2]

        # 0 < eps < 1
        eps = torch.sigmoid(eps_raw)

        # 0 < S < 500
        S = 500.0 * torch.sigmoid(S_raw)

        return eps, S