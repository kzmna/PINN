import torch

def generate_dataset(
    n_pde=10,
    n_bc=10,
    n_ic=10,
    S_max=500.0,
    S0=50.0,
    k=0.7,
    v_m=0.16,
    t_max=1800.0,
    z_max=10.0,
    noise_std=10.0
):
    # -------------------------
    # PDE точки (внутри области)
    # -------------------------
    t_pde = torch.rand(n_pde, 1) * t_max
    z_pde = torch.rand(n_pde, 1) * z_max

    # -------------------------
    # BC: z = 0
    # -------------------------
    t_bc = torch.rand(n_bc, 1) * t_max
    z_bc = torch.zeros(n_bc, 1)
    S_bc = torch.full((n_bc, 1), S0)

    # -------------------------
    # IC: t = 0
    # -------------------------
    t_ic = torch.zeros(n_ic, 1)
    z_ic = torch.rand(n_ic, 1) * z_max
    S_ic = torch.full((n_ic, 1), S0)

    # -------------------------
    # Данные
    # -------------------------
    t_data = torch.rand(n_pde, 1) * t_max
    z_data = torch.rand(n_pde, 1) * z_max

    # аналитическое решение
    tau = t_data - z_data / v_m

    S_true = torch.where(
        tau > 0,
        S_max - (S_max - S0) * torch.exp(-k * tau),
        torch.full_like(tau, S0)
    )

    # шум
    noise = noise_std * torch.randn_like(S_true)
    S_true_noisy = S_true + noise

    return {
        "t_pde": t_pde,
        "z_pde": z_pde,
        "t_bc": t_bc,
        "z_bc": z_bc,
        "S_bc": S_bc,
        "t_ic": t_ic,
        "z_ic": z_ic,
        "S_ic": S_ic,
        "t_data": t_data,
        "z_data": z_data,
        "S_true": S_true_noisy
    }