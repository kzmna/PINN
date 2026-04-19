import torch

def generate_dataset(
    # == кол-во точек ==
    n_pde=1000,
    n_data=100,
    n_bc=500,
    n_ic=500,
    n_test=100,

    # == константы ==
    S_max=500.0,
    S0=50.0,
    k=0.12,
    v_m=0.2,
    t_max=1800.0,
    z_max=10.0,

    # == параметры ==
    noise_std=10.0,
):
    # -------------------------
    # PDE
    # -------------------------
    t_pde = torch.rand(n_pde, 1) * t_max
    z_pde = torch.rand(n_pde, 1) * z_max

    # -------------------------
    # BC: z = 0
    # -------------------------
    # t_bc = torch.linspace(0, t_max/3+1, n_bc)
    t_bc = torch.rand(n_bc, 1) * t_max
    z_bc = torch.zeros(n_bc, 1)
    S_bc = torch.full((n_bc, 1), S0)

    # -------------------------
    # IC: t = 0
    # -------------------------
    t_ic = torch.zeros(n_ic, 1)
    # z_ic = torch.linspace(0, z_max/3+1, n_ic)
    z_ic = torch.rand(n_ic, 1) * z_max
    S_ic = torch.full((n_ic, 1), S0)

    # -------------------------
    # Данные для data-driven
    # -------------------------
    t_train = torch.rand(n_data, 1) * t_max/2
    z_train = torch.rand(n_data, 1) * z_max/2

    # аналитическое решение
    tau = t_train - z_train / v_m

    S_train = torch.where(
        tau > 0,
        # S_max - (S_max - S0) * torch.exp(-k * tau),
        # torch.full_like(tau, S0),
        S_max - (S_max - S0) * torch.exp(-k * z_train / v_m), # Зависит от Z
        S_max - (S_max - S0) * torch.exp(-k * t_train)        # Зависит от T
    )

    # шум
    noise = noise_std * torch.randn_like(S_train)
    S_train_noisy = S_train + noise

    # -------------------------
    # Данные для тестирования
    # -------------------------
    # t_test = torch.rand(n_test, 1) * t_max
    # z_test = torch.rand(n_test, 1) * z_max

    t_test = torch.linspace(0, t_max, n_test)
    z_test = torch.linspace(0, z_max, n_test)

    T, Z = torch.meshgrid(t_test, z_test, indexing='ij')

    # аналитическое решение
    tau = T - Z / v_m

    S = torch.where(
        tau > 0,
        # S_max - (S_max - S0) * torch.exp(-k * tau),
        # torch.full_like(tau, S0),
        S_max - (S_max - S0) * torch.exp(-k * Z / v_m), # Зависит от Z
        S_max - (S_max - S0) * torch.exp(-k * T)        # Зависит от T
    )

    # шум
    noise = noise_std * torch.randn_like(S)
    S_test_noisy = S + noise

    return {
        # physics
        "t_pde": t_pde,
        "z_pde": z_pde,

        # BC
        "t_bc": t_bc,
        "z_bc": z_bc,
        "S_bc": S_bc,

        # IC
        "t_ic": t_ic,
        "z_ic": z_ic,
        "S_ic": S_ic,

        # data-driven
        "t_train": t_train,
        "z_train": z_train,
        "S_train": S_train_noisy.view(n_data, 1),

        # testing
        "t_test": t_test,
        "z_test": z_test,
        "S_test": S_test_noisy, # тут meshgrid

        # for plots
        "T": T, # тут meshgrid
        "Z": Z, # тут meshgrid
        "S": S, # тут meshgrid

    }