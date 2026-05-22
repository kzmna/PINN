import torch
import numpy as np

def generate_dataset(
    # == кол-во точек ==
    n_pde=1000,
    n_test=30,

    # == константы ==
    S_max=750,
    S0=0.06,
    # k=0.87,
    # v_m=0.01,
    # t_max=408.0,
    z_max=4.5,
    Sbc=1.,
):






















    # -------------------------
    # PDE
    # -------------------------
    t_pde = torch.rand(n_pde, 1) * t_max
    z_pde = torch.rand(n_pde, 1) * z_max

    # -------------------------
    # BC: z = 0
    # -------------------------
    t_bc = torch.rand(n_bc, 1) * t_max
    z_bc = torch.zeros(n_bc, 1)
    S_bc = torch.full((n_bc, 1), Sbc)

    # -------------------------
    # IC: t = 0
    # -------------------------
    t_ic = torch.zeros(n_ic, 1)
    z_ic = torch.rand(n_ic, 1) * z_max
    S_ic = torch.full((n_ic, 1), S0)

    # -------------------------
    # Данные для data-driven
    # -------------------------
    t_train = torch.rand(n_data, 1) * t_max/3
    z_train = torch.rand(n_data, 1) * z_max/3


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

    }