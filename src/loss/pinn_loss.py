import torch
import torch.nn.functional as F

def pinn_loss(model, 
            t_pde, z_pde, 
            t_bc, z_bc, S_bc, 
            t_ic, z_ic, S_ic, 
            t_real=None, z_real=None, S_real=None,
            lambda_pde=1.0, 
            lambda_bc=1.0, 
            lambda_ic=1.0,
            lambda_data=1.0
             ):
        """
        
        Аргументы:
            t_pde, z_pde : коллокационные точки для PDE (тензоры)
            t_bc, z_bc, S_bc : точки на границе (z=0), где задано S(0,t)=S_bc
            t_ic, z_ic, S_ic : начальные условия (t=0)
            lambda_* : веса компонент потерь
        """
        # PDE
        r = model.pde_residual(t_pde, z_pde)
        loss_pde = torch.mean(r**2)
        
        # Граничное условие на входе (z=0)
        S_pred_bc = model.forward(t_bc, z_bc)
        loss_bc = F.mse_loss(S_pred_bc, S_bc)

        # Начальное условие (t=0)
        S_pred_ic = model.forward(t_ic, z_ic)
        loss_ic = F.mse_loss(S_pred_ic, S_ic)

        # Потери по данным
        loss_data = torch.tensor(0.0, device=t_pde.device)
        if t_real is not None and z_real is not None and S_real is not None:
            S_pred_real = model.forward(t_real, z_real)
            loss_data = F.mse_loss(S_pred_real, S_real)

        total_loss = (
            lambda_pde * loss_pde 
            + lambda_bc * loss_bc 
            + lambda_ic * loss_ic 
            + lambda_data * loss_data)

        return total_loss