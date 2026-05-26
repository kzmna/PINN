import torch
import torch.nn.functional as F

def pinn_loss_v2(model, 
            t_pde, z_pde, v_pde, k_pde,
            t_bc1, z_bc1, v_bc1, k_bc1, S_bc1,
            t_bc2, z_bc2, v_bc2, k_bc2, S_bc2, 
            t_ic, z_ic, v_ic, k_ic, S_ic,
            lambda_pde=1.0, 
            lambda_bc1=1.0, 
            lambda_bc2=1.0,
            lambda_ic=1.0,
             ):
        """
        
        Аргументы:
            t_pde, z_pde : коллокационные точки для PDE (тензоры)
            t_bc, z_bc, S_bc : точки на границе (z=0), где задано S(0,t)=S_bc
            t_ic, z_ic, S_ic : начальные условия (t=0)
            lambda_* : веса компонент потерь
        """
        # PDE
        r = model.pde_residual(t_pde, z_pde, v_pde, k_pde)
        loss_pde = torch.mean(r**2)
        
        # Граничное условие на входе (z=0)
        S_pred_bc1 = model.forward(t_bc1, z_bc1, v_bc1, k_bc1)
        loss_bc1 = F.mse_loss(S_pred_bc1, S_bc1)

        # Граничное условие на выходе (z=z_max)
        S_pred_bc2 = model.forward(t_bc2, z_bc2, v_bc2, k_bc2)
        loss_bc2 = F.mse_loss(S_pred_bc2, S_bc2)

        # Начальное условие при t=0
        S_pred_ic = model.forward(t_ic, z_ic, v_ic, k_ic)
        loss_ic = F.mse_loss(S_pred_ic, S_ic)

        total_loss = (
            lambda_pde * loss_pde 
            + lambda_bc1 * loss_bc1 
            + lambda_bc2 * loss_bc2
            + lambda_ic * loss_ic)

        return total_loss, loss_pde.item(), loss_bc1.item(), loss_bc2.item()