import torch
import numpy as np


# def grad(outputs, inputs):
#     """Computes the partial derivative of 
#     an output with respect to an input.
#     Args:
#         outputs: (N, 1) tensor
#         inputs: (N, D) tensor
#     """
#     return torch.autograd.grad(
#         outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
#     )


def grinding_kinetics(time, Smax, S0, k):
    S = Smax + (S0 - Smax) * np.exp(-k * time)
    return S

def grinding_kinetics_2d(z_points, t_points, Smax, S0, k, L_m=10.0, T_max=1800):
    """
    Генерация данных для уравнения кинетики измельчения в зависимости от z и t.
    
    Параметры:
    z_points: массив координат z [м]
    t_points: массив времени t [с]
    Smax: предельная удельная поверхность [м^2/кг]
    S0: начальная удельная поверхность на входе [м^2/кг]
    k: константа скорости измельчения [1/с]
    L_m: длина мельницы [м]
    T_max: максимальное время [с]
    
    Возвращает:
    S: массив удельной поверхности S(z,t)
    v_m: массив скоростей материала
    """
    # Создаем сетку
    Z, T = np.meshgrid(z_points, t_points, indexing='ij')
    S = np.zeros_like(Z)
    v_m = np.zeros_like(Z)
    
    v_m[:, :] = 0.01  # постоянная скорость 0.01 м/с
    
    # Решаем уравнение вдоль характеристик
    for i in range(len(z_points)):
        for j in range(len(t_points)):
            z = z_points[i]
            t = t_points[j]
            
            # Время движения от входа до точки z
            # Интегрируем 1/v_m вдоль пути
            if z > 0:
                # Делим путь на отрезки для численного интегрирования
                z_segments = np.linspace(0, z, 100)
                dt_dz = 1.0 / np.interp(z_segments, z_points, v_m[:, j])
                travel_time = np.trapz(dt_dz, z_segments)
            else:
                travel_time = 0
            
            # Время входа материала
            t_entry = t - travel_time
            
            if t_entry < 0:
                # Материал был в мельнице с самого начала
                # Начальное условие: S = S0 везде
                S_entry = S0
            else:
                # Материал вошел в мельницу в момент t_entry
                S_entry = S0  # или можно задать функцию S_in(t)
            
            # Интеграл вдоль характеристики
            # ∫k dz/v_m от 0 до z
            if z > 0:
                # Численное интегрирование k/v_m
                integrand = k / np.interp(z_segments, z_points, v_m[:, j])
                integral = np.trapezoid(integrand, z_segments)
            else:
                integral = 0
            
            # Решение вдоль характеристики
            S[i, j] = Smax - (Smax - S_entry) * np.exp(-integral)
    
    return S, v_m