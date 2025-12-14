import torch
import numpy as np


def grad(outputs, inputs):
    """Computes the partial derivative of 
    an output with respect to an input.
    Args:
        outputs: (N, 1) tensor
        inputs: (N, D) tensor
    """
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )


def grinding_kinetics(time, Smax, S0, k):
    S = Smax + (S0 - Smax) * np.exp(-k * time)
    return S