import torch
import torch.nn as nn
import torch.nn.functional as F

def mdn_loss(mu, sigma, y):
    eps = 1e-6
    sigma = sigma.clamp(min=eps)

    return torch.mean(
        0.5 * ((y - mu) / sigma)**2 + torch.log(sigma)
    )