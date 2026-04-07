import torch
# import torch.nn.functional as F

# def mdn_loss(pi, sigma, mu, S_true):
#     """
#     Calculates the MDN loss (Negative Log Likelihood).
#     pi: Mixing coefficients [Batch, K]
#     sigma: Standard deviations [Batch, K, Output_Dim]
#     mu: Means [Batch, K, Output_Dim]
#     S_true: Ground truth [Batch, Output_Dim]
#     """
#     # Expand target to match the shape of the mixture components
#     S_true = S_true.unsqueeze(1).expand_as(mu)
    
#     # Define the normal distribution
#     m = torch.distributions.Normal(loc=mu, scale=sigma)
    
#     # Calculate log-probability of target under each Gaussian component
#     log_prob = m.log_prob(target) 
#     log_prob = torch.sum(log_prob, dim=2) # Sum across output dimensions
    
#     # Combine with mixing coefficients using logsumexp for stability
#     # log(sum(exp(log_pi + log_prob)))
#     loss = -torch.logsumexp(torch.log(pi) + log_prob, dim=1)
    
#     return torch.mean(loss)

def mdn_loss(pi, mu, sigma, target):
    target = target.expand_as(mu)

    # Gaussian PDF
    coeff = 1.0 / (sigma * torch.sqrt(torch.tensor(2.0 * torch.pi)))
    exponent = torch.exp(-0.5 * ((target - mu) / sigma)**2)

    probs = coeff * exponent

    weighted = pi * probs
    loss = -torch.log(torch.sum(weighted, dim=1) + 1e-8)

    return torch.mean(loss)