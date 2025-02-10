import torch
from torch import nn
import torch.nn.functional as F

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def reprloss(x,y):
    return F.mse_loss(x, y)

def stdloss(x):
    x = x - x.mean(dim=0)
    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    return torch.mean(F.relu(1 - std_x)) / 2

def covloss(x, batch_size, num_features):
    x = x - x.mean(dim=0)
    cov_x = (x.T @ x) / (batch_size - 1)
    return off_diagonal(cov_x).pow_(2).sum().div(num_features)

def vicreg_loss(x, y, batch_size = 32, sim_coeff = 25.0, std_coeff = 25.0, cov_coeff = 1.0, num_features = 768):
        # Make sure the context embedding is x and target embedding is y
        #repr_loss = reprloss(x, y)
        std_loss = stdloss(x)
        cov_loss = covloss(x, batch_size, num_features)
        loss = (
            std_coeff * std_loss
            + cov_coeff * cov_loss
        )
        return loss