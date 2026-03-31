import torch


@torch.no_grad()
def weighted_average(y, weights, dim=0, eps: float = 0.0):
    return (weights * y).sum(dim=dim) / (weights.sum(dim=0) + eps)