import torch
import numpy as np

def sort_rank(a):
    sort = torch.tensor(a.cpu().numpy().argsort())
    ranks = torch.empty_like(a)
    ranks[sort] = torch.arange(len(a)).float()
    return ranks

def LCC(y, y_bar):
    y = y.squeeze()
    y_bar = y_bar.squeeze()
    assert y.dim() == 1 and y_bar.dim() == 1
    y_mean = y.mean()
    y_bar_mean = y_bar.mean()
    return torch.sum(\
        (y-y_mean)*(y_bar-y_bar_mean))/\
        torch.sqrt(torch.sum((y-y_mean)**2)*torch.sum((y_bar-y_bar_mean)**2))

def SROCC(y, y_bar):
    y = y.squeeze()
    y_bar = y_bar.squeeze()
    assert y.dim() == 1 and y_bar.dim() == 1
    N = y.shape[0]
    assert N == y_bar.shape[0]
    y_rank = sort_rank(y)
    y_bar_rank = sort_rank(y_bar)
    return 1 - 6*torch.sum((y_rank-y_bar_rank)**2)/(N*(N**2 - 1))
