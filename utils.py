import numpy as np
import torch
import torch.nn as nn


def init_network_weights(net, method="normal_"):
    for module in net.modules():
        if isinstance(module, nn.Linear):
            if method == "xavier_uniform_":
                nn.init.xavier_uniform_(module.weight)
            elif method == "kaiming_uniform_":
                nn.init.kaiming_uniform_(module.weight)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)


def get_device(tensor):
    return tensor.device if tensor.is_cuda else torch.device("cpu")


def sample_standard_gaussian(mu, sigma):
    return mu + torch.randn_like(mu) * sigma.float()


def check_mask(data, mask):
    if not torch.all((mask == 0.0) | (mask == 1.0)):
        raise ValueError("observation masks must be binary")
    if torch.any(data[mask == 0.0] != 0.0):
        raise ValueError("unobserved values must be zero")


class SolverWrapper_GNN(nn.Module):
    def __init__(self, solver):
        super().__init__()
        self.solver = solver

    def forward(self, x, h, t_start, t_end, adj, backwards=False):
        if len(x.shape) - len(t_end.shape) != 1:
            raise ValueError("flow state and time dimensions are incompatible")
        t_start = t_start.unsqueeze(-1)
        t_end = t_end.unsqueeze(-1)
        if t_end.shape[-3] == 1:
            t_start = t_start.repeat_interleave(x.shape[-3], dim=-3)
            t_end = t_end.repeat_interleave(x.shape[-3], dim=-3)
        if len(x.shape) == 4 and x.shape[0] != t_end.shape[0]:
            t_start = t_start.repeat_interleave(x.shape[0], dim=0)
            t_end = t_end.repeat_interleave(x.shape[0], dim=0)
        nodes = getattr(self.solver, "nodes", x.shape[-1])
        return self.solver(nodes, x, h, t_start, t_end, adj)


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
