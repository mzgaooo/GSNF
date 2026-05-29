import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


PROJ_FOLDER = Path(__file__).parents[0]


def init_network_weights(net, method="normal_"):
    for module in net.modules():
        if isinstance(module, nn.Linear):
            if method == "xavier_uniform_":
                nn.init.xavier_uniform_(module.weight)
            elif method == "kaiming_uniform_":
                nn.init.kaiming_uniform_(module.weight)
            else:
                nn.init.normal_(module.weight, mean=0, std=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, val=0)


def get_device(tensor):
    if tensor.is_cuda:
        return tensor.get_device()
    return torch.device("cpu")


def sample_standard_gaussian(mu, sigma):
    device = get_device(mu)
    epsilon = torch.distributions.normal.Normal(
        torch.tensor([0.0], device=device),
        torch.tensor([1.0], device=device),
    )
    sample = epsilon.sample(mu.size()).squeeze(-1)
    return sample * sigma.float() + mu.float()


def check_mask(data, mask):
    n_zeros = torch.sum(mask == 0.0).cpu().numpy()
    n_ones = torch.sum(mask == 1.0).cpu().numpy()
    assert (n_zeros + n_ones) == np.prod(list(mask.size()))
    assert torch.sum(data[mask == 0.0] != 0.0) == 0


class SolverWrapper_GNN(nn.Module):
    def __init__(self, solver):
        super().__init__()
        self.solver = solver

    def forward(self, x, h, t_start, t_end, adj, backwards=False):
        assert len(x.shape) - len(t_end.shape) == 1
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


def record_experiment(args, model):
    file_result = PROJ_FOLDER / "results" / "result.txt"
    file_result.parent.mkdir(parents=True, exist_ok=True)
    with open(file_result, "a") as fr:
        json.dump(vars(args), fr, indent=0)
        print(model, file=fr)


def calc_mean_std(df):
    mean = df.mean()
    if len(df) == 1 or df.std() == 0:
        std = 1
    else:
        std = df.std()
    return mean, std
