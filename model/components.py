import torch
import torch.nn as nn

import utils


class Embedding_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, use_delta=True):
        super().__init__()
        self.use_delta = bool(use_delta)
        multiplier = 3 if self.use_delta else 2
        self.layers = nn.Sequential(
            nn.Linear(in_dim * multiplier, 200),
            nn.ReLU(),
            nn.Linear(200, out_dim),
        )
        utils.init_network_weights(self.layers, method="kaiming_uniform_")

    def forward(self, values, mask, delta=None):
        if self.use_delta:
            delta = torch.zeros_like(values) if delta is None else delta
            inputs = torch.cat([values, mask, delta], dim=-1)
        else:
            inputs = torch.cat([values, mask], dim=-1)
        if torch.isnan(inputs).any():
            raise ValueError("embedding input contains NaN")
        return self.layers(inputs)


class Embedding_MLP_GNN(nn.Module):
    def __init__(self, in_dim, out_dim, use_delta=True):
        super().__init__()
        self.use_delta = bool(use_delta)
        multiplier = 3 if self.use_delta else 2
        self.layers = nn.Sequential(
            nn.Linear(in_dim * multiplier, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )
        utils.init_network_weights(self.layers, method="kaiming_uniform_")

    def forward(self, values, mask, delta=None):
        if self.use_delta:
            delta = torch.zeros_like(values) if delta is None else delta
            inputs = torch.cat([values, mask, delta], dim=-1)
        else:
            inputs = torch.cat([values, mask], dim=-1)
        if torch.isnan(inputs).any():
            raise ValueError("graph embedding input contains NaN")
        return self.layers(inputs)


class Reconst_Mapper_MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 200),
            nn.ReLU(),
            nn.Linear(200, out_dim),
        )
        utils.init_network_weights(self.layers, method="kaiming_uniform_")

    def forward(self, latent):
        return self.layers(latent)


class Z_to_mu_ReLU(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.ReLU(),
            nn.Linear(100, latent_dim),
        )
        utils.init_network_weights(self.net, method="kaiming_uniform_")

    def forward(self, latent):
        return self.net(latent)


class Z_to_std_ReLU(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.ReLU(),
            nn.Linear(100, latent_dim),
            nn.Softplus(),
        )
        utils.init_network_weights(self.net)

    def forward(self, latent):
        return self.net(latent)


class BinaryClassifier(nn.Module):
    def __init__(self, in_dim, args=None):
        super().__init__()
        dropout = float(getattr(args, "classifier_dropout", 0.1))
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )
        utils.init_network_weights(self.layers, method="kaiming_uniform_")

    def forward(self, inputs):
        original_shape = inputs.shape[:-1]
        output = self.layers(inputs.reshape(-1, inputs.shape[-1]))
        return output.reshape(*original_shape, 1)
