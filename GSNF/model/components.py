import torch
import torch.nn as nn

import utils


class Embedding_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, use_delta=False):
        super().__init__()
        self.use_delta = use_delta
        input_multiplier = 3 if use_delta else 2

        self.layers = nn.Sequential(
            nn.Linear(in_dim * input_multiplier, 200),
            nn.ReLU(),
            nn.Linear(200, out_dim)
        )
        utils.init_network_weights(self.layers, method='kaiming_uniform_')

    def forward(self, truth, mask, delta=None): # [B, T, D]
        if self.use_delta:
            if delta is None:
                delta = torch.zeros_like(truth)
            x = torch.cat((truth, mask, delta), -1)
        else:
            x = torch.cat((truth, mask), -1)
        assert (not torch.isnan(x).any())
        out = self.layers(x)
        return out

class Embedding_MLP_GNN(nn.Module):
    def __init__(self, in_dim, out_dim, use_delta=False):
        super().__init__()
        self.use_delta = use_delta
        input_multiplier = 3 if use_delta else 2
        self.layers = nn.Sequential(
            nn.Linear(in_dim * input_multiplier, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )
        utils.init_network_weights(self.layers, method='kaiming_uniform_')

    def forward(self, truth, mask, delta=None):
        if self.use_delta:
            if delta is None:
                delta = torch.zeros_like(truth)
            x = torch.cat((truth, mask, delta), -1)
        else:
            x = torch.cat((truth, mask), -1)
        assert (not torch.isnan(x).any())
        out = self.layers(x)
        return out

class Reconst_Mapper_MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 200),
            nn.ReLU(),
            nn.Linear(200, out_dim)
        )
        utils.init_network_weights(self.layers, method='kaiming_uniform_')

    def forward(self, data):
        truth = self.layers(data)
        return truth


class Z_to_mu_ReLU(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.ReLU(),
            nn.Linear(100, latent_dim),)
        utils.init_network_weights(self.net, method='kaiming_uniform_')

    def forward(self, data):
        return self.net(data)


class Z_to_std_ReLU(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.ReLU(),
            nn.Linear(100, latent_dim),
            nn.Softplus(),)
        utils.init_network_weights(self.net)

    def forward(self, data):
        return self.net(data)


class BinaryClassifier(nn.Module):
    def __init__(self, in_dim, args=None):
        super().__init__()
        head = getattr(args, "classifier_head", "gsnf") if args is not None else "gsnf"
        dropout = getattr(args, "classifier_dropout", 0.0) if args is not None else 0.0
        hidden_override = getattr(args, "classifier_hidden_dim", 0) if args is not None else 0
        grad_scale = getattr(args, "classifier_grad_scale", -1.0) if args is not None else -1.0
        if grad_scale < 0.0:
            grad_scale = 100.0 if head == "deep" else 1.0

        if head == "gsnf":
            hidden_dim = hidden_override or 512
            layers = [
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            ]
        elif head == "deep":
            hidden_dim = hidden_override or 300
            layers = [
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            ]
        elif head == "norm":
            hidden_dim = hidden_override or in_dim
            layers = [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            ]
        elif head == "bn_mlp":
            hidden_dim = hidden_override or 128
            layers = [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            ]
        elif head == "small":
            hidden_dim = hidden_override or 200
            layers = [
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            ]
        elif head == "linear":
            layers = [nn.Linear(in_dim, 1)]
        else:
            raise ValueError(f"Unknown classifier_head: {head}")

        self.layers = nn.Sequential(
            *layers
        )
        utils.init_network_weights(self.layers, method='kaiming_uniform_')
        if grad_scale != 1.0:
            last_linear = next(
                (module for module in reversed(self.layers) if isinstance(module, nn.Linear)),
                None)
            if last_linear is not None:
                last_linear.weight.register_hook(lambda grad: grad_scale * grad)
                if last_linear.bias is not None:
                    last_linear.bias.register_hook(lambda grad: grad_scale * grad)

    def forward(self, x):
        original_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        out = self.layers(x)
        return out.reshape(*original_shape, -1)

