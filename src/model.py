from math import pi

import torch
import torch.nn as nn
import torch.nn.functional as F

activations_dict = {'relu': F.relu}

class Activation(nn.Module):
    def __init__(self, activation: str) -> None:
        super().__init__()
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return activations_dict.get(self.activation, lambda x: x)(x)


class MLPBlock(nn.Module):
    def __init__(self, input_features: int, output_features: int, activation: str) -> None:
        super().__init__()
        self.dense_layer = nn.Linear(input_features, output_features)
        self.activation = Activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dense_layer(x)
        return self.activation(y)


class MLP(nn.Module):
    def __init__(self, input_features: int, output_features: int, width: int, depth: int, activation: str) -> None:
        super().__init__()
        self.input_features = input_features
        self.model = nn.Sequential(
            MLPBlock(input_features + 1, width, activation),
            *[MLPBlock(width, width, activation) for _ in range(depth)],
            MLPBlock(width, output_features, None)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_features)
        t = t.reshape(-1, 1).expand(x.shape[0], 1).float()
        h = torch.cat([x, t], dim=1)
        return self.model(h)


class InvariantModel(nn.Module):
    '''
        Wrapper for a model that makes it invariant under a group action.
        Only implemented for Sp(2) / U(1).
    '''
    def __init__(self, model: nn.Module, n_samples: int = 100) -> None:
        self.n_samples = n_samples
        self.model = model

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        '''
            X_inv(p) = \int_{U(1)} Ad_{g^{-1}} (X(g.p)) d mu (g)
            d mu = Haar measure = d theta / 2 pi
        '''
        for _ in range(n_samples):
            theta = 2 * pi * torch.rand(out.shape[0])
            sin = torch.sin(theta).unsqueeze(-1)
            cos = torch.cos(theta).unsqueeze(-1)



if __name__ == '__main__':

    torch.manual_seed(43)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MLP(2, 2, width=64, depth=2, activation='relu')
    bs = 10

    x = torch.rand((bs, 2), device=device)
    t = torch.randint(high=2, size=(bs,), device=device)
    print(model(x, t))





