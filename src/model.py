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
        return self.model(h).reshape(*sz)


class InvariantModel(nn.Module):
    '''
        Wrapper for a model that makes it invariant under a group action.
        Only implemented for Sp(2) / U(1).

        Warning: the model is averaged every time forward is called.
    '''
    def __init__(self, model: nn.Module, n_samples: int = 100, device: str = 'cpu') -> None:
        super().__init__()
        self.n_samples = n_samples
        self.model = model
        self.device = device

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        '''
            X_inv(p) = \int_{U(1)} Ad_{g^{-1}} (X(g.p)) d mu (g)
            d mu = Haar measure = d theta / 2 pi
            This is achieved by sampling elements from U(1) and taking a discrete average.

            Warning: the current implementation required the timestamp tensor t to have
                     shape [batch_size, n_samples]. This has to be manually done by the user
                     by sampling batch_size timestamps and expanding them before calling forward.
        '''
        def gen_group_elements(n_samples: int, device: str) -> torch.Tensor:
            theta = 2 * pi * torch.rand(n_samples, device=self.device)
            sin = torch.sin(theta)
            cos = torch.cos(theta)
            row_1 = torch.cat([cos[:, None], sin[:, None]], dim=1)
            row_2 = torch.cat([-sin[:, None], cos[:, None]], dim=1)
            g = torch.stack([row_1, row_2], dim=1)
            return g

        gs = gen_group_elements(self.n_samples, self.device).unsqueeze(0)
        new_x = torch.matmul(gs, x.unsqueeze(1))
        out = self.model(new_x, t)
        out = torch.matmul(torch.transpose(gs, -1, -2), torch.matmul(out, gs))
        out = torch.mean(out, dim=1)
        return out






if __name__ == '__main__':

    torch.manual_seed(43)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MLP(4, 4, width=64, depth=2, activation='relu').to(device)
    bs = 10
    ns = 4

    x = torch.rand((bs, 2, 2), device=device)
    t = torch.randint(high=2, size=(bs, 1), device=device)
    t = t.expand(bs, ns)
    # print(model(x, t))

    inv_model = InvariantModel(model, n_samples=ns, device=device).to(device)
    print(inv_model(x, t))




