from math import pi

import torch
import torch.nn as nn
import torch.nn.functional as F

activations_dict = {'relu': F.relu,
                    'swish': nn.SiLU()}

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










if __name__ == '__main__':

    torch.manual_seed(43)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MLP(4, 4, width=64, depth=2, activation='swish').to(device)
    bs = 10
    ns = 4

    x = torch.rand((bs, 2, 2), device=device)
    t = torch.randint(high=2, size=(bs, 1), device=device)
    t = t.expand(bs, ns)
    # print(model(x, t))





