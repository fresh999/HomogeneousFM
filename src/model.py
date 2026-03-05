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

    '''
        reshape input tensor
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dense_layer(x)
        return self.activation(y)


class MLP(nn.Module):
    def __init__(self, input_features: int, output_features: int, width: int, depth: int, activation: str) -> None:
        super().__init__()
        self.net = nn.Sequential(
            MLPBlock(input_features, width, activation),
            *[MLPBlock(width, width, activation) for _ in range(depth)],
            MLPBlock(width, output_features, None)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)





