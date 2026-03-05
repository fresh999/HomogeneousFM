from math import pi

import torch
from torch.linalg import vector_norm

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



def inf_train_gen(batch_size: int = 200, device: str = 'cpu') -> torch.Tensor:
    '''
        Generates checkerboard distribution on the upper half-plane model of hyperbolic space.
    '''

    x = torch.rand(batch_size, device=device) * 6 - 3
    y = torch.rand(batch_size, device=device) - torch.randint(high=3, size=(batch_size,), device=device) * 2
    y += (torch.floor(x) % 2)

    data = torch.cat([x[:, None], y[:, None]], dim=1) * 0.45
    d = vector_norm(data, dim=1).unsqueeze(-1)
    data /= (1+d)

    dd = (1 - data[:, 0]) ** 2 + data[:, 1] ** 2
    new_x = -2 * data[:, 1] / dd
    new_y = (1 - data[:, 0] ** 2 - data[:, 1] ** 2) / dd
    new_data = torch.cat([new_x[:, None], new_y[:, None]], dim=1)

    return new_data.float()


if __name__ == '__main__':

    torch.manual_seed(42)

    data = inf_train_gen(10000)

    plt.scatter(data[:, 0].detach(), data[:, 1].detach(), marker='.', alpha=0.5)
    plt.show()











