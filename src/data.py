from math import pi

import torch
from torch.linalg import vector_norm



def inf_train_gen(batch_size: int = 200, device: str = 'cpu') -> torch.Tensor:
    '''
        Generates checkerboard distribution on the upper half-plane model of hyperbolic space.
    '''

    # first generate distribution in polar coordinates
    u = torch.rand(batch_size, device=device) * 4
    v_ = torch.rand(batch_size, device=device) - torch.randint(high=2, size=(batch_size,), device=device) * 2
    v = v_ + (torch.floor(u) % 2) + 2
    v *= (pi / 2)
    # data = 1.0 * torch.cat([u[:, None], v[:, None]], dim=1)

    x = torch.sinh(u) * torch.cos(v)
    y = torch.sinh(u) * torch.sin(v)
    z = torch.cosh(u)

    data = 1.0 * torch.cat([x[:, None], y[:, None]], dim=1)
    z.view(50, 1)
    print(data.shape)
    data /= (z+1)

    return data.float()


if __name__ == '__main__':

    torch.manual_seed(42)

    data = inf_train_gen(50)
    # print(data)






