from math import pi

import torch
from torch.linalg import vector_norm

import matplotlib.pyplot as plt


def inf_train_gen(batch_size: int = 2048, device: str = 'cpu') -> torch.Tensor:
    '''Generates checkerboard distribution on the upper half-plane model of hyperbolic space.'''

    x = torch.rand(batch_size, device=device) * 4 - 2
    y = torch.rand(batch_size, device=device) - torch.randint(high=2, size=(batch_size,), device=device) * 2
    y += (torch.floor(x) % 2 + 2.25)

    data = torch.cat([x[:, None], y[:, None]], dim=1) / 0.45
    return data.float()

def sl2_noise(batch_size: int = 2048, device: str = 'cpu') -> torch.Tensor:
    coeffs = torch.rand(batch_size, 3, device=device)
    a, b, c = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]

    X = torch.zeros(batch_size, 2, 2, device=device)
    X[:, 0, 0] = a
    X[:, 1, 1] = -a
    X[:, 0, 1] = b
    X[:, 1, 0] = c

    return torch.linalg.matrix_exp(X)


def section(data: torch.Tensor) -> torch.Tensor:
    '''Maps points on upper half-plane to symplectic group.'''

    x, y = data[:, 0, None], data[:, 1, None]
    sqrt_y = torch.sqrt(y)
    row_1 = torch.cat([sqrt_y, x / sqrt_y], dim=1)
    row_2 = torch.cat([torch.zeros_like(y), 1 / sqrt_y], dim=1)

    return torch.stack([row_1, row_2], dim=1)


def project(data: torch.Tensor) -> torch.Tensor:
    '''Maps data points on Sp(2) to upper half-plane (Sp(2) acts on the upper half-plane by fractional transformations).
    data.shape = [batch_size, 2, 2]
    '''

    data = data.to(dtype=torch.cfloat)
    z = torch.tensor([1j, 1.], dtype=torch.cfloat, device=data.device)
    w_vec = data @ z

    w = w_vec[..., 0] / w_vec[..., 1]
    return torch.stack([w.real, w.imag], dim=-1)


def augment_data_random(data: torch.Tensor, n_samples: int = 256) -> torch.Tensor:
    '''Samples elements from U(1) and averages data around.
    data.shape = [batch_size, 2, 2]
    '''

    def gen_group_elements(n_samples: int, device: str) -> torch.Tensor:
        theta = 2 * pi * torch.rand(n_samples, device=device)
        sin = torch.sin(theta)
        cos = torch.cos(theta)
        row_1 = torch.cat([cos[:, None], sin[:, None]], dim=1)
        row_2 = torch.cat([-sin[:, None], cos[:, None]], dim=1)
        g = torch.stack([row_1, row_2], dim=1)
        return g

    gs = gen_group_elements(n_samples, data.device).unsqueeze(0) # shape [1, n_samples, 2, 2]
    aug_data = torch.matmul(data.unsqueeze(1), gs)
    return aug_data.reshape(-1, aug_data.shape[-2], aug_data.shape[-1])





if __name__ == '__main__':

    torch.manual_seed(42)
    batch_size = 4096

    '''
    data = inf_train_gen(batch_size)
    sec = section(data)
    # sec = augment_data_random(sec, 2)
    p = project(sec)

    plot = plt.scatter(data[:, 0].detach(), data[:, 1].detach(), marker='.', alpha=0.5)
    # plt.show()
    plt.savefig('data.pdf')
    plot.remove()

    plt.scatter(p[:, 0].detach(), p[:, 1].detach(), marker='.', alpha=0.5)
    plt.savefig('proj.pdf')
    '''

    noise = project(sl2_noise(batch_size))

    plt.scatter(noise[:, 0].detach(), noise[:, 1].detach(), marker='.', alpha=0.5)
    plt.savefig('noise.pdf')












