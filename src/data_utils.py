from math import pi

import torch
from torch.linalg import vector_norm

import matplotlib.pyplot as plt



def inf_train_gen(batch_size: int = 200, device: str = 'cpu') -> torch.Tensor:
    '''Generates checkerboard distribution on the upper half-plane model of hyperbolic space.'''

    x = torch.rand(batch_size, device=device) * 6 - 3
    y = torch.rand(batch_size, device=device) - torch.randint(high=3, size=(batch_size,), device=device) * 2
    y += (torch.floor(x) % 2 + 1)

    data = torch.cat([x[:, None], y[:, None]], dim=1) * 0.45
    d = vector_norm(data, dim=1).unsqueeze(-1)
    data /= (1+d)

    dd = (1 - data[:, 0]) ** 2 + data[:, 1] ** 2
    new_x = -2 * data[:, 1] / dd
    new_y = (1 - data[:, 0] ** 2 - data[:, 1] ** 2) / dd
    new_data = torch.cat([new_x[:, None], new_y[:, None]], dim=1)

    return new_data.float()


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
    z_vec = torch.tensor([1j, 1.], dtype=torch.cfloat, device=data.device)
    z_vec = z_vec.unsqueeze(0).expand(data.shape[0], -1).unsqueeze(-1)
    w_vec = torch.matmul(data, z_vec).squeeze(-1)

    w = w_vec[:, 0] / w_vec[:, 1]
    return torch.stack([w.real, w.imag], dim=1)


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
    batch_size = 2048

    data = inf_train_gen(batch_size)
    sec = section(data)
    # sec = augment_data_random(sec, 2)

    plt.scatter(data[:, 0].detach(), data[:, 1].detach(), marker='.', alpha=0.5)
    # plt.show()
    plt.savefig('data.pdf')

    proj = project(sec)
    print(proj.shape)
    plt.scatter(proj[:, 0].detach(), proj[:, 1].detach(), marker='.', alpha=0.5)
    plt.savefig('proj.pdf')












