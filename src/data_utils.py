from math import pi

import torch
from torch.linalg import vector_norm

import matplotlib.pyplot as plt


def inf_train_gen(batch_size: int = 2048, device: str = 'cpu', upper: bool = True) -> torch.Tensor:
    '''
    Generates checkerboard distribution on the plane.
    If upper is True, the points are generated on the upper half-plane.
    '''

    shift = 2.25 if upper else 0.0
    x = torch.rand(batch_size, device=device) * 4 - 2
    y = torch.rand(batch_size, device=device) - torch.randint(high=2, size=(batch_size,), device=device) * 2
    y += (torch.floor(x) % 2 + shift)

    data = torch.cat([x[:, None], y[:, None]], dim=1) / 0.45
    return data.float()


def sl2_noise(batch_size: int = 2048, device: str = 'cpu') -> torch.Tensor:
    coeffs = torch.rand(batch_size, 3, device=device)
    a, b, c = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]

    x = torch.zeros(batch_size, 2, 2, device=device)
    x[:, 0, 0] = a
    x[:, 1, 1] = -a
    x[:, 0, 1] = b
    x[:, 1, 0] = c

    return torch.linalg.matrix_exp(x)


def sl2_section(data: torch.Tensor) -> torch.Tensor:
    '''Maps points on upper half-plane to SL(2, R).'''

    x, y = data[:, 0, None], data[:, 1, None]
    sqrt_y = torch.sqrt(y)
    row_1 = torch.cat([sqrt_y, x / sqrt_y], dim=1)
    row_2 = torch.cat([torch.zeros_like(y), 1 / sqrt_y], dim=1)

    return torch.stack([row_1, row_2], dim=1)


def sl2_project(data: torch.Tensor) -> torch.Tensor:
    '''
    Maps data points on Sp(2) to upper half-plane (Sp(2) acts on the upper half-plane by fractional transformations).
    data.shape = [batch_size, 2, 2].
    '''

    data = data.to(dtype=torch.cfloat)
    z = torch.tensor([1j, 1.], dtype=torch.cfloat, device=data.device)
    w_vec = data @ z

    w = w_vec[..., 0] / w_vec[..., 1]
    return torch.stack([w.real, w.imag], dim=-1)


def so3_noise(batch_size: int = 2048, device: str = 'cpu') -> torch.Tensor:
    '''Samples noise from SO(3).'''

    coeffs = torch.randn([batch_size, 3], device=device)
    a, b, c = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]

    x = torch.stack([
        torch.stack([torch.zeros_like(a), a, b], dim=-1),
        torch.stack([-a, torch.zeros_like(a), c], dim=-1),
        torch.stack([-b, -c, torch.zeros_like(a)], dim=-1)
    ], dim=-2)

    return torch.linalg.matrix_exp(x)

"""
def so3_section(data: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:

    n = torch.tensor([0.0, 0.0, 1.0], dtype=data.dtype, device=data.device).reshape(1, -1)
    I = torch.eye(3, dtype=data.dtype, device=data.device)[None, ...].expand(data.shape[:-1] + (3, 3))
    out = torch.zeros_like(I)

    delta = data[..., -1] - 1
    mask_away = torch.abs(delta) > eps
    mask_north = torch.abs(delta) <= eps

    factor = 1 / (1 - data[mask_away][..., -1])
    out[mask_away] = I[mask_away] - factor[..., None, None] * torch.matmul((data[mask_away] - n).unsqueeze(-1), (data[mask_away] - n).unsqueeze(-2))
    out[mask_north] = I[mask_north]
    return out
"""

def so3_section(data: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    '''Maps data points on S^2 to representatives in SO(3).
    Warning: does not handle batch size dimensions robustly. It assumes data.shape = [bs, 3]
    '''

    n = torch.tensor([0.0, 0.0, 1.0], dtype=data.dtype, device=data.device).reshape(1, -1)
    I = torch.eye(3, dtype=data.dtype, device=data.device)[None, ...].expand(data.shape[:-1] + (3, 3))
    out = torch.zeros_like(I)

    mask_north = torch.abs(data[..., -1] - 1) <= eps
    mask_south = torch.abs(data[..., -1] + 1) <= eps
    mask_away = ~(mask_north | mask_south)

    axis = torch.linalg.cross(n, data)
    axis = axis[mask_away] / axis[mask_away].norm(dim=-1)[..., None]
    cos = torch.linalg.vecdot(n, data)[..., None, None]

    K = torch.zeros_like(I[mask_away])
    K[..., 0, 1] = -axis[..., 2]
    K[..., 1, 0] = axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 1] = axis[..., 0]

    out[mask_away] = I[mask_away] + torch.sqrt(1 - cos ** 2) * K + (1 - cos) * torch.matmul(K, K)
    out[mask_north] = I[mask_north]
    out[mask_south] = torch.diag(torch.tensor([1, -1, -1], dtype=data.dtype, device=data.device))

    return out


def so3_project(data: torch.Tensor) -> torch.Tensor:
    '''
    Projects data points on SO(3, R) to S^2.
    data.shape = [bs, 3, 3].
    '''

    n = torch.tensor([0.0, 0.0, 1.0], dtype=data.dtype, device=data.device)
    return torch.matmul(data, n)


def stereo_project(data: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    '''
    Stereographically projects points from the sphere to the plane (as implemented, it works in any dimension).
    Maps the north pole to zero (there might be a better choice here, for instance discarding any occurrence of the north pole).
    '''

    delta = 1 - data[..., -1:]
    safe_delta = torch.where(torch.abs(delta) > eps, delta, torch.ones_like(delta))
    out = data[..., :-1] / safe_delta
    out = torch.where(torch.abs(delta) > eps, out, torch.zeros_like(out))
    return out


def stereo_inverse(data: torch.Tensor) -> torch.Tensor:
    '''Wraps points from plane to sphere using the inverse of the stereographic projection (as implemented, it works in any dimension).'''

    norm_squared = data.norm(dim=tuple(range(1, data.ndim)), keepdim=True) ** 2
    out = (2 / (1 + norm_squared)) * data
    v = (norm_squared.reshape(-1) - 1) / (norm_squared.reshape(-1) + 1)
    return torch.cat([out, v[..., None]], dim=-1)




if __name__ == '__main__':

    torch.manual_seed(42)
    batch_size = 40960

    data = stereo_inverse(inf_train_gen(batch_size, upper=False))
    pp = stereo_project(data)
    plot = plt.scatter(pp[:, 0].detach(), pp[:, 1].detach(), marker='.', alpha=0.5)
    plt.savefig('data.pdf')
    plot.remove()


    sec = so3_section(data)
    print((sec.det() -1).norm())
    p = stereo_project(so3_project(sec))
    plt.scatter(p[:, 0].detach(), p[:, 1].detach(), marker='.', alpha=0.5)
    plt.savefig('proj.pdf')

    # noise = sl2_project(sl2_noise(batch_size))

    #plt.scatter(noise[:, 0].detach(), noise[:, 1].detach(), marker='.', alpha=0.5)
    # plt.savefig('noise.pdf')

    '''
    noise = so3_noise(batch_size)

    x = torch.randn([batch_size, 3])
    x = torch.cat([x, torch.Tensor([0.0, 0.0, 1.0]).unsqueeze(0)], dim=0)
    norm = x.norm(dim=-1)
    x /= norm[..., None]

    print(x)
    p = stereo_project(x)
    print(p)
    print(stereo_inverse(p))
    print(x - stereo_inverse(p))
    '''











