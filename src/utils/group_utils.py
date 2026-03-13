from abc import ABC, abstractmethod
from scipy.linalg import logm

import torch



class Group(ABC):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    @abstractmethod
    def mul(self, g_1: torch.Tensor, g_2: torch.Tensor) -> torch.Tensor:
        '''Returns the group elements g_1 g_2 (group multiplication).'''
        ...

    @abstractmethod
    def inv(self, g: torch.Tensor) -> torch.Tensor:
        '''Returns g^{-1}.'''
        ...


class LieGroup(Group):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)

    @abstractmethod
    def exp(self, x: torch.Tensor) -> torch.Tensor:
        '''Returns exponential of element in Lie algebra.'''
        ...


class MatrixGroup(LieGroup):
    def __init__(self, dim: int, matrix_size: int, lie_algebra_basis: torch.Tensor = None) -> None:
        super().__init__(dim)
        self.matrix_size = matrix_size
        self.lie_algebra_basis = lie_algebra_basis

    def mul(self, g_1: torch.Tensor, g_2: torch.Tensor) -> torch.Tensor:
        return g_1 @ g_2

    def inv(self, g: torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv(g)

    def exp(self, x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.matrix_exp(x)

    def log(self, g: torch.Tensor) -> torch.Tensor:
        '''
        Warning: very slow implementation since logm is not implemented on GPU (yet).
        Override this method when a more efficient implementation exists.
        '''
        device = g.device
        return torch.from_numpy(logm(g.cpu().numpy())).to(device)


class SL2R(MatrixGroup):
    def __init__(self, lie_algebra_basis: torch.Tensor = None) -> None:
        super().__init__(3, 2, lie_algebra_basis)

    def exp(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:

        '''
        Assumes x is traceless (i.e. it lies in the Lie algebra of SL(2, R)). Spits out gibberish if it doesn't.
        '''

        I = torch.eye(2, device=x.device, dtype=x.dtype).expand(x.shape)
        out = torch.zeros_like(x)

        delta = x[..., 0, 0] ** 2 + x[..., 0, 1] * x[..., 1, 0]

        mask_h = delta > eps
        mask_e = delta < -eps
        mask_p = torch.abs(delta) <= eps

        lam = torch.sqrt(delta[mask_h])
        cosh = torch.cosh(lam)
        factor = torch.sinh(lam) / lam
        out[mask_h] = cosh[..., None, None] * I[mask_h] + factor[..., None, None] * x[mask_h]

        theta = torch.sqrt(-delta[mask_e])
        cos = torch.cos(theta)
        factor = torch.sin(theta) / theta
        out[mask_e] = cos[..., None, None] * I[mask_e] + factor[..., None, None] * x[mask_e]

        out[mask_p] = I[mask_p] + x[mask_p]
        return out

    def log(self, g: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        '''
        Assumes that g lies in the image of the exponential map. Spits out gibberish if it doesn't.
        Warning: unstable near t = 0. Change it.
        '''

        if g.ndim == 2:
            g = g.unsqueeze(0)

        I = torch.eye(2, device=g.device, dtype=g.dtype).expand_as(g)
        out = torch.zeros_like(g)

        t = 0.5 * g.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

        mask_h = torch.abs(t) > 1 + eps
        mask_e = torch.abs(t) < 1 - eps
        mask_p = torch.abs(t-1) <= eps

        factor_h = torch.acosh(t[mask_h]) / torch.sqrt(t[mask_h] ** 2 - 1)
        factor_e = torch.acos(t[mask_e]) / torch.sqrt(1 - t[mask_e] ** 2)

        out[mask_h] = (g[mask_h] - t[mask_h][..., None, None] * I[mask_h]) * factor_h[:, None, None]
        out[mask_e] = (g[mask_e] - t[mask_e][..., None, None] * I[mask_e]) * factor_e[:, None, None]
        out[mask_p] = g[mask_p] - I[mask_p]

        return out


class SO3R(MatrixGroup):
    def __init__(self, lie_algebra_basis: torch.Tensor = None) -> None:
        super().__init__(3, 2, lie_algebra_basis)

    def exp(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        '''
        Assumes x is antisymmetric (i.e. it lies in the Lie algebra of SO(3)). Spits out gibberish if it doesn't.
        Warning: unstable near theta = 0. Change it.
        '''

        I = torch.eye(3, device=x.device, dtype=x.dtype).expand_as(x)
        out = torch.zeros_like(x)

        x_1 = x[..., 2, 1].unsqueeze(-1) # shape (batch_dimesions, 1)
        x_2 = x[..., 0, 2].unsqueeze(-1)
        x_3 = x[..., 1, 0].unsqueeze(-1)
        x_vec = torch.cat([x_1, x_2, x_3], dim=-1)

        theta = torch.sqrt((x_vec ** 2).sum(-1))
        x_vec = x_vec.unsqueeze(-1) # shape (batch_dimensions, 3, 1)
        x_squared = torch.matmul(x_vec, x_vec.transpose(-1, -2)) - (theta ** 2)[..., None, None] * I

        a = torch.sin(theta) / theta
        b = (1 - torch.cos(theta)) / (theta ** 2)

        mask_away = theta > eps
        mask_zero = theta <= eps

        out[mask_away] = I[mask_away] \
                         + a[mask_away][..., None, None] * x[mask_away] \
                         + b[mask_away][..., None, None] * x_squared[mask_away]
        out[mask_zero] = I[mask_zero] + x[mask_zero] + 0.5 * x_squared[mask_zero]
        return out

    def log(self, g: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        '''
        Assumes g is an orthogonal matrix. Spits out gibberish if it's not.
        Warning: unstable at theta = 0. Change it.
        '''

        out = torch.zeros_like(g)

        trace = torch.clamp(g.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1), -1.0, 1.0)
        theta = torch.acos(0.5 * (trace - 1))

        g_asym = g - g.transpose(-1, -2)

        mask_away = (torch.abs(theta) > eps) & (torch.abs(theta - torch.pi) > eps)
        mask_zero = torch.abs(theta) <= eps
        mask_pi = torch.abs(theta - torch.pi) <= eps

        factor = theta / (2 * torch.sin(theta))
        out[mask_away] = factor[mask_away][..., None, None] * g_asym[mask_away]

        out[mask_zero] = g_asym[mask_zero]

        g_pi = g[mask_pi]
        u_abs = torch.zeros(g_pi.shape[:-2] + (3,), device=g.device, dtype=g.dtype)
        for i in range(3):
            idx = g_pi[..., i, i] + 1 > eps
            u_abs[idx, i] = torch.sqrt((g_pi[idx, i, i] + 1) / 2)

        vals, ids = torch.max(u_abs, dim=-1)
        u = g_pi[torch.arange(g_pi.shape[0]), ids]
        u /= vals[:, None]
        u[torch.arange(g_pi.shape[0]), ids] = u_abs[torch.arange(g_pi.shape[0]), ids]
        print(u)
        print(u_abs)
        print(vals)
        print(ids)
        print(u)





        return None








if __name__ == '__main__':

    G = SO3R()

    torch.manual_seed(42)

    x = torch.randn(size=(2000, 3, 3))
    x = x - x.transpose(-1, -2)

    e = G.exp(x)
    print(G.log(e))


