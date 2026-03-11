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
        return torch.from_numpy(logm(g.numpy()))




if __name__ == '__main__':

    circle = MatrixGroup(1, 2)
    g = torch.Tensor(
        [[0.0, -1.0],
        [1.0, 0.0]]
    )

    log = circle.log(g)
    print(circle.exp(log))

