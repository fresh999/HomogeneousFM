from abc import ABC, abstractmethod

import torch


class Solver(ABC, torch.nn.Module):
    '''Abstract base class for solvers.'''

    @abstractmethod
    def sample(self, x_0: torch.Tensor) -> torch.Tensor:
        ...
