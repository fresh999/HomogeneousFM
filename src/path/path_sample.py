from dataclasses import dataclass, field

import torch


@dataclass
class PathSample:
    r'''Represents a sample of a conditional flow generated probability path.

    Attributes:
        x_1 (Tensor): the target sample :math:`X_1`.
        x_0 (Tensor): the source sample :math:`X_0`.
        t (Tensor): the time sample :math:`t`.
        x_t (Tensor): samples :math:`X_t \sim p_t(X_t)`, shape (batch_size, ...).
        dx_t (Tensor): conditional target :math:`\frac{\partial X}{\partial t}`, shape: (batch_size, ...).
    '''
    x_0: torch.Tensor = field(metadata={'help': 'source samples X_0 (batch_size, ...).'}, default=None)
    x_1: torch.Tensor = field(metadata={'help': 'target samples X_1 (batch_size, ...).'}, default=None)
    t: torch.Tensor = field(metadata={'help': 'timestamps t (batch_size, ...).'}, default=None)
    x_t: torch.Tensor = field(
        metadata={'help': 'samples x_t ~ p_t(X_t), shape (batch_size, ...).'},
        default=None
    )
    dx_t: torch.Tensor = field(
        metadata={'help': 'conditional target dX_t, shape (batch_size, ...).'},
        default=None
    )





if __name__ == '__main__':

    path_sample = PathSample()

