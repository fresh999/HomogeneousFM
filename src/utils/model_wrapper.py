from abc import ABC

import torch


class ModelWrapper(ABC, torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        r'''
        This method defines how inputs should be passed through the wrapped model.
        Here, we're assuming that the wrapped model takes both :math:`x` and :math:`t` as input,
        along with any additional keyword arguments.

        Optional things to do here:
            - check that t is in the dimensions that the model is expecting.
            - add a custom forward pass logic.
            - call the wrapped model.

        | given x, t
        | returns the model output for input x at time t, with extra information `extra`.

        Args:
            x (Tensor): input data to the model (batch_size, ...).
            t (Tensor): time (batch_size).
            **extras: additional information forwarded to the model, e.g., text condition.

        Returns:
            Tensor: model output.
        '''

        return self.model(x=x, t=t, **extras)
