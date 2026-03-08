import torch


def expand_tensor_like(input_tensor: torch.Tensor, expand_to: torch.Tensor) -> torch.Tensor:
    '''`input_tensor` is a 1d vector of length equal to the batch size of `expand_to`,
    expand `input_tensor` to have the same shape as `expand_to` along all remaining dimensions.

    Args:
        input_tensor (Tensor): (batch_size,).
        expand_to (Tensor): (batch_size, ...).

    Returns:
        Tensor: (batch_size, ...).
    '''

    assert input_tensor.ndim == 1, 'Input tensor must be a 1d vector.'
    assert (
        input_tensor.shape[0] == expand_to.shape[0]
    ), f'The first (batch size) dimension must match. Got shapes {input_tensor.shape} and {expand_to.shape}.'

    dim_diff = expand_to.ndim - input_tensor.ndim
    t_expanded = input_tensor.clone()
    t_expanded = t_expanded.reshape(-1, *([1] * dim_diff))

    return t_expanded.expand_as(expand_to)
