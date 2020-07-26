import os
import time

import torch


def loss_MSE(input, target):
    """Computes the MSE loss between the ``input`` and ``target`` tensors.

    Parameters
    ----------
    input : torch.Tensor instance
        The input tensor to be used in the MSE loss computation.
    target : torch.Tensor instance
        The target tensor to be used in the MSE loss computation.

    Returns
    -------
    float
        The MSE loss between the ``input`` and the ``target`` tensors.
    
    """
    return torch.sum(torch.abs(input.data - target.data) ** 2)
