"""
data_utils.py

Helper utilities for preparing toy datasets used in AsyncPP experiments.
This includes:
- Generating synthetic input tensors
- Splitting data into batches for training

These utilities are mainly for quick experiments and benchmarking.

Example:
    >>> from data_utils import generate_data, get_batches
    >>> x, y = generate_data(num_samples=100, input_dim=10)
    >>> batches = get_batches(x, y, batch_size=16)
    >>> len(batches)
    7
"""

import torch


def generate_data(num_samples: int = 1000, input_dim: int = 32):
    """
    Generate a simple synthetic dataset (inputs and labels).

    Args:
        num_samples (int): Number of samples to generate.
        input_dim (int): Size of each input vector.

    Returns:
        tuple:
            - x (torch.Tensor): Input features of shape (num_samples, input_dim).
            - y (torch.Tensor): Labels (sum of each input row) of shape (num_samples, 1).

    Example:
        >>> x, y = generate_data(5, 4)
        >>> x.shape
        torch.Size([5, 4])
        >>> y.shape
        torch.Size([5, 1])
    """
    x = torch.randn(num_samples, input_dim)   # Random normal input
    y = x.sum(dim=1, keepdim=True)            # Label = sum of features
    return x, y


def get_batches(x: torch.Tensor, y: torch.Tensor, batch_size: int = 32):
    """
    Split dataset into batches.

    Args:
        x (torch.Tensor): Input features of shape (N, D).
        y (torch.Tensor): Labels of shape (N, 1).
        batch_size (int): Number of samples per batch.

    Returns:
        list of tuples: Each element is (x_batch, y_batch).

    Example:
        >>> x, y = generate_data(100, 10)
        >>> batches = get_batches(x, y, batch_size=16)
        >>> len(batches)
        7
        >>> x_b, y_b = batches[0]
        >>> x_b.shape, y_b.shape
        (torch.Size([16, 10]), torch.Size([16, 1]))
    """
    n = x.shape[0]
    batches = []
    for i in range(0, n, batch_size):
        x_batch = x[i:i + batch_size]
        y_batch = y[i:i + batch_size]
        batches.append((x_batch, y_batch))
    return batches
