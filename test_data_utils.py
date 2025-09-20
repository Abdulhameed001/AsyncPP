"""
Basic tests for data_utils.py
Ensures that generate_data and get_batches behave as expected.
"""

import torch
from data_utils import generate_data, get_batches


def test_generate_data_shapes():
    x, y = generate_data(num_samples=10, input_dim=5)
    assert x.shape == (10, 5)
    assert y.shape == (10, 1)
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)


def test_get_batches_count_and_shapes():
    x, y = generate_data(num_samples=20, input_dim=4)
    batches = get_batches(x, y, batch_size=6)
    # Expect 4 batches: (6 + 6 + 6 + 2)
    assert len(batches) == 4
    xb, yb = batches[0]
    assert xb.shape[1] == 4
    assert yb.shape[1] == 1
    assert xb.shape[0] <= 6
    assert yb.shape[0] <= 6
