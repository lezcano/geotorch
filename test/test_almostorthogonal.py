from unittest import TestCase

import pytest
import torch

import geotorch
from geotorch.almostorthogonal import (
    AlmostOrthogonal,
    inv_scaled_sigmoid,
    scaled_sigmoid,
)


class TestLowRank(TestCase):
    def test_almostorthogonal_errors(self):
        with self.assertRaises(ValueError):
            AlmostOrthogonal(size=(5,), lam=1.0)
        # Not a predefined f
        with self.assertRaises(ValueError):
            AlmostOrthogonal(size=(5, 4), lam=1, f="fail")
        # Not callable
        with self.assertRaises(ValueError):
            AlmostOrthogonal(size=(5, 4), lam=1, f=3.0)
        # But a callable should work
        AlmostOrthogonal(size=(5, 4), lam=0.5)
        # Too large a lambda
        with self.assertRaises(ValueError):
            AlmostOrthogonal(size=(5, 4), lam=2)
        # Or too small
        with self.assertRaises(ValueError):
            AlmostOrthogonal(size=(5, 4), lam=-1.0)

    def test_scaled_sigmoid_is_accurate_near_zero(self):
        x = torch.tensor([1e-8], dtype=torch.float32)
        y = scaled_sigmoid(x)
        self.assertNotEqual(y.item(), 0.0)
        torch.testing.assert_close(inv_scaled_sigmoid(y), x)


@pytest.mark.parametrize("matrix_size", [(3, 5), (5, 3), (3, 3), (1, 5), (5, 1)])
@pytest.mark.parametrize("batch_size", [(), (2, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("triv", ["expm", "cayley"])
@pytest.mark.parametrize("distribution", ["uniform", "torus"])
def test_sample_preserves_rectangular_shape(
    matrix_size, batch_size, dtype, triv, distribution
):
    size = batch_size + matrix_size
    manifold = AlmostOrthogonal(size, lam=0.5, triv=triv).to(dtype=dtype)
    sample = manifold.sample(distribution=distribution)

    assert sample.shape == size
    assert sample.dtype == dtype
    assert not sample.requires_grad
    torch.testing.assert_close(
        torch.linalg.svdvals(sample),
        torch.ones(batch_size + (min(matrix_size),), dtype=dtype),
        atol=1e-6,
        rtol=1e-6,
    )
    assert manifold.in_manifold(sample)
    torch.testing.assert_close(
        manifold(manifold.right_inverse(sample)), sample, atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("matrix_size", [(3, 5), (5, 3), (3, 3)])
@pytest.mark.parametrize("triv", ["expm", "cayley"])
def test_rectangular_constraint_registration_and_initialization(matrix_size, triv):
    rows, columns = matrix_size
    layers = [
        torch.nn.Linear(columns, rows, bias=False, dtype=torch.float64),
        torch.nn.Conv2d(2, 3, matrix_size, bias=False, dtype=torch.float64),
    ]
    for layer in layers:
        size = layer.weight.shape
        geotorch.almost_orthogonal(layer, "weight", lam=0.5, triv=triv)
        manifold = layer.parametrizations.weight[0]
        assert layer.weight.shape == size
        assert manifold.in_manifold(layer.weight)

        sample = manifold.sample(distribution="torus")
        layer.weight = sample
        torch.testing.assert_close(layer.weight, sample, atol=1e-6, rtol=1e-6)
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
        layer.weight.sum().backward()
        assert all(
            p.grad is not None and p.grad.isfinite().all() for p in layer.parameters()
        )
        optimizer.step()
        assert layer.weight.shape == size
        assert manifold.in_manifold(layer.weight)
