from unittest import TestCase

import torch

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
