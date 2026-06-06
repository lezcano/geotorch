from unittest import TestCase

import torch

from geotorch.lowrank import LowRank
from geotorch.fixedrank import FixedRank, inv_softplus_epsilon, softplus_epsilon


class TestLowRank(TestCase):
    def test_lowrank_errors(self):
        # rank always has to be <= min(n, k)
        for cls in [LowRank, FixedRank]:
            with self.assertRaises(ValueError):
                cls(size=(4, 3), rank=5)
            with self.assertRaises(ValueError):
                cls(size=(2, 3), rank=3)
            # Try to instantiate it in a vector rather than a matrix
            with self.assertRaises(ValueError):
                cls(size=(5,), rank=1)

        # On a non-callable
        with self.assertRaises(ValueError):
            FixedRank(size=(5, 3), rank=2, f=3)
        # On the wrong string
        with self.assertRaises(ValueError):
            FixedRank(size=(5, 3), rank=2, f="wrong")

    def test_fixed_rank_rejects_rank_deficient_matrix(self):
        manifold = FixedRank(size=(3, 3), rank=3)
        matrix = torch.diag(torch.tensor([3.0, 2.0, 0.0]))
        self.assertFalse(manifold.in_manifold(matrix))

    def test_inverse_softplus_is_stable(self):
        x = torch.tensor([1e-4, 1.0, 100.0], dtype=torch.float64)
        inverse = inv_softplus_epsilon(x)
        self.assertTrue(torch.isfinite(inverse).all())
        self.assertTrue(torch.allclose(softplus_epsilon(inverse), x))
