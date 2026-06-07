from unittest import TestCase

import torch

from geotorch.sl import SL


class TestSL(TestCase):
    def test_SL_errors(self):
        # Non square
        with self.assertRaises(ValueError):
            SL(size=(4, 3))
        # Try to instantiate it in a vector rather than a matrix
        with self.assertRaises(ValueError):
            SL(size=(5,))

    def test_singular_values_are_normalized_in_log_domain(self):
        manifold = SL(size=(3, 3))
        singular_values = manifold.f(torch.tensor([[-100.0, 0.0, 100.0]]))
        self.assertTrue(torch.isfinite(singular_values).all())
        self.assertTrue(
            torch.allclose(
                singular_values.log().sum(dim=-1),
                torch.zeros(1),
                atol=1e-5,
            )
        )

    def test_custom_map_is_normalized(self):
        manifold = SL(size=(2, 3, 3), f=(torch.exp, torch.log)).double()
        X = torch.randn(2, 3, 3, dtype=torch.float64)
        matrix = manifold(X)
        sign, logabsdet = torch.linalg.slogdet(matrix)

        self.assertTrue(torch.equal(sign, torch.ones_like(sign)))
        torch.testing.assert_close(
            logabsdet, torch.zeros_like(logabsdet), atol=1e-6, rtol=0
        )

        inverse = manifold.right_inverse(matrix)
        torch.testing.assert_close(manifold(inverse), matrix, atol=1e-6, rtol=1e-6)

    def test_negative_determinant_is_rejected(self):
        manifold = SL(size=(2, 2))
        self.assertFalse(manifold.in_manifold(torch.diag(torch.tensor([-1.0, 1.0]))))

    def test_sample_has_positive_unit_determinant(self):
        manifold = SL(size=(4, 5, 5)).double()
        sample = manifold.sample()
        self.assertTrue(manifold.in_manifold(sample))
        determinant = torch.linalg.det(sample)
        self.assertTrue(torch.allclose(determinant, torch.ones_like(determinant)))
