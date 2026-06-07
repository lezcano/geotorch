from unittest import TestCase

import torch

from geotorch.pssdlowrank import PSSDLowRank
from geotorch.pssdfixedrank import PSSDFixedRank
from geotorch.pssd import PSSD
from geotorch.psd import PSD


class TestPSSDLowRank(TestCase):
    def test_positive_semidefinite_errors(self):
        for cls in [PSSDLowRank, PSSDFixedRank]:
            # rank always has to be 1 <= rank <= n
            with self.assertRaises(ValueError):
                cls(size=(4, 4), rank=5)
            with self.assertRaises(ValueError):
                cls(size=(3, 3), rank=0)
            # Instantiate it in a non-square matrix
            with self.assertRaises(ValueError):
                cls(size=(3, 6), rank=2)
            # Try to instantiate it in a vector rather than a matrix
            with self.assertRaises(ValueError):
                cls(size=(5,), rank=1)

        for cls in [PSSD, PSD]:
            # Try to instantiate it in a vector rather than a matrix
            with self.assertRaises(ValueError):
                cls(size=(5,))
            # Or a non-square
            with self.assertRaises(ValueError):
                cls(size=(5, 3))

        # Pass a non-callable object
        with self.assertRaises(ValueError):
            PSSDFixedRank(size=(5, 2), rank=1, f=3)
        # Or the wrong string
        with self.assertRaises(ValueError):
            PSSDFixedRank(size=(5, 3), rank=2, f="fail")
        # Same with PSD
        with self.assertRaises(ValueError):
            PSD(size=(5, 2), f=3)
        with self.assertRaises(ValueError):
            PSD(size=(5, 3), f="fail")

    def test_fixed_rank_sample_clamps_eigenvalues(self):
        manifold = PSSDFixedRank(size=(3, 3), rank=2).double()
        sample = manifold.sample(init_=lambda X: X.zero_(), eps=0.25)
        eigenvalues = torch.linalg.eigvalsh(sample)
        expected = torch.tensor([0.0, 0.25, 0.25], dtype=torch.float64)
        torch.testing.assert_close(eigenvalues, expected)
