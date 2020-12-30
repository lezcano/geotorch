from unittest import TestCase

from geotorch.lowrank import LowRank
from geotorch.fixedrank import FixedRank


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
