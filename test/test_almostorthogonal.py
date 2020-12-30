from unittest import TestCase

from geotorch.almostorthogonal import AlmostOrthogonal


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
