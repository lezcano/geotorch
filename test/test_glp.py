from unittest import TestCase

from geotorch.glp import GLp


class TestGLp(TestCase):
    def test_GLp_errors(self):
        # Non square
        with self.assertRaises(ValueError):
            GLp(size=(4, 3))
        # Try to instantiate it in a vector rather than a matrix
        with self.assertRaises(ValueError):
            GLp(size=(5,))
