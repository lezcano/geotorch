# Tests for the Stiefel manifold, grassmannian and SO(n)
from unittest import TestCase

from geotorch.so import SO
from geotorch.stiefel import Stiefel
from geotorch.grassmannian import Grassmannian
from geotorch.exceptions import NonSquareError, VectorError


class TestOrthogonal(TestCase):
    def test_constructor_stiefel(self):
        self._test_constructor(Stiefel)

    def test_constructors_grassmannian(self):
        self._test_constructor(Grassmannian)

    def _test_constructor(self, cls):
        with self.assertRaises(ValueError):
            cls(size=(3, 3), triv="wrong")

        with self.assertRaises(ValueError):
            SO(size=(3, 3), triv="wrong")

        # Try a custom trivialization (it should break in the forward)
        try:
            cls(size=(3, 3), triv=lambda: 3)
        except ValueError:
            self.fail("{} raised ValueError unexpectedly!".format(cls))

        # Try to instantiate it in a vector rather than a matrix
        with self.assertRaises(VectorError):
            cls(size=(7,))

        with self.assertRaises(VectorError):
            SO(size=(7,))

        # Try to instantiate it in an on-square matrix
        with self.assertRaises(NonSquareError):
            SO(size=(7, 3, 2))
