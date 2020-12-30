# Tests for the Sphere
from unittest import TestCase

from geotorch.sphere import Sphere, SphereEmbedded


class TestSphere(TestCase):
    def test_construction(self):
        # Negative curvature
        with self.assertRaises(ValueError):
            Sphere(size=(5,), radius=-1.0)
        with self.assertRaises(ValueError):
            SphereEmbedded(size=(4,), radius=-1.0)

    def test_repr(self):
        print(SphereEmbedded(size=(3,)))
        print(Sphere(size=(3,)))
