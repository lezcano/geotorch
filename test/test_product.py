from unittest import TestCase

import torch
from geotorch.product import ProductManifold
from geotorch.so import SO


class TestManifold(TestCase):
    def test_product_manifold(self):
        # Should not throw
        SO3SO3 = ProductManifold([SO((3, 3)), SO((3, 3))])

        # A tuple should work as well
        SO3SO3 = ProductManifold((SO((3, 3)), SO((3, 3))))

        # Forward should work
        X = (torch.rand(3, 3), torch.rand(3, 3))
        Y1, Y2 = SO3SO3(X)
