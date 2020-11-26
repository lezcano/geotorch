from unittest import TestCase

import geotorch.constructions as constructions
from geotorch.so import SO


class TestManifold(TestCase):
    def test_product_manifold(self):
        SO3SO3 = constructions.ProductManifold([SO((3, 3)), SO((3, 3))])
        # Len
        self.assertEqual(len(SO3SO3), 2)
        # Dir
        print(dir(SO3SO3))
        # Get item
        self.assertTrue(isinstance(SO3SO3[0], SO))
        # Iter
        for M in SO3SO3:
            self.assertTrue(isinstance(M, SO))
        # repr
        print(SO3SO3)

        # Pass something that is not a manifold raises
        with self.assertRaises(TypeError):
            SO3SO3 = constructions.ProductManifold([SO((3, 3)), 3])
