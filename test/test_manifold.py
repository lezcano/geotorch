from unittest import TestCase

import geotorch.constructions as constructions
from geotorch.so import SO


class TestManifold(TestCase):
    def test_tensor_manifold(self):
        M = constructions.AbstractManifold(dimensions=3, size=(3, 4, 2, 1))
        self.assertEqual(M.tensorial_size, (3,))
        self.assertEqual(M.dim, (4, 2, 1))

    def test_product_manifold(self):
        SO3SO3 = constructions.ProductManifold([SO(3), SO(3)])
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
        with self.assertRaises(ValueError):
            SO3SO3.update_base()

        # Pass something that is not a manifold raises
        with self.assertRaises(TypeError):
            SO3SO3 = constructions.ProductManifold([SO(3), 3])

    def test_errors(self):
        # Pass something that is not a manifold raises
        with self.assertRaises(TypeError):
            constructions.Fibration(dimensions=2, size=(2, 4), total_space=None)

        # update_base before registering it should throw
        M = SO(3)
        with self.assertRaises(ValueError):
            M.update_base()

        # Not passing the dimensions raises
        with self.assertRaises(ValueError):
            constructions.AbstractManifold(dimensions=None, size=(2, 4))

        # Pasing a negative number raises
        with self.assertRaises(ValueError):
            constructions.AbstractManifold(dimensions=-1, size=(2, 4))

        # Passing zero should raise
        with self.assertRaises(ValueError):
            constructions.AbstractManifold(dimensions=0, size=(2, 4))
