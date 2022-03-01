from unittest import TestCase

from geotorch.sl import SL


class TestSL(TestCase):
    def test_SL_errors(self):
        # Non square
        with self.assertRaises(ValueError):
            SL(size=(4, 3))
        # Try to instantiate it in a vector rather than a matrix
        with self.assertRaises(ValueError):
            SL(size=(5,))
