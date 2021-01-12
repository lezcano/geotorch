# Tests for the Sphere
from unittest import TestCase
import itertools

import torch
import torch.nn as nn

import geotorch.parametrize as P
from geotorch.skew import Skew


class TestSkew(TestCase):
    def test_backprop(self):
        r"""Test that we may instantiate the parametrizations and
        register them in modules of several sizes. Check that the
        results are on the sphere
        """
        sizes = [1, 2, 3, 8]

        for n, lower in itertools.product(sizes, [True, False]):
            layer = nn.Linear(n, n)
            P.register_parametrization(layer, "weight", Skew(lower=lower))

            input_ = torch.rand(5, n)
            optim = torch.optim.SGD(layer.parameters(), lr=1.0)

            # Assert that is stays in Skew(n) after some optimiser steps
            for _ in range(2):
                with P.cached():
                    self.assertTrue(Skew.in_manifold(layer.weight))
                    loss = layer(input_).sum()
                optim.zero_grad()
                loss.backward()
                optim.step()

    def test_non_square(self):
        # Non-square skew
        with self.assertRaises(ValueError):
            Skew()(torch.rand(3, 2))

        with self.assertRaises(ValueError):
            Skew()(torch.rand(1, 3))

        # Try to instantiate it in a vector rather than a matrix
        with self.assertRaises(ValueError):
            Skew()(torch.rand(4))

    def test_repr(self):
        print(Skew())
