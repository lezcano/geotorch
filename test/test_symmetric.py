# Tests for the Sphere
from unittest import TestCase
import itertools

import torch
import torch.nn as nn

import geotorch.parametrize as P
from geotorch.symmetric import Symmetric, SymF


class TestSymmetric(TestCase):
    def test_backprop(self):
        r"""Test that we may instantiate the parametrizations and
        register them in modules of several sizes. Check that the
        results are on the sphere
        """
        sizes = [1, 2, 3, 8]

        for n, lower in itertools.product(sizes, [True, False]):
            layer = nn.Linear(n, n)
            P.register_parametrization(layer, "weight", Symmetric(lower=lower))

            input_ = torch.rand(5, n)
            optim = torch.optim.SGD(layer.parameters(), lr=1.0)

            # Assert that is stays in Sym(n) after some optimiser steps
            for _ in range(2):
                with P.cached():
                    self.assertTrue(Symmetric.in_manifold(layer.weight))
                    loss = layer(input_).sum()
                optim.zero_grad()
                loss.backward()
                optim.step()

    def test_construction(self):
        # Non-square sym
        with self.assertRaises(ValueError):
            Symmetric()(torch.rand(3, 2))

        with self.assertRaises(ValueError):
            Symmetric()(torch.rand(1, 3))

        # Try to instantiate it in a vector rather than a matrix
        with self.assertRaises(ValueError):
            Symmetric()(torch.rand(4))

        # Instantiate it with a non-callable object
        with self.assertRaises(ValueError):
            SymF(size=(4, 4), rank=4, f=3.0)
        # Or with the wrong rank
        with self.assertRaises(ValueError):
            SymF(size=(4, 4), rank=5, f=lambda: None)
        with self.assertRaises(ValueError):
            SymF(size=(4, 4), rank=0, f=lambda: None)
        # Or on vectors
        with self.assertRaises(ValueError):
            SymF(size=(4,), rank=2, f=lambda: None)
        # Or on non-square matrices
        with self.assertRaises(ValueError):
            SymF(size=(4, 3), rank=2, f=lambda: None)

    def test_repr(self):
        print(Symmetric())
