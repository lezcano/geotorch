# Tests for the Sphere
from unittest import TestCase

import torch
import torch.nn as nn

import geotorch.parametrize as P
from geotorch.sphere import Sphere, SphereEmbedded


class TestSphere(TestCase):
    def assertInSn(self, X):
        norm = X.norm(dim=-1)
        ones = torch.ones_like(norm)
        self.assertAlmostEqual(
            torch.norm(norm - ones, p=float("inf")).item(), 0.0, places=5
        )

    def test_backprop(self):
        r"""Test that we may instantiate the parametrizations and
        register them in modules of several sizes. Check that the
        results are on the sphere
        """
        sizes = [1, 2, 3, 4, 7, 8]

        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.random.manual_seed(8888)
            for n in sizes:
                for cls in [Sphere, SphereEmbedded]:
                    layer = nn.Linear(n, 4)
                    P.register_parametrization(
                        layer, "bias", cls(size=layer.bias.size())
                    )
                    P.register_parametrization(
                        layer, "weight", cls(size=layer.weight.size())
                    )

                    with torch.no_grad():
                        layer.parametrizations.weight.uniform_init_()
                        layer.parametrizations.bias.uniform_init_()

                    input_ = torch.rand(5, n)
                    optim = torch.optim.SGD(layer.parameters(), lr=1.0)

                    # Assert that is stays in S^n after some optimiser steps
                    for i in range(2):
                        print(i)
                        with P.cached():
                            self.assertInSn(layer.weight)
                            self.assertInSn(layer.bias)
                            loss = layer(input_).sum()
                        optim.zero_grad()
                        loss.backward()
                        optim.step()

                    # If we change the base, the forward pass should give the same
                    # SphereEmbedded does not have a base
                    if cls != SphereEmbedded:
                        for w in ["weight", "bias"]:
                            out_old = layer(input_)
                            getattr(layer.parametrizations, w).update_base()
                            out_new = layer(input_)
                            self.assertAlmostEqual(
                                (out_old - out_new).abs().max().item(), 0.0, places=5
                            )

    def test_construction(self):
        # Negative curvature
        with self.assertRaises(ValueError):
            Sphere(size=(5,), r=-1.0)
        with self.assertRaises(ValueError):
            SphereEmbedded(size=(4,), r=-1.0)

        # Wrong trivialization
        with self.assertRaises(ValueError):
            SphereEmbedded(size=(5,), triv="wrong")

        # Custom trivialization
        def proj(x):
            return x / x.norm()

        try:
            SphereEmbedded(size=(3,), triv=proj)
        except Exception as e:
            self.fail("SphereEmbedded.__init__ raised {} unexpectedly!".format(type(e)))

    def test_repr(self):
        print(SphereEmbedded(size=(3,)))
        print(Sphere(size=(3,)))
