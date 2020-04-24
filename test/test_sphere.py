# Tests for the Sphere
from unittest import TestCase
import itertools

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

from geotorch.sphere import Sphere


class TestSphere(TestCase):
    def assertInSn(self, X):
        norm = X.norm(dim=-1)
        ones = torch.ones_like(norm)
        self.assertAlmostEqual(
            torch.norm(norm - ones, p=float("inf")).item(), 0.0, places=4
        )

    def test_sphere(self):
        r"""Test that we may instantiate the parametrizations and
        register them in modules of several sizes. Check that the
        results are on the sphere
        """
        sizes = [1, 2, 3, 4, 7, 8]
        trivs = ["exp"]

        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.random.manual_seed(8888)
            for n, triv in itertools.product(sizes, trivs):
                layer = nn.Linear(n, 4)
                P.register_parametrization(
                    layer, "bias", Sphere(size=layer.bias.size(), triv=triv)
                )
                P.register_parametrization(
                    layer, "weight", Sphere(size=layer.weight.size(), triv=triv)
                )

                with torch.no_grad():
                    layer.parametrizations.weight.uniform_init_()
                    # layer.parametrizations.bias.uniform_init_()

                input_ = torch.rand(5, n)
                optim = torch.optim.SGD(layer.parameters(), lr=1.0)

                # Assert that is stays in S^n after some optimiser steps
                for i in range(5):
                    print(i)
                    with P.cached(layer):
                        self.assertInSn(layer.weight)
                        self.assertInSn(layer.bias)
                        loss = layer(input_).sum()
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                # If we change the base, the forward pass should give the same
                for w in ["weight", "bias"]:
                    out_old = layer(input_)
                    getattr(layer.parametrizations, w).update_base()
                    out_new = layer(input_)
                    self.assertAlmostEqual(
                        (out_old - out_new).abs().max().item(), 0.0, places=5
                    )
