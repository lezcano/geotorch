from unittest import TestCase
import itertools

import torch
import torch.nn as nn

import geotorch.parametrize as P
from geotorch.lowrank import LowRank


class TestLowRank(TestCase):
    def assertIsOrthogonal(self, X):
        if X.size(-2) < X.size(-1):
            X = X.transpose(-2, -1)
        Id = torch.eye(X.size(-1))
        if X.dim() > 2:
            Id = Id.repeat(*(X.size()[:-2] + (1, 1)))
        norm = torch.norm(X.transpose(-2, -1) @ X - Id, dim=(-2, -1))
        self.assertTrue((norm < 1e-3).all())

    def assertHasSingularValues(self, X, S_orig):
        if X.ndimension() == 2:
            self._assertHasSingularValues(X, S_orig)
        elif X.ndimension() > 2:
            # Sample a few elements and see that they have the correct sing values
            for _ in range(4):
                coords = [
                    torch.randint(low=0, high=s, size=(1,)).item()
                    for s in X.size()[:-2]
                ]
                coords.append(...)
                self._assertHasSingularValues(X[coords], S_orig[coords])

    def _assertHasSingularValues(self, X, S_orig):
        _, S, _ = torch.svd(X)
        # Sort the singular values from our parametrization
        S_orig = torch.sort(torch.abs(S_orig), descending=True).values

        def first_nonzero(x):
            x = x.flip(0)
            nonz = x > 0
            return ((nonz.cumsum(0) == 1) & nonz).max(0).indices

        # Clamp almost-zero values
        S[S < 1e-5] = 0.0
        idx = first_nonzero(S)
        if idx != 0:
            # Delete zero values to reveal true rank
            S = S[:-idx]
        # Rather lax as the SVD is quite unstable
        self.assertAlmostEqual((S_orig - S).abs().max().item(), 0.0, places=1)

    def test_lowrank(self):
        sizes = [
            (8, 1),
            (8, 4),
            (8, 8),
            (7, 1),
            (7, 3),
            (7, 4),
            (7, 7),
            (1, 7),
            (2, 7),
            (1, 8),
            (2, 8),
            (1, 1),
            (2, 1),
            (1, 2),
        ]

        rs = [1, 3, 8]

        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.random.manual_seed(8888)
            for (n, k), r in itertools.product(sizes, rs):
                for layer in [nn.Linear(n, k), nn.Conv2d(n, 4, k)]:
                    r = min(n, k, r)
                    LR = LowRank(size=layer.weight.size(), rank=r)
                    P.register_parametrization(layer, "weight", LR)
                    print(LR)
                    self.assertTrue(P.is_parametrized(layer, "weight"))
                    U_orig, S_orig, V_orig = LR.original
                    self.assertIsOrthogonal(U_orig)
                    self.assertIsOrthogonal(V_orig)
                    self.assertHasSingularValues(layer.weight, S_orig)

                    optim = torch.optim.SGD(layer.parameters(), lr=0.1)
                    if isinstance(layer, nn.Linear):
                        input_ = torch.rand(5, n)
                    elif isinstance(layer, nn.Conv2d):
                        # batch x in_channel x in_length x in_width
                        input_ = torch.rand(6, n, 9, 8)

                    for i in range(2):
                        print(i)
                        loss = layer(input_).sum()
                        optim.zero_grad()
                        loss.backward()
                        optim.step()

                        U_orig, S_orig, V_orig = LR.original
                        self.assertIsOrthogonal(U_orig)
                        self.assertIsOrthogonal(V_orig)
                        self.assertHasSingularValues(layer.weight, S_orig)

                    # Test update_base
                    prev_out = layer(input_)
                    layer.parametrizations.weight.update_base()
                    new_out = layer(input_)
                    self.assertAlmostEqual(
                        torch.norm(prev_out - new_out).abs().max().item(), 0.0, places=3
                    )

    def test_lowrank_errors(self):
        # rank always has to be <= min(n, k)
        with self.assertRaises(ValueError):
            LowRank(size=(4, 3), rank=5)
        with self.assertRaises(ValueError):
            LowRank(size=(2, 3), rank=3)
        # Try to instantiate it in a vector rather than a matrix
        with self.assertRaises(ValueError):
            LowRank(size=(5,), rank=1)
