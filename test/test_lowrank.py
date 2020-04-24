from unittest import TestCase
import itertools

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

from geotorch.lowrank import LowRank


class TestLowRank(TestCase):
    def assertIsOrthogonal(self, X):
        if X.ndimension() == 2:
            self._assertIsOrthogonal(X)
        elif X.ndimension() > 2:
            # Sample a few elements and see that they are orthogonal
            for _ in range(4):
                coords = [
                    torch.randint(low=0, high=s, size=(1,)).item()
                    for s in X.size()[:-2]
                ]
                coords = coords + [...]
                self._assertIsOrthogonal(X[coords])

    def _assertIsOrthogonal(self, X):
        if X.size(0) < X.size(1):
            X = X.t()
        Id = torch.eye(X.size(1))
        self.assertAlmostEqual(torch.norm(X.t() @ X - Id).item(), 0.0, places=3)

    def assertHasSingularValues(self, X, S_orig):
        if X.ndimension() == 2:
            self._assertHasSingularValues(X, S_orig)
        elif X.ndimension() > 2:
            # Sample a few elements and see that they are orthogonal
            for _ in range(4):
                coords = [
                    torch.randint(low=0, high=s, size=(1,)).item()
                    for s in X.size()[:-2]
                ]
                coords = coords + [...]
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
        self.assertAlmostEqual(
            torch.norm(S_orig - S, p=float("Inf")).item(), 0.0, places=2
        )

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
                    U_orig, S_orig, V_orig = LR.total_space.evaluate()
                    self.assertIsOrthogonal(U_orig)
                    self.assertIsOrthogonal(V_orig)
                    self.assertHasSingularValues(layer.weight, S_orig)

                    optim = torch.optim.SGD(layer.parameters(), lr=0.1)
                    if isinstance(layer, nn.Linear):
                        input_ = torch.rand(5, n)
                    elif isinstance(layer, nn.Conv2d):
                        # batch x in_channel x in_length x in_width
                        input_ = torch.rand(6, n, 9, 8)

                    loss = layer(input_).sum()
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    U_orig, S_orig, V_orig = LR.total_space.evaluate()
                    self.assertIsOrthogonal(U_orig)
                    self.assertIsOrthogonal(V_orig)
                    self.assertHasSingularValues(layer.weight, S_orig)

                    loss = layer(input_).sum()
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    U_orig, S_orig, V_orig = LR.total_space.evaluate()
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
