from unittest import TestCase
import itertools

import torch
import torch.nn as nn
import torch.nn.utils.parametrization as P

from mantorch.lowrank import LowRank

class TestLowRank(TestCase):

    def assertIsOrthogonal(self, X):
        if X.size(0) < X.size(1):
            X = X.t()
        Id = torch.eye(X.size(1))
        self.assertAlmostEqual(torch.norm(X.t() @ X - Id, p=float("Inf")).item(), 0., places=3)

    def assertHasSingularValues(self, X, S_orig):
        _, S, _ = torch.svd(X)
        # Sort the singular values from our parametrization
        S_orig = torch.sort(torch.abs(S_orig), descending=True).values
        def first_nonzero(x):
            x = x.flip(0)
            nonz = (x > 0)
            return ((nonz.cumsum(0) == 1) & nonz).max(0).indices
        # Clamp almost-zero values
        S[S < 1e-5] = 0.
        idx = first_nonzero(S)
        if idx != 0:
            # Delete zero values to reveal true rank
            S = S[:-idx]
        self.assertAlmostEqual(torch.norm(S_orig - S, p=float("Inf")).item(), 0., places=2)

    def test_lowrank(self):
        sizes = [(8,1), (8,3), (8,4), (8,8),
                 (7,1), (7,3), (7,4), (7,7),
                 (1,7), (2,7), (5,7), (1,8)]

        rs = [1, 3, 8]


        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.random.manual_seed(8888)
            for (n, k), r in itertools.product(sizes, rs):
                layer = nn.Linear(n, k, bias=False)
                r = min(n, k, r)
                LR = LowRank(size=layer.weight.size(), rank=r)
                P.register_parametrization(layer, LR, "weight")
                self.assertTrue(P.is_parametrized(layer, "weight"))
                print(layer)
                U_orig, S_orig, V_orig = LR.total_space.evaluate()
                self.assertIsOrthogonal(U_orig)
                self.assertIsOrthogonal(V_orig)
                self.assertHasSingularValues(layer.weight, S_orig)

                optim = torch.optim.SGD(layer.parameters(), lr=1.)
                ones = torch.ones(5, n)
                loss = layer(ones).sum()
                optim.zero_grad()
                loss.backward()
                optim.step()

                U_orig, S_orig, V_orig = LR.total_space.evaluate()
                self.assertIsOrthogonal(U_orig)
                self.assertIsOrthogonal(V_orig)
                self.assertHasSingularValues(layer.weight, S_orig)

                loss = layer(ones).sum()
                optim.zero_grad()
                loss.backward()
                optim.step()

                U_orig, S_orig, V_orig = LR.total_space.evaluate()
                self.assertIsOrthogonal(U_orig)
                self.assertIsOrthogonal(V_orig)
                self.assertHasSingularValues(layer.weight, S_orig)
