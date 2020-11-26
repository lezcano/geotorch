from unittest import TestCase
import itertools

import torch
import torch.nn as nn
import geotorch.parametrize as P

from geotorch.constructions import ProductManifold
from geotorch.lowrank import LowRank
from geotorch.fixedrank import FixedRank
from geotorch.utils import update_base


def get_svd(M):
    # Very hacky, but will do for now
    size = M.original.size()
    X = M.original if size[-2] >= size[-1] else M.original.transpose(-2, -1)
    U, S, V = ProductManifold.forward(M, M.frame(X))
    if hasattr(M, "f"):
        S = M.f(S)
    return U, S, V


class TestLowRank(TestCase):
    def assertIsOrthogonal(self, X):
        if X.size(-2) < X.size(-1):
            X = X.transpose(-2, -1)
        Id = torch.eye(X.size(-1))
        if X.dim() > 2:
            Id = Id.repeat(*(X.size()[:-2] + (1, 1)))
        norm = torch.norm(X.transpose(-2, -1) @ X - Id, dim=(-2, -1))
        self.assertTrue((norm < 1e-3).all())

    def vector_error(self, X, Y):
        # Error relative to the size in infinity norm
        X_inf = X.abs().max(dim=-1).values
        abs_error = (Y - X).abs().max(dim=-1).values
        error = abs_error / X_inf
        # Unless X is very small, if so we do absolute error
        small = X_inf < 1e-5
        error[small] = abs_error[small]
        return error

    def assertHasSingularValues(self, X, S_orig):
        _, S, _ = torch.svd(X)
        # Sort the singular values from our parametrization
        S_orig = torch.sort(torch.abs(S_orig), descending=True).values
        # Add the missign dimensions
        batch_dim = S.size()[:-1]
        pad_dim = batch_dim + (S.size(-1) - S_orig.size(-1),)
        S_orig = torch.cat([S_orig, torch.zeros(pad_dim)], dim=-1)

        error = self.vector_error(S_orig, S)
        self.assertTrue((error < 1e-3).all())

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
            for cls in [FixedRank, LowRank]:
                for (n, k), r in itertools.product(sizes, rs):
                    for layer in [nn.Linear(n, k), nn.Conv2d(n, 4, k)]:
                        print(
                            "{}({}, {}, {}) on {}".format(
                                cls.__name__, n, k, r, str(layer)
                            )
                        )
                        r = min(n, k, r)
                        M = cls(size=layer.weight.size(), rank=r)
                        P.register_parametrization(layer, "weight", M)
                        self.assertTrue(P.is_parametrized(layer, "weight"))
                        U_orig, S_orig, V_orig = get_svd(M)
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

                            U_orig, S_orig, V_orig = get_svd(M)
                            self.assertIsOrthogonal(U_orig)
                            self.assertIsOrthogonal(V_orig)
                            self.assertHasSingularValues(layer.weight, S_orig)

                        # Test update_base
                        prev_out = layer(input_)
                        update_base(layer, "weight")
                        new_out = layer(input_)
                        self.assertAlmostEqual(
                            torch.norm(prev_out - new_out).abs().max().item(),
                            0.0,
                            places=3,
                        )

    def test_lowrank_errors(self):
        # rank always has to be <= min(n, k)
        for cls in [LowRank, FixedRank]:
            with self.assertRaises(ValueError):
                cls(size=(4, 3), rank=5)
            with self.assertRaises(ValueError):
                cls(size=(2, 3), rank=3)
            # Try to instantiate it in a vector rather than a matrix
            with self.assertRaises(ValueError):
                cls(size=(5,), rank=1)

        # On a non-callable
        with self.assertRaises(ValueError):
            FixedRank(size=(5, 3), rank=2, f=3)
        # On the wrong string
        with self.assertRaises(ValueError):
            FixedRank(size=(5, 3), rank=2, f="wrong")
