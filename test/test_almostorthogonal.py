from unittest import TestCase
import itertools

import torch
import torch.nn as nn

import geotorch.parametrize as P
from geotorch.almostorthogonal import AlmostOrthogonal
from geotorch.so import uniform_init_
from geotorch.utils import update_base
from .test_lowrank import get_svd


class TestLowRank(TestCase):
    def assertIsOrthogonal(self, X):
        if X.size(-2) < X.size(-1):
            X = X.transpose(-2, -1)
        Id = torch.eye(X.size(-1), device=X.device, dtype=X.dtype)
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

    def assertHasSingularValues(self, X, S_orig, lam):
        _, S, _ = torch.svd(X)
        # Sort the singular values from our parametrization
        S_orig = torch.sort(torch.abs(S_orig), descending=True).values

        error = self.vector_error(S_orig, S)
        self.assertTrue((error < 1e-3).all())

        dist = torch.abs(S_orig - 1.0)
        self.assertTrue((dist < lam + 1e-6).all())

    def test_almostorthogonal(self):
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

        lams = [0.0, 0.5, 1.0]
        fs = ["scaled_sigmoid", "sin", "tanh"]

        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.random.manual_seed(8888)
            for (n, k), lam, f in itertools.product(sizes, lams, fs):
                for layer in [nn.Linear(n, k), nn.Conv2d(n, 4, k)]:
                    cls = AlmostOrthogonal
                    print(
                        "{}({}, {}, {}, {}) on {}".format(
                            cls.__name__, n, k, lam, f, str(layer)
                        )
                    )
                    M = cls(size=layer.weight.size(), lam=lam, f=f)
                    P.register_parametrization(layer, "weight", M)
                    layer.weight = uniform_init_(layer.weight)
                    self.assertTrue(P.is_parametrized(layer, "weight"))
                    U_orig, S_orig, V_orig = get_svd(layer.parametrizations.weight)
                    S_orig = 1.0 + lam * S_orig
                    self.assertIsOrthogonal(U_orig)
                    self.assertIsOrthogonal(V_orig)
                    self.assertHasSingularValues(layer.weight, S_orig, lam)

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

                        U_orig, S_orig, V_orig = get_svd(layer.parametrizations.weight)
                        S_orig = 1.0 + lam * S_orig
                        self.assertIsOrthogonal(U_orig)
                        self.assertIsOrthogonal(V_orig)
                        self.assertHasSingularValues(layer.weight, S_orig, lam)

                    # Test update_base
                    prev_out = layer(input_)
                    update_base(layer, "weight")
                    new_out = layer(input_)
                    self.assertAlmostEqual(
                        torch.norm(prev_out - new_out).abs().max().item(),
                        0.0,
                        places=3,
                    )

    def test_almostorthogonal_errors(self):
        with self.assertRaises(ValueError):
            AlmostOrthogonal(size=(5,), lam=1.0)
        # Not a predefined f
        with self.assertRaises(ValueError):
            AlmostOrthogonal(size=(5, 4), lam=1, f="fail")
        # Not callable
        with self.assertRaises(ValueError):
            AlmostOrthogonal(size=(5, 4), lam=1, f=3.0)
        # But a callable should work
        AlmostOrthogonal(size=(5, 4), lam=0.5)
        # Too large a lambda
        with self.assertRaises(ValueError):
            AlmostOrthogonal(size=(5, 4), lam=2)
        # Or too small
        with self.assertRaises(ValueError):
            AlmostOrthogonal(size=(5, 4), lam=-1.0)
