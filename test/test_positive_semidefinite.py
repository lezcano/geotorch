from unittest import TestCase
import itertools

import torch
import torch.nn as nn

import geotorch.parametrize as P

from geotorch.constructions import ProductManifold
from geotorch.pssdlowrank import PSSDLowRank
from geotorch.pssdfixedrank import PSSDFixedRank
from geotorch.pssd import PSSD
from geotorch.psd import PSD
from geotorch.utils import update_base


def get_eigen(M):
    size = M.original.size()
    X = M.original if size[-2] >= size[-1] else M.original.transpose(-2, -1)
    Q, L = ProductManifold.forward(M, M.frame(X))
    if hasattr(M, "f"):
        L = M.f(L)
    return Q, L


class TestPSSDLowRank(TestCase):
    def assertIsSymmetric(self, X):
        self.assertAlmostEqual(
            torch.norm(X - X.transpose(-2, -1), p=float("inf")).item(), 0.0, places=5
        )

    def assertIsOrthogonal(self, X):
        if X.size(-2) < X.size(-1):
            X = X.transpose(-2, -1)
        Id = torch.eye(X.size(-1))
        if X.dim() > 2:
            Id = Id.repeat(*(X.size()[:-2] + (1, 1)))
        norm = torch.norm(X.transpose(-2, -1) @ X - Id, dim=(-2, -1))
        self.assertTrue((norm < 4e-3).all())

    def vector_error(self, X, Y):
        # Error relative to the size in infinity norm
        X_inf = X.abs().max(dim=-1).values
        abs_error = (Y - X).abs().max(dim=-1).values
        error = abs_error / X_inf
        # Unless X is very small, if so we do absolute error
        small = X_inf < 1e-5
        error[small] = abs_error[small]
        return error

    def assertHasEigenvalues(self, X, L_orig):
        L = torch.symeig(X).eigenvalues
        L_orig = torch.sort(L_orig.abs(), descending=False).values

        # Add the missign dimensions
        batch_dim = L.size()[:-1]
        pad_dim = batch_dim + (L.size(-1) - L_orig.size(-1),)
        L_orig = torch.cat([torch.zeros(pad_dim), L_orig], dim=-1)

        error = self.vector_error(L_orig, L)
        self.assertTrue((error < 1e-3).all())

    def test_positive_semidefinite(self):
        sizes = [
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (7, 7),
            (8, 8),
        ]

        rs = [1, 3, 4]

        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.random.manual_seed(8888)
            for cls in [PSSDLowRank, PSSDFixedRank, PSSD, PSD]:
                for (n, k), r in itertools.product(sizes, rs):
                    for layer in [nn.Linear(n, k), nn.Conv2d(n, 4, k)]:
                        needs_rank = cls in [PSSDLowRank, PSSDFixedRank]
                        if not needs_rank and r != 1:
                            continue
                        # Only show r when we have a non-full rank
                        print(
                            "{}({}, {}{}) on {}".format(
                                cls.__name__,
                                n,
                                k,
                                ", {}".format(r) if needs_rank else "",
                                str(layer),
                            )
                        )
                        r = min(n, k, r)
                        if needs_rank:
                            M = cls(size=layer.weight.size(), rank=r)
                        else:
                            M = cls(size=layer.weight.size())
                        P.register_parametrization(layer, "weight", M)
                        self.assertTrue(P.is_parametrized(layer, "weight"))
                        Q_orig, L_orig = get_eigen(M)
                        self.assertIsOrthogonal(Q_orig)
                        self.assertIsSymmetric(layer.weight)
                        self.assertHasEigenvalues(layer.weight, L_orig)

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

                            Q_orig, L_orig = get_eigen(M)
                            self.assertIsOrthogonal(Q_orig)
                            self.assertIsSymmetric(layer.weight)
                            self.assertHasEigenvalues(layer.weight, L_orig)

                        # Test update_base
                        prev_out = layer(input_)
                        update_base(layer, "weight")
                        new_out = layer(input_)
                        self.assertAlmostEqual(
                            torch.norm(prev_out - new_out).abs().max().item(),
                            0.0,
                            places=3,
                        )

    def test_positive_semidefinite_errors(self):
        for cls in [PSSDLowRank, PSSDFixedRank]:
            # rank always has to be 1 <= rank <= n
            with self.assertRaises(ValueError):
                cls(size=(4, 4), rank=5)
            with self.assertRaises(ValueError):
                cls(size=(3, 3), rank=0)
            # Instantiate it in a non-square matrix
            with self.assertRaises(ValueError):
                cls(size=(3, 6), rank=2)
            # Try to instantiate it in a vector rather than a matrix
            with self.assertRaises(ValueError):
                cls(size=(5,), rank=1)

        for cls in [PSSD, PSD]:
            # Try to instantiate it in a vector rather than a matrix
            with self.assertRaises(ValueError):
                cls(size=(5,))
            # Or a non-square
            with self.assertRaises(ValueError):
                cls(size=(5, 3))

        # Pass a non-callable object
        with self.assertRaises(ValueError):
            PSSDFixedRank(size=(5, 2), rank=1, f=3)
        # Or the wrong string
        with self.assertRaises(ValueError):
            PSSDFixedRank(size=(5, 3), rank=2, f="fail")
        # Same with PSD
        with self.assertRaises(ValueError):
            PSD(size=(5, 2), f=3)
        with self.assertRaises(ValueError):
            PSD(size=(5, 3), f="fail")
