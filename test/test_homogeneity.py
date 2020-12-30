# Tests for the interface of all the manifolds to be homogeneous
from unittest import TestCase
import itertools

import torch
import torch.nn as nn
import geotorch.parametrize as P

from geotorch.so import SO
from geotorch.stiefel import Stiefel
from geotorch.grassmannian import Grassmannian
from geotorch.lowrank import LowRank
from geotorch.fixedrank import FixedRank
from geotorch.psd import PSD
from geotorch.pssd import PSSD
from geotorch.pssdlowrank import PSSDLowRank
from geotorch.pssdfixedrank import PSSDFixedRank
from geotorch.glp import GLp
from geotorch.almostorthogonal import AlmostOrthogonal
from geotorch.sphere import Sphere, SphereEmbedded

from geotorch.utils import update_base


class TestHomogeneous(TestCase):
    def sizes(self, square):
        sizes_sq = [(i, i) for i in range(1, 11)]
        sizes_non_sq = [
            (i, j) for i, j in itertools.product(range(1, 5), range(1, 5)) if i != j
        ]
        sizes_non_sq += [(1, 7), (2, 7), (1, 8), (2, 8), (7, 1), (7, 2), (8, 1), (8, 2)]
        if square:
            return sizes_sq
        else:
            return sizes_sq + sizes_non_sq

    def ranks(self):
        return [{"rank": rank} for rank in [1, 3, 7]]

    def lambdas(self):
        return [{"lam": lam} for lam in [0.0, 0.5, 1.0]]

    def test_so(self):
        self._test_matrix_manifold(
            itertools.product(
                [SO],
                [{"distribution": "uniform"}, {"distribution": "torus"}],
                [{}],
                self.sizes(square=True),
            )
        )

    def test_orthogonal(self):
        self._test_matrix_manifold(
            itertools.product(
                [Stiefel, Grassmannian],
                [{"distribution": "uniform"}, {"distribution": "torus"}],
                [{}],
                self.sizes(square=False),
            )
        )

    def test_rank(self):
        self._test_matrix_manifold(
            itertools.product(
                [LowRank, FixedRank],
                [{"factorized": True}, {"factorized": False}],
                self.ranks(),
                self.sizes(square=False),
            )
        )

    def test_psd_and_glp(self):
        self._test_matrix_manifold(
            itertools.product(
                [PSD, PSSD, GLp],
                [{"factorized": True}, {"factorized": False}],
                [{}],
                self.sizes(square=True),
            )
        )

    def test_pssd_rank(self):
        self._test_matrix_manifold(
            itertools.product(
                [PSSDLowRank, PSSDFixedRank],
                [{"factorized": True}, {"factorized": False}],
                self.ranks(),
                self.sizes(square=True),
            )
        )

    def test_almost_orthogonal(self):
        self._test_matrix_manifold(
            itertools.product(
                [AlmostOrthogonal],
                [
                    {"factorized": True, "distribution": "uniform"},
                    {"factorized": False, "distribution": "uniform"},
                    {"factorized": True, "distribution": "torus"},
                    {"factorized": False, "distribution": "torus"},
                ],
                self.lambdas(),
                self.sizes(square=True),
            )
        )

    def _test_matrix_manifold(self, man):
        # man7 = itertools.product(
        #    sizes_sq,
        #    [Sphere, SphereEmbedded],
        #    [{}],
        #    [{}]
        # )
        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.random.manual_seed(8888)
            for M, args, args_constr, size in man:
                if "rank" in args_constr and args_constr["rank"] > min(size):
                    continue
                # Test Linear
                layer = nn.Linear(*size)
                input_ = torch.rand(3, size[0])
                old_size = layer.weight.size()
                P.register_parametrization(
                    layer, "weight", M(size=layer.weight.size(), **args_constr)
                )
                self.assertEqual(old_size, layer.weight.size(), msg=f"{layer}")
                self._test_interface(layer, args, input_)

                # Test Convolutionar (tensorial)
                layer = nn.Conv2d(5, 4, size)
                input_ = torch.rand(6, 5, 2 * size[0] + 1, size[1] + 3)
                old_size = layer.weight.size()
                P.register_parametrization(
                    layer, "weight", M(size=layer.weight.size(), **args_constr)
                )
                self.assertEqual(old_size, layer.weight.size(), msg=f"{layer}")
                self._test_interface(layer, args, input_)

    def matrix_from_factor_svd(self, U, S, V):
        Vt = V.transpose(-2, -1)
        # Multiply the three of them, S as a diagonal matrix
        return U @ (S.unsqueeze(-1).expand_as(Vt) * Vt)

    def matrix_from_factor_eigen(self, L, Q):
        Qt = Q.transpose(-2, -1)
        # Multiply the three of them as Q\LambdaQ^T
        return Q @ (L.unsqueeze(-1).expand_as(Qt) * Qt)

    def matrix_from_factor(self, X, M):
        transpose = hasattr(M, "transposed") and M.transposed
        if not isinstance(X, tuple):
            return X
        elif len(X) == 2:
            X = self.matrix_from_factor_eigen(X[0], X[1])
        else:
            X = self.matrix_from_factor_svd(X[0], X[1], X[2])
        if transpose:
            return X.transpose(-2, -1)
        else:
            return X

    def _test_interface(self, layer, args, input_):
        msg = f"{layer}\n{args}"
        M = layer.parametrizations.weight[0]
        initial_size = layer.weight.size()
        X = M.sample(**args)
        self.assertTrue(M.in_manifold(X), msg=msg)
        X_matrix = self.matrix_from_factor(X, M)
        layer.weight = X
        with P.cached():
            # Size does not change
            self.assertEqual(initial_size, layer.weight.size(), msg=msg)
            # Tha initialisation initialisation is equal to what we passed
            self.assertTrue(torch.allclose(layer.weight, X_matrix, atol=1e-6), msg=msg)

        # Take a couple SGD steps
        optim = torch.optim.SGD(layer.parameters(), lr=1e-3)
        for i in range(3):
            with P.cached():
                loss = layer(input_).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            # The layer stays in the manifold while being optimised
            self.assertTrue(M.in_manifold(layer.weight), msg=f"i:{i}\n" + msg)

        with P.cached():
            weight_old = layer.weight
            update_base(layer, "weight")
            # After changing the base, the weight stays the same
            self.assertTrue(
                torch.allclose(layer.weight, weight_old, atol=1e-6), msg=msg
            )
