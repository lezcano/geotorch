# Integration tests for all the manifold
from unittest import TestCase
import itertools
import types

import torch
import torch.nn as nn
import geotorch.parametrize as P

import geotorch
from geotorch.skew import Skew
from geotorch.symmetric import Symmetric
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


def dicts_product(**kwargs):
    """Returns a product of all the lists of the keys"""
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


class TestIntegration(TestCase):
    def sizes(self, square):
        sizes = [(i, i) for i in range(1, 11)]
        if not square:
            sizes.extend(
                [
                    (i, j)
                    for i, j in itertools.product(range(1, 5), range(1, 5))
                    if i != j
                ]
            )
            sizes.extend(
                [(1, 7), (2, 7), (1, 8), (2, 8), (7, 1), (7, 2), (8, 1), (8, 2)]
            )
        if torch.cuda.is_available():
            sizes.extend([(256, 256), (512, 512)])
            if not square:
                sizes.extend([(256, 128), (128, 512), (1024, 512)])
        return sizes

    def ranks(self):
        return [1, 3, 7]

    def lambdas(self):
        return [0.01, 0.5, 1.0]

    def radii(self):
        return [0.01, 1.0, 2.0, 10.0]

    def devices(self):
        if torch.cuda.is_available():
            return [torch.device("cuda")]
        else:
            return [torch.device("cpu")]

    def test_vector_spaces(self):
        self._test_manifolds(
            [Skew, Symmetric, geotorch.skew, geotorch.symmetric],
            [],
            dicts_product(lower=[True, False]),
            self.devices(),
            self.sizes(square=True),
            initialize=False,
        )

    def test_so(self):
        self._test_manifolds(
            [SO],
            dicts_product(distribution=["uniform", "torus"]),
            [{}],
            self.devices(),
            self.sizes(square=True),
        )

    def test_orthogonal(self):
        self._test_manifolds(
            [Stiefel, Grassmannian, geotorch.orthogonal, geotorch.grassmannian],
            dicts_product(distribution=["uniform", "torus"]),
            dicts_product(triv=["expm", "cayley"]),
            self.devices(),
            self.sizes(square=False),
        )

    def test_rank(self):
        self._test_manifolds(
            [LowRank, FixedRank, geotorch.low_rank, geotorch.fixed_rank],
            [{}],
            dicts_product(rank=self.ranks()),
            self.devices(),
            self.sizes(square=False),
        )

    def test_psd_and_glp(self):
        self._test_manifolds(
            [
                PSD,
                PSSD,
                GLp,
                geotorch.positive_definite,
                geotorch.positive_semidefinite,
                geotorch.invertible,
            ],
            [{}],
            [{}],
            self.devices(),
            self.sizes(square=True),
        )

    def test_pssd_rank(self):
        self._test_manifolds(
            [
                PSSDLowRank,
                PSSDFixedRank,
                geotorch.positive_semidefinite_low_rank,
                geotorch.positive_semidefinite_fixed_rank,
            ],
            [{}],
            dicts_product(rank=self.ranks()),
            self.devices(),
            self.sizes(square=True),
        )

    def test_almost_orthogonal(self):
        self._test_manifolds(
            [AlmostOrthogonal, geotorch.almost_orthogonal],
            dicts_product(distribution=["uniform", "torus"]),
            dicts_product(lam=self.lambdas(), f=list(AlmostOrthogonal.fs.keys())),
            self.devices(),
            self.sizes(square=True),
        )

    def test_sphere(self):
        self._test_manifolds(
            [Sphere, SphereEmbedded, geotorch.sphere],
            [{}],
            dicts_product(radius=self.radii()),
            self.devices(),
            self.sizes(square=False),
        )

    def _test_manifolds(
        self, Ms, argss_sample, argss_constr, devices, sizes, initialize=False
    ):
        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.random.manual_seed(8888)
            for M, args_sample, args_constr, device, size in itertools.product(
                Ms, argss_sample, argss_constr, devices, sizes
            ):
                if "rank" in args_constr and args_constr["rank"] > min(size):
                    continue
                self._test_manifold(
                    M, args_sample, args_constr, device, size, initialize
                )

    def _test_manifold(self, M, args_sample, args_constr, device, size, initialize):
        inputs = [torch.rand(3, size[0], device=device)]
        layers = [nn.Linear(*size, device=device)]
        # Just test on convolution for small layers, otherwise it takes too long
        if min(size) < 100:
            inputs.append(torch.rand(6, 5, size[0] + 7, size[1] + 3, device=device))
            layers.append(nn.Conv2d(5, 4, size, device=device))

        for input_, layer in zip(inputs, layers):
            old_size = layer.weight.size()
            # Somewhat dirty but will do
            if isinstance(M, types.FunctionType):
                M(layer, "weight", **args_constr)
            else:
                # initialize the weight first (annoying)
                M_ = M(size=layer.weight.size(), **args_constr).to(device)
                X = M_.sample(**args_sample)
                with torch.no_grad():
                    layer.weight.copy_(X)
                P.register_parametrization(layer, "weight", M_)
            # Check that it does not change the size of the layer
            self.assertEqual(old_size, layer.weight.size(), msg=f"{layer}")
            self._test_training(layer, args_sample, input_, initialize)

    def _test_training(self, layer, args_sample, input_, initialize):
        msg = f"{layer}\n{args_sample}"
        M = layer.parametrizations.weight[0]
        if initialize:
            initial_size = layer.weight.size()
            X = M.sample(**args_sample)
            self.assertTrue(M.in_manifold(X), msg=msg)
            layer.weight = X
            with P.cached():
                # Compute the product if it is factorized
                # The sampled matrix should not have a gradient
                self.assertFalse(X.requires_grad)
                # Size does not change
                self.assertEqual(initial_size, layer.weight.size(), msg=msg)
                # Tha initialisation initialisation is equal to what we passed
                self.assertTrue(torch.allclose(layer.weight, X, atol=1e-5), msg=msg)

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
