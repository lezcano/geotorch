# Tests for the Stiefel manifold, grassmannian and SO(n)
from unittest import TestCase
from copy import deepcopy
import itertools

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

from geotorch.so import SO
from geotorch.stiefel import Stiefel, StiefelTall
from geotorch.grassmanian import Grassmanian, GrassmanianTall


class TestOrthogonal(TestCase):
    def test_stiefel(self):
        self._test_orthogonality(Stiefel, StiefelTall)

    def test_grassmanian(self):
        self._test_orthogonality(Grassmanian, GrassmanianTall)

    def _test_orthogonality(self, cls, cls_tall):
        r"""Test that we may instantiate the parametrizations and
        register them in modules of several sizes. Check that the
        results are orthogonal and equal in the three cases.
        """
        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.random.manual_seed(8888)
            for layers in self._test_layers(cls, cls_tall):
                # Check that the initialization of the layers is orthogonal
                for layer in layers:
                    layer.parametrizations.weight.uniform_init_()
                    self.assertIsOrthogonal(layer.weight)
                    self.assertIsOrthogonal(layer.parametrizations.weight.base)

                # Make the initialization the same
                X = (
                    layers[0].weight.t()
                    if layers[0].parametrizations.weight.transpose
                    else layers[0].weight
                )
                for layer in layers[1:]:
                    with torch.no_grad():
                        layer.parametrizations.weight.base.copy_(X)
                    self.assertAlmostEqual(
                        torch.norm(layers[0].weight - layer.weight).item(),
                        0.0,
                        places=5,
                    )
                    self.assertIsOrthogonal(layer.parametrizations.weight.base)

                if isinstance(layers[0], nn.Linear):
                    input_ = torch.rand(5, layers[0].in_features)
                elif isinstance(layers[0], nn.Conv2d):
                    # batch x in_channel x in_length x in_width
                    input_ = torch.rand(6, layers[0].in_channels, 9, 8)

                results = []
                for layer in layers:
                    print(layer)
                    # Take one SGD step
                    optim = torch.optim.SGD(layer.parameters(), lr=0.1)
                    results.append([])

                    for _ in range(2):
                        with P.cached(layer):
                            self.assertIsOrthogonal(layer.weight)
                            loss = layer(input_).sum()
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        results[-1].append(layer.weight)
                    # If we change the base, the forward pass should give the same
                    prev_out = layer(input_)
                    layer.parametrizations.weight.update_base()
                    new_out = layer(input_)
                    self.assertAlmostEqual(
                        torch.norm(prev_out - new_out).abs().max().item(), 0.0, places=3
                    )

                self.assertPairwiseEqual(results)

    def _test_layers(self, cls, cls_tall):
        sizes = [
            (8, 1),
            (8, 3),
            (8, 4),
            (8, 8),
            (7, 1),
            (7, 3),
            (7, 4),
            (7, 7),
            (1, 7),
            (2, 7),
            (1, 1),
            (1, 2),
        ]
        trivs = ["expm"]

        for (n, k), triv in itertools.product(sizes, trivs):
            for layer in [nn.Linear(n, k), nn.Conv2d(n, 4, k)]:
                layers = []
                test_so = cls != Grassmanian and n == k
                layers.append(layer)
                layers.append(deepcopy(layer))
                if test_so:
                    layers.append(deepcopy(layer))
                    P.register_parametrization(
                        layers[2], "weight", SO(size=layers[2].weight.size(), triv=triv)
                    )

                P.register_parametrization(
                    layers[0], "weight", cls(size=layers[0].weight.size(), triv=triv)
                )
                P.register_parametrization(
                    layers[1],
                    "weight",
                    cls_tall(size=layers[1].weight.size(), triv=triv),
                )
                yield layers

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

    def assertPairwiseEqual(
        self, results,
    ):
        # Check pairwise equality
        with torch.no_grad():
            for l, m in itertools.combinations(range(len(results)), 2):
                # Compute the infinity norm
                norm0 = (results[l][0] - results[m][0]).abs().max().item()
                norm1 = (results[l][1] - results[m][1]).abs().max().item()
                self.assertAlmostEqual(norm0, 0.0, places=3)
                self.assertAlmostEqual(norm1, 0.0, places=2)
