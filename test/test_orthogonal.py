# Tests for the Stiefel manifold, grassmannian and SO(n)
from unittest import TestCase
from copy import deepcopy
import itertools

import torch
import torch.nn as nn

import geotorch.parametrize as P
from geotorch.so import SO, torus_init_, uniform_init_
from geotorch.stiefel import Stiefel, StiefelTall
from geotorch.grassmannian import Grassmannian, GrassmannianTall


class TestOrthogonal(TestCase):
    def test_orthogonality_stiefel(self):
        self._test_orthogonality(Stiefel, StiefelTall)

    def test_initialization_stiefel(self):
        self._test_initializations(Stiefel, StiefelTall)

    def test_constructor_stiefel(self):
        self._test_constructor(Stiefel, StiefelTall)

    def test_custom_trivialization_stiefel(self):
        self._test_custom_trivialization(Stiefel)

    def test_orthogonality_grassmannian(self):
        self._test_orthogonality(Grassmannian, GrassmannianTall)

    def test_initializations_grassmannian(self):
        self._test_initializations(Grassmannian, GrassmannianTall)

    def test_constructors_grassmannian(self):
        self._test_constructor(Grassmannian, GrassmannianTall)

    def test_custom_trivialization_grassmannian(self):
        self._test_custom_trivialization(Grassmannian)

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
                        with P.cached():
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

    def _test_custom_trivialization(self, cls):
        def qr(X):
            return torch.qr(X).Q

        # Note that qr is not an analytic function. As such, it may not be used with StiefelTall
        layer = nn.Linear(5, 3)
        P.register_parametrization(
            layer, "weight", cls(size=layer.weight.size(), triv=qr)
        )

        optim = torch.optim.SGD(layer.parameters(), lr=0.1)
        input_ = torch.rand(5, layer.in_features)
        for _ in range(2):
            with P.cached():
                self.assertIsOrthogonal(layer.weight)
                loss = layer(input_).sum()
            optim.zero_grad()
            loss.backward()
            optim.step()

    def _test_constructor(self, cls, cls_tall):
        with self.assertRaises(ValueError):
            cls(size=(3, 3), triv="wrong")

        with self.assertRaises(ValueError):
            cls_tall(size=(3, 3), triv="wrong")

        with self.assertRaises(ValueError):
            SO(size=(3, 3), triv="wrong")

        try:
            cls(size=(3, 3), triv=lambda: 3)
        except ValueError:
            self.fail("{} raised ValueError unexpectedly!".format(cls))

        try:
            cls_tall(size=(3, 3), triv=lambda: 3)
        except ValueError:
            self.fail("{} raised ValueError unexpectedly!".format(cls_tall))

        # Try to instantiate it in a vector rather than a matrix
        with self.assertRaises(ValueError):
            cls(size=(7,))

        with self.assertRaises(ValueError):
            cls_tall(size=(7,))

        with self.assertRaises(ValueError):
            SO(size=(7,))

    def _test_initializations(self, cls, cls_tall):
        for layers in self._test_layers(cls, cls_tall):
            for layer in layers:
                p = layer.parametrizations.weight
                p.uniform_init_()
                self.assertIsOrthogonal(layer.weight)
                if isinstance(p, SO):
                    p.torus_init_()
                    self.assertIsOrthogonal(layer.weight)
        t = torch.empty(3, 4)
        uniform_init_(t)
        self.assertIsOrthogonal(t)
        t.zero_()
        # torus_init_ is just available for square matrices
        with self.assertRaises(ValueError):
            torus_init_(t)

        # Number of dimensions < 2 should raise an error
        t = torch.empty(3)
        with self.assertRaises(ValueError):
            torus_init_(t)
        with self.assertRaises(ValueError):
            uniform_init_(t)
        t = torch.empty(0)
        with self.assertRaises(ValueError):
            torus_init_(t)
        with self.assertRaises(ValueError):
            uniform_init_(t)

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
                test_so = cls != Grassmannian and n == k
                layers.append(layer)
                layers.append(deepcopy(layer))
                if test_so:
                    layers.append(deepcopy(layer))
                    P.register_parametrization(
                        layers[2], "weight", SO(size=layers[2].weight.size(), triv=triv)
                    )
                elif n != k:
                    # If it's not square it should throw
                    with self.assertRaises(ValueError):
                        size = layer.weight.size()[:-2] + (n, k)
                        SO(size=size, triv=triv)

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
        if X.size(-2) < X.size(-1):
            X = X.transpose(-2, -1)
        Id = torch.eye(X.size(-1))
        if X.dim() > 2:
            Id = Id.repeat(*(X.size()[:-2] + (1, 1)))
        norm = torch.norm(X.transpose(-2, -1) @ X - Id, dim=(-2, -1))
        self.assertTrue((norm < 1e-4).all())

    def assertPairwiseEqual(self, results):
        # Check pairwise equality
        with torch.no_grad():
            for i, j in itertools.combinations(range(len(results)), 2):
                # Compute the infinity norm
                norm0 = (results[i][0] - results[j][0]).abs().max().item()
                norm1 = (results[i][1] - results[j][1]).abs().max().item()
                self.assertAlmostEqual(norm0, 0.0, places=4)
                self.assertAlmostEqual(norm1, 0.0, places=2)
