from unittest import TestCase
import torch
import torch.nn as nn

import geotorch


class TestConstraints(TestCase):
    def assertInSn(self, X, r=1.0):
        norm = X.norm(dim=-1)
        ones = r * torch.ones_like(norm)
        self.assertAlmostEqual(
            torch.norm(norm - ones, p=float("inf")).item(), 0.0, places=4
        )

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

    def assertIsSkew(self, X):
        self.assertAlmostEqual(
            torch.norm(X + X.transpose(-2, -1), p=float("inf")).item(), 0.0, places=6
        )

    def assertIsSymmetric(self, X):
        self.assertAlmostEqual(
            torch.norm(X - X.transpose(-2, -1), p=float("inf")).item(), 0.0, places=6
        )

    def test_sphere(self):
        net = nn.Linear(6, 3)
        geotorch.sphere(net, "weight")
        self.assertInSn(net.weight)
        geotorch.sphere(net, "bias", r=2.0)
        self.assertInSn(net.bias, r=2.0)

    def test_orthogonal(self):
        net = nn.Linear(6, 2)
        geotorch.orthogonal(net, "weight")
        self.assertIsOrthogonal(net.weight)
        net = nn.Linear(7, 7)
        geotorch.orthogonal(net, "weight", triv="cayley")
        self.assertIsOrthogonal(net.weight)
        # Try to instantiate it in a vector rather than a matrix
        with self.assertRaises(ValueError):
            geotorch.orthogonal(net, "bias")

    def test_grassmannian(self):
        net = nn.Linear(6, 2)
        geotorch.grassmannian(net, "weight")
        self.assertIsOrthogonal(net.weight)
        net = nn.Linear(7, 7)
        geotorch.grassmannian(net, "weight", triv="cayley")
        self.assertIsOrthogonal(net.weight)
        # Try to instantiate it in a vector rather than a matrix
        with self.assertRaises(ValueError):
            geotorch.grassmannian(net, "bias")

    def test_lowrank(self):
        net = nn.Linear(4, 7)
        geotorch.lowrank(net, "weight", rank=3)
        net = nn.Linear(3, 3)
        geotorch.lowrank(net, "weight", rank=2)

    def test_skew(self):
        net = nn.Linear(4, 4)
        geotorch.skew(net, "weight")
        self.assertIsSkew(net.weight)
        net = nn.Linear(3, 3)
        geotorch.skew(net, "weight", lower=False)
        self.assertIsSkew(net.weight)

    def test_symmetric(self):
        net = nn.Linear(4, 4)
        geotorch.symmetric(net, "weight")
        self.assertIsSymmetric(net.weight)
        net = nn.Linear(3, 3)
        geotorch.symmetric(net, "weight", lower=False)
        self.assertIsSymmetric(net.weight)
