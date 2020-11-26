from unittest import TestCase
import torch
import torch.nn as nn

import geotorch
from .test_positive_semidefinite import get_eigen
from .test_lowrank import get_svd


class TestConstraints(TestCase):
    def assertInSn(self, X, r=1.0):
        norm = X.norm(dim=-1)
        ones = r * torch.ones_like(norm)
        self.assertAlmostEqual(
            torch.norm(norm - ones, p=float("inf")).item(), 0.0, places=4
        )

    def assertIsOrthogonal(self, X):
        if X.size(-2) < X.size(-1):
            X = X.transpose(-2, -1)
        Id = torch.eye(X.size(-1))
        if X.dim() > 2:
            Id = Id.repeat(*(X.size()[:-2] + (1, 1)))
        norm = torch.norm(X.transpose(-2, -1) @ X - Id, dim=(-2, -1))
        self.assertTrue((norm < 1e-4).all())

    def assertIsSkew(self, X):
        self.assertAlmostEqual(
            torch.norm(X + X.transpose(-2, -1), p=float("inf")).item(), 0.0, places=6
        )

    def assertIsSymmetric(self, X):
        self.assertAlmostEqual(
            torch.norm(X - X.transpose(-2, -1), p=float("inf")).item(), 0.0, places=6
        )

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

    def assertHasEigenvalues(self, X, L_orig):
        L = torch.symeig(X).eigenvalues
        L_orig = torch.sort(L_orig.abs(), descending=False).values

        # Add the missign dimensions
        batch_dim = L.size()[:-1]
        pad_dim = batch_dim + (L.size(-1) - L_orig.size(-1),)
        L_orig = torch.cat([torch.zeros(pad_dim), L_orig], dim=-1)

        error = self.vector_error(L_orig, L)
        self.assertTrue((error < 1e-3).all())

    def test_sphere(self):
        net = nn.Linear(6, 3)
        geotorch.sphere(net, "weight")
        self.assertInSn(net.weight)
        geotorch.sphere(net, "bias", r=2.0)
        self.assertInSn(net.bias, r=2.0)

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

    def test_orthogonal(self):
        net = nn.Linear(6, 1)
        geotorch.orthogonal(net, "weight")
        self.assertIsOrthogonal(net.weight)
        net = nn.Linear(7, 4)
        geotorch.orthogonal(net, "weight")
        self.assertIsOrthogonal(net.weight)
        net = nn.Linear(7, 7)
        geotorch.orthogonal(net, "weight", triv="cayley")
        self.assertIsOrthogonal(net.weight)
        # Try to instantiate it in a vector rather than a matrix
        with self.assertRaises(ValueError):
            geotorch.orthogonal(net, "bias")

    def test_almost_orthogonal(self):
        def is_almost_orthogonal(net, lam):
            M = net.parametrizations.weight
            U_orig, S_orig, V_orig = get_svd(M)
            S_orig = 1.0 + lam * S_orig
            self.assertIsOrthogonal(U_orig)
            self.assertIsOrthogonal(V_orig)
            self.assertHasSingularValues(net.weight, S_orig)
            dist = torch.abs(S_orig - 1.0)
            self.assertTrue((dist < lam + 1e-6).all())

        net = nn.Linear(6, 2)
        geotorch.almost_orthogonal(net, "weight", lam=0.5)
        is_almost_orthogonal(net, 0.5)
        net = nn.Linear(7, 7)
        geotorch.almost_orthogonal(net, "weight", lam=0.3, triv="cayley")
        is_almost_orthogonal(net, 0.3)
        geotorch.almost_orthogonal(net, "weight", lam=1.0, f="tanh", triv="cayley")
        is_almost_orthogonal(net, 1.0)
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

    def test_low_and_fixed_rank(self):
        def is_low_rank(net):
            M = net.parametrizations.weight
            U_orig, S_orig, V_orig = get_svd(M)
            self.assertIsOrthogonal(U_orig)
            self.assertIsOrthogonal(V_orig)
            self.assertHasSingularValues(net.weight, S_orig)

        for f_set in [geotorch.low_rank, geotorch.fixed_rank]:
            net = nn.Linear(4, 7)
            f_set(net, "weight", rank=3)
            is_low_rank(net)
            net = nn.Linear(3, 3)
            f_set(net, "weight", rank=2, triv="cayley")
            is_low_rank(net)
            with self.assertRaises(ValueError):
                f_set(net, "bias", rank=2)

        def f(x):
            return 1.0 + x * x

        net = nn.Linear(3, 3)
        geotorch.fixed_rank(net, "weight", rank=2, f=f, triv="cayley")
        is_low_rank(net)

    def test_invertible(self):
        def is_invertible(net):
            self.assertTrue(net.weight.det() > 0)

        net = nn.Linear(7, 7)
        geotorch.invertible(net, "weight")
        is_invertible(net)
        net = nn.Linear(3, 3)
        geotorch.invertible(net, "weight", triv="cayley")
        is_invertible(net)
        with self.assertRaises(ValueError):
            geotorch.invertible(net, "bias")

    def test_positive_semidefinite(self):
        def is_pssd_low_rank(net):
            M = net.parametrizations.weight
            Q_orig, L_orig = get_eigen(M)
            self.assertIsOrthogonal(Q_orig)
            self.assertHasEigenvalues(net.weight, L_orig)

        for f_set in [
            geotorch.positive_semidefinite_low_rank,
            geotorch.positive_semidefinite_fixed_rank,
        ]:
            net = nn.Linear(7, 7)
            f_set(net, "weight", rank=3)
            is_pssd_low_rank(net)
            net = nn.Linear(3, 3)
            f_set(net, "weight", rank=2, triv="cayley")
            is_pssd_low_rank(net)
            with self.assertRaises(ValueError):
                f_set(net, "bias", rank=2)

        for f_set in [geotorch.positive_definite, geotorch.positive_semidefinite]:
            net = nn.Linear(7, 7)
            f_set(net, "weight")
            is_pssd_low_rank(net)
            net = nn.Linear(3, 3)
            f_set(net, "weight", triv="cayley")
            is_pssd_low_rank(net)
            with self.assertRaises(ValueError):
                f_set(net, "bias")
