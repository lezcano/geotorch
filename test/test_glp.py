from unittest import TestCase

import torch
import torch.nn as nn

import geotorch.parametrize as P
from geotorch.glp import GLp
from geotorch.utils import update_base


class TestGLp(TestCase):
    def log_scalar_error(self, X, Y):
        # Error relative to the size in infinity norm
        X_inf = X.abs()
        abs_error = (Y - X).abs()
        error = abs_error / X_inf
        # If the magnitude is too small, we use the absolute error
        small = X < 1e-5
        error[small] = (Y.exp() - X.exp()).abs()[small]
        return error

    def assertPositiveDet(self, X):
        logdet = X.logdet()

        # Assert that the determinants are positive
        self.assertTrue(not torch.isnan(logdet).any() and not torch.isinf(logdet).any())

    def test_GLp(self):
        sizes = [1, 2, 3, 4, 7, 8]

        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.random.manual_seed(8888)
            for n in sizes:
                for layer in [nn.Linear(n, n), nn.Conv2d(7, 4, n)]:
                    print("GLp({}) on {}".format(n, str(layer)))
                    M = GLp(size=layer.weight.size())
                    P.register_parametrization(layer, "weight", M)
                    with torch.no_grad():
                        layer.parametrizations.weight.original.zero_()

                    self.assertTrue(P.is_parametrized(layer, "weight"))
                    self.assertPositiveDet(layer.weight)

                    optim = torch.optim.SGD(layer.parameters(), lr=0.1)
                    if isinstance(layer, nn.Linear):
                        input_ = torch.rand(5, n)
                    elif isinstance(layer, nn.Conv2d):
                        # batch x in_channel x in_length x in_width
                        input_ = torch.rand(6, 7, 9, 8)

                    for i in range(2):
                        print(i)
                        loss = layer(input_).sum()
                        optim.zero_grad()
                        loss.backward()
                        optim.step()

                        self.assertPositiveDet(layer.weight)

                    # Test update_base
                    prev_out = layer(input_)
                    update_base(layer, "weight")
                    new_out = layer(input_)
                    self.assertAlmostEqual(
                        torch.norm(prev_out - new_out).abs().max().item(),
                        0.0,
                        places=3,
                    )

    def test_GLp_errors(self):
        # Non square
        with self.assertRaises(ValueError):
            GLp(size=(4, 3))
        # Try to instantiate it in a vector rather than a matrix
        with self.assertRaises(ValueError):
            GLp(size=(5,))
