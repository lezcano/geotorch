import math
from unittest import TestCase
import torch

from geotorch.linalg.expm import expm, taylor_approx


class TestLinalg(TestCase):
    def taylor(self, X, deg):
        assert X.size(-1) == X.size(-2)
        n = X.size(-1)
        Id = torch.eye(n, n, dtype=X.dtype, device=X.device)
        if X.ndimension() > 2:
            Id = Id.expand_as(X)
        acc = Id
        last = Id
        for i in range(1, deg + 1):
            last = last @ X / float(i)
            acc = acc + last
        return acc

    def scale_square(self, X):
        """
        Scale-squaring trick
        """
        norm = X.norm()
        if norm < 0.5:
            return self.taylor(X, 12)

        k = int(math.ceil(math.log2(float(norm)))) + 2
        X = X * (2 ** -k)
        E = self.taylor(X, 18)
        for _ in range(k):
            E = torch.mm(E, E)
        return E

    def assertIsCloseSquare(self, X, Y, places=4):
        self.assertEqual(X.ndimension(), 2)
        self.assertEqual(X.size(0), X.size(1))
        self.assertAlmostEqual(torch.norm(X - Y).item(), 0.0, places=places)

    def compare_f(self, f_batching, f_simple, allows_batches, dtype, gradients=False):
        # Test expm without batching
        for _ in range(8):
            A = torch.rand(10, 10, dtype=dtype)
            if gradients:
                G = torch.rand(10, 10, dtype=dtype)
                B1 = f_batching(A, G)
                B2 = f_simple(A, G)
                self.assertIsCloseSquare(B1, B2, places=2)
            else:
                B1 = f_batching(A)
                B2 = f_simple(A)
                self.assertIsCloseSquare(B1, B2, places=3)

        # Test batching
        for _ in range(3):
            len_shape = torch.randint(1, 4, (1,))
            shape_batch = torch.randint(1, 5, size=(len_shape,))
            shape = list(shape_batch) + [8, 8]
            A = torch.rand(*shape, dtype=dtype)
            if gradients:
                G = torch.rand(*shape, dtype=dtype)
                B1 = f_batching(A, G)
            else:
                B1 = f_batching(A)
            if allows_batches:
                B2 = f_simple(A)
                self.assertEqual(B1.size(), A.size())
                self.assertEqual(B2.size(), A.size())

                # sample a few coordinates and evaluate the equality
                # of those elements in the batch
                for _ in range(3):
                    coords = [
                        torch.randint(low=0, high=s, size=(1,)).item()
                        for s in shape_batch
                    ]
                    coords = coords + [...]
                    self.assertIsCloseSquare(B1[coords], B2[coords], places=3)
            else:
                # sample a few coordinates and evaluate the equality
                # of those elements in the batch
                for _ in range(3):
                    coords = [
                        torch.randint(low=0, high=s, size=(1,)).item()
                        for s in shape_batch
                    ]
                    coords = coords + [...]
                    if gradients:
                        self.assertIsCloseSquare(
                            B1[coords], f_simple(A[coords], G[coords]), places=2
                        )
                    else:
                        self.assertIsCloseSquare(
                            B1[coords], f_simple(A[coords]), places=3
                        )

    def test_expm(self):
        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.random.manual_seed(8888)
            # Test different Taylor approximations
            degs = [1, 2, 4, 8, 12, 18]
            for deg in degs:
                for dtype in [torch.float, torch.double]:
                    self.compare_f(
                        lambda X: taylor_approx(X, deg),
                        lambda X: self.taylor(X, deg),
                        allows_batches=True,
                        dtype=dtype,
                    )

            # Test the main function
            for dtype in [torch.float, torch.double]:
                self.compare_f(
                    expm, self.scale_square, allows_batches=False, dtype=dtype
                )

            # Test the gradients
            def diff(f):
                def wrap(A, G):
                    A.requires_grad_()
                    return torch.autograd.grad([f(A)], [A], [G])[0]

                return wrap

            for dtype in [torch.float, torch.double]:
                self.compare_f(
                    diff(expm),
                    diff(self.scale_square),
                    allows_batches=False,
                    dtype=dtype,
                    gradients=True,
                )

    def test_errors(self):
        with self.assertRaises(ValueError):
            expm(torch.empty(3, 4))
        with self.assertRaises(ValueError):
            expm(torch.empty(1, 4))
