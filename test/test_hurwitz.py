import torch
torch.set_default_dtype(torch.float64)
from unittest import TestCase
from geotorch.hurwitz import Hurwitz, get_lyap_exp
from geotorch.exceptions import NonSquareError, VectorError, InManifoldError
import geotorch

class TestHurwitz(TestCase):

    def setUp(self):
        self.size = (5, 5)
        self.alpha = 1e-3
        self.hurwitz = Hurwitz(self.size, alpha=self.alpha)

    def test_parse_size_correct(self):
        n = self.hurwitz.parse_size(self.size)
        self.assertEqual(n, 5)

    def test_parse_size_non_square_error(self):
        with self.assertRaises(NonSquareError):
            self.hurwitz.parse_size((5, 3))

    def test_parse_size_vector_error(self):
        with self.assertRaises(VectorError):
            self.hurwitz.parse_size((5,))

    def test_sample_shape_and_hurwitz(self):
        A = self.hurwitz.sample()
        self.assertEqual(A.shape, self.size)
        self.assertTrue(self.hurwitz.in_manifold_eigen(A))

    def test_forward_and_submersion(self):
        X1, X2, X3 = [torch.randn(5, 5) for _ in range(3)]
        A = self.hurwitz.forward(X1, X2, X3)
        self.assertEqual(A.shape, self.size)
        self.assertTrue(torch.is_tensor(A))

    def test_submersion_inv_valid_matrix(self):
        A = self.hurwitz.sample()
        alpha = get_lyap_exp(A) - 1e-4
        self.hurwitz.alpha = alpha
        Q, P_inv, S = self.hurwitz.submersion_inv(A)
        self.assertEqual(Q.shape, self.size)
        self.assertEqual(P_inv.shape, self.size)
        self.assertEqual(S.shape, self.size)

        A_reconstructed = P_inv @ (-0.5 * Q + S) - self.hurwitz.alpha * torch.eye(5)
        self.assertTrue(torch.allclose(A, A_reconstructed, atol=1e-4))

    def test_submersion_inv_in_manifold_error(self):
        A_unstable = torch.eye(5)
        with self.assertRaises(InManifoldError):
            self.hurwitz.submersion_inv(A_unstable)

    def test_right_inverse(self):
        A = self.hurwitz.sample()
        self.hurwitz.alpha = get_lyap_exp(A)
        X1, X2, X3 = self.hurwitz.right_inverse(A)
        A_from_ri = self.hurwitz.forward(X1, X2, X3)
        print(torch.dist(A_from_ri, A))
        self.assertTrue(torch.allclose(A, A_from_ri, atol=1e-4))

    def test_in_manifold_eigen_true(self):
        A = -torch.eye(5) * (self.alpha + 1)
        hurwitz = Hurwitz((1,5,5), alpha=self.alpha)
        self.assertTrue(hurwitz.in_manifold_eigen(A.unsqueeze(0)))

    def test_in_manifold_eigen_false(self):
        A = torch.eye(5)
        self.assertFalse(self.hurwitz.in_manifold_eigen(A))

    def test_initialization_alpha_negative(self):
        with self.assertRaises(AssertionError):
            Hurwitz(self.size, alpha=-0.1)

    def test_extra_repr(self):
        repr_str = self.hurwitz.extra_repr()
        self.assertIn('n=5', repr_str)
        self.assertIn('alpha=', repr_str)

    def test_sample(self):
        hurwitz = Hurwitz((5, 5), alpha=self.alpha)
        A = hurwitz.sample()
        self.assertEqual(A.shape, (5, 5))
        self.assertTrue(hurwitz.in_manifold_eigen(A))

    def test_batch_submersion_inv(self):
        hurwitz_batch = Hurwitz((1, 2, 2))
        A_batch = hurwitz_batch.sample()
        hurwitz_batch.alpha = get_lyap_exp(A_batch)
        Q, P_inv, S = hurwitz_batch.submersion_inv(A_batch)
        A_reconstructed = P_inv @ (-0.5 * Q + S) - (hurwitz_batch.alpha -1e-6) * torch.eye(2)

        self.assertTrue(torch.allclose(A_batch, A_reconstructed, atol=1e-4))

    def test_register_hurwitz(self):
        layer = torch.nn.Linear(self.size[-2], self.size[-1])
        geotorch.alpha_stable(layer, "weight", alpha=0.5)
        self.assertTrue(torch.all(torch.real(torch.linalg.eigvals(layer.weight)) <= -0.5))