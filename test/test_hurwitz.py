import geotorch
from geotorch.exceptions import NonSquareError, VectorError, InManifoldError
from geotorch.hurwitz import Hurwitz, get_lyap_exp
from unittest import TestCase, skipUnless
import torch


class TestHurwitz(TestCase):

    def setUp(self):
        self.size = (2, 5, 5)
        self.alpha = 1e-3
        self.hurwitz = Hurwitz(self.size, alpha=self.alpha).double()

    def test_parse_size_correct(self):
        n, tensorial_size = self.hurwitz.parse_size(self.size)
        self.assertEqual(n, 5)
        self.assertEqual(tensorial_size, (2,))

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
        X1, X2, X3 = [torch.randn(*self.size, dtype=torch.float64) for _ in range(3)]
        A = self.hurwitz.forward(X1, X2, X3)
        self.assertEqual(A.shape, self.size)
        self.assertTrue(torch.is_tensor(A))

    def test_submersion_inv_valid_matrix(self):
        A = self.hurwitz.sample()
        alpha = float(get_lyap_exp(A)) / 2
        inverse = Hurwitz(self.size, alpha=alpha).double()
        Q, P_inv, S = inverse.submersion_inv(A)
        self.assertEqual(Q.shape, self.size)
        self.assertEqual(P_inv.shape, self.size)
        self.assertEqual(S.shape, self.size)

        A_reconstructed = inverse.submersion(Q, P_inv, S)
        self.assertTrue(torch.allclose(A, A_reconstructed, atol=1e-4))

    def test_submersion_inv_in_manifold_error(self):
        A_unstable = torch.eye(5, dtype=torch.float64).expand(self.size)
        with self.assertRaises(InManifoldError):
            self.hurwitz.submersion_inv(A_unstable)

    def test_right_inverse(self):
        A = self.hurwitz.sample()
        alpha = float(get_lyap_exp(A)) / 2
        inverse = Hurwitz(self.size, alpha=alpha).double()
        X1, X2, X3 = inverse.right_inverse(A)
        A_from_ri = inverse.forward(X1, X2, X3)
        self.assertTrue(torch.allclose(A, A_from_ri, atol=1e-4))

    def test_in_manifold_eigen_true(self):
        A = -torch.eye(5) * (self.alpha + 1)
        hurwitz = Hurwitz((1, 5, 5), alpha=self.alpha)
        self.assertTrue(hurwitz.in_manifold_eigen(A.unsqueeze(0)))

    def test_in_manifold_eigen_false(self):
        A = torch.eye(5)
        self.assertFalse(self.hurwitz.in_manifold_eigen(A))

    def test_initialization_alpha_negative(self):
        with self.assertRaises(ValueError):
            Hurwitz(self.size, alpha=-0.1)

    def test_extra_repr(self):
        repr_str = self.hurwitz.extra_repr()
        self.assertIn("n=5", repr_str)
        self.assertIn("alpha=", repr_str)

    def test_sample(self):
        hurwitz = Hurwitz((5, 5), alpha=self.alpha)
        A = hurwitz.sample()
        self.assertEqual(A.shape, (5, 5))
        self.assertTrue(hurwitz.in_manifold_eigen(A))

    def test_batch_sample(self):
        shapes = []

        def init_(tensor):
            shapes.append(tensor.shape)
            return torch.nn.init.normal_(tensor)

        hurwitz = Hurwitz((4, 5, 5), alpha=self.alpha).double()
        A = hurwitz.sample(init_)
        self.assertEqual(A.shape, (4, 5, 5))
        self.assertEqual(shapes, [torch.Size((4, 5, 5))] * 3)
        self.assertFalse(torch.equal(A[0], A[1]))
        self.assertTrue(hurwitz.in_manifold_eigen(A))

    def test_batch_submersion_inv(self):
        hurwitz_batch = Hurwitz((7, 2, 2)).double()
        A_batch = hurwitz_batch.sample()
        alpha = float(get_lyap_exp(A_batch)) / 2
        inverse = Hurwitz((7, 2, 2), alpha=alpha).double()
        Q, P_inv, S = inverse.submersion_inv(A_batch)
        A_reconstructed = inverse.submersion(Q, P_inv, S)

        self.assertTrue(torch.allclose(A_batch, A_reconstructed, atol=1e-4))

    def test_multi_batch_submersion_inv(self):
        size = (2, 3, 2, 2)
        hurwitz_batch = Hurwitz(size).double()
        A_batch = hurwitz_batch.sample()
        alpha = float(get_lyap_exp(A_batch)) / 2
        inverse = Hurwitz(size, alpha=alpha).double()
        Q, P_inv, S = inverse.submersion_inv(A_batch)
        A_reconstructed = inverse.submersion(Q, P_inv, S)

        self.assertTrue(torch.allclose(A_batch, A_reconstructed, atol=1e-4))

    def test_register_hurwitz(self):
        layer = torch.nn.Linear(self.size[-2], self.size[-1])
        geotorch.alpha_stable(layer, "weight", alpha=0.5)
        self.assertTrue(
            torch.all(torch.real(torch.linalg.eigvals(layer.weight)) <= -0.5)
        )

    def test_sample_uses_manifold_dtype(self):
        dtype = (
            torch.float64
            if torch.get_default_dtype() == torch.float32
            else torch.float32
        )
        hurwitz = Hurwitz((3, 3), alpha=self.alpha).to(dtype=dtype)
        self.assertEqual(hurwitz.sample().dtype, dtype)

    def test_alpha_round_trips_in_state_dict(self):
        source = Hurwitz((3, 3), alpha=0.75).double()
        target = Hurwitz((3, 3), alpha=0.25).double()
        target.load_state_dict(source.state_dict())
        self.assertEqual(target.alpha, source.alpha)
        self.assertEqual(target.sample().dtype, source.sample().dtype)

    @skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_sampling_and_registration(self):
        hurwitz = Hurwitz((2, 4, 4), alpha=0.5).to(device="cuda", dtype=torch.float64)
        sample = hurwitz.sample()
        self.assertEqual(sample.device.type, "cuda")
        self.assertEqual(sample.dtype, torch.float64)
        self.assertTrue(hurwitz.in_manifold_eigen(sample))

        layer = torch.nn.Linear(4, 4, device="cuda", dtype=torch.float64)
        geotorch.alpha_stable(layer, "weight", alpha=0.5)
        self.assertEqual(layer.weight.device.type, "cuda")
        self.assertEqual(layer.weight.dtype, torch.float64)
        self.assertTrue(torch.all(torch.linalg.eigvals(layer.weight).real <= -0.5))
