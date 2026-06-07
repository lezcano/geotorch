# Tests for the Sphere
from unittest import TestCase

import pytest
import torch

from geotorch.sphere import Sphere, SphereEmbedded


class TestSphere(TestCase):
    def test_construction(self):
        # Negative curvature
        with self.assertRaises(ValueError):
            Sphere(size=(5,), radius=-1.0)
        with self.assertRaises(ValueError):
            SphereEmbedded(size=(4,), radius=-1.0)

    def test_repr(self):
        print(SphereEmbedded(size=(3,)))
        print(Sphere(size=(3,)))

    def test_autograd(self):
        sphere = Sphere(size=(3,)).double()
        v = (0.1 * torch.randn(3, dtype=torch.float64)).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(sphere, (v,)))

    def test_torch_func(self):
        sphere = Sphere(size=(3,)).double()
        v = torch.zeros(3, dtype=torch.float64)
        tangent = torch.randn_like(v)

        try:
            torch.func.jacrev(lambda x: x.square())(v)
        except NotImplementedError as error:
            if "TensorWrapper" in str(error):
                pytest.skip("torch.func is broken in this PyTorch build")
            raise

        jacobian = torch.func.jacrev(sphere)(v)
        output, jvp = torch.func.jvp(sphere, (v,), (tangent,))
        batched = torch.vmap(sphere)(torch.stack((v, tangent)))

        self.assertTrue(torch.isfinite(jacobian).all())
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(jvp).all())
        self.assertTrue(torch.allclose(jvp, jacobian @ tangent))
        self.assertEqual(batched.shape, (2, 3))

    def test_embedded_sample_uses_module_dtype(self):
        sphere = SphereEmbedded(size=(2, 3)).double()
        sample = sphere.sample()
        self.assertEqual(sample.dtype, torch.float64)
        self.assertEqual(sample.shape, (2, 3))

    def test_membership_preserves_strict_epsilon_boundary(self):
        sphere = SphereEmbedded(size=(1,), radius=2.0)
        x = torch.tensor([2.0001], dtype=torch.float64)
        eps = (torch.linalg.vector_norm(x) - sphere.radius).item()

        self.assertFalse(sphere.in_manifold(x, eps=eps))
        self.assertTrue(sphere.in_manifold(x, eps=2.0 * eps))
