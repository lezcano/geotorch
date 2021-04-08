import torch
from torch import nn

from .exceptions import InManifoldError
from .utils import _extra_repr


def project(x):
    return x / x.norm(dim=-1, keepdim=True)


def uniform_init_sphere_(x, r=1.0):
    r"""Samples a point uniformly on the sphere into the tensor ``x``.
    If ``x`` has :math:`d > 1` dimensions, the first :math:`d-1` dimensions
    are treated as batch dimensions.
    """
    with torch.no_grad():
        x.normal_()
        x.data = r * project(x)
    return x


def _in_sphere(x, r, eps):
    norm = x.norm(dim=-1)
    rs = torch.full_like(norm, r)
    return (torch.norm(norm - rs, p=float("inf")) < eps).all()


class sinc_class(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        # Hardocoded for float, will do for now
        ret = torch.sin(x) / x
        ret[x.abs() < 1e-45] = 1.0
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        ret = torch.cos(x) / x - torch.sin(x) / (x * x)
        ret[x.abs() < 1e-10] = 0.0
        return ret * grad_output


sinc = sinc_class.apply


class SphereEmbedded(nn.Module):
    def __init__(self, size, radius=1.0):
        r"""
        Sphere as the orthogonal projection from
        :math:`\mathbb{R}^n` to :math:`\mathbb{S}^{n-1}`, that is,
        :math:`x \mapsto \frac{x}{\lVert x \rVert}`.

        Args:
            size (torch.size): Size of the tensor to be parametrized
            radius (float): Optional.
                Radius of the sphere. A positive number. Default: ``1.``
        """
        super().__init__()
        self.n = size[-1]
        self.tensorial_size = size[:-1]
        self.radius = SphereEmbedded.parse_radius(radius)

    @staticmethod
    def parse_radius(radius):
        if radius <= 0.0:
            raise ValueError(
                "The radius has to be a positive real number. Got {}".format(radius)
            )
        return radius

    def forward(self, x):
        return self.radius * project(x)

    def right_inverse(self, x, check_in_manifold=True):
        if check_in_manifold and not self.in_manifold(x):
            raise InManifoldError(x, self)
        return x / self.radius

    def in_manifold(self, x, eps=1e-5):
        r"""
        Checks that a vector is on the sphere.

        For tensors with more than 2 dimensions the first dimensions are
        treated as batch dimensions.

        Args:
            X (torch.Tensor): The vector to be checked.
            eps (float): Optional. Threshold at which the norm is considered
                    to be equal to `1`.
                    Default: ``1e-5``
        """
        return _in_sphere(x, self.radius, eps)

    def sample(self):
        r"""
        Returns a uniformly sampled vector on the sphere.
        """
        x = torch.empty(*(self.tensorial_size) + (self.n,))
        return uniform_init_sphere_(x, r=self.radius)

    def extra_repr(self):
        return _extra_repr(
            n=self.n, radius=self.radius, tensorial_size=self.tensorial_size
        )


class Sphere(nn.Module):
    def __init__(self, size, radius=1.0):
        r"""
        Sphere as a map from the tangent space onto the sphere using the
        exponential map.

        Args:
            size (torch.size): Size of the tensor to be parametrized
            radius (float): Optional.
                Radius of the sphere. A positive number. Default: ``1.``
        """
        super().__init__()
        self.n = size[-1]
        self.tensorial_size = size[:-1]
        self.radius = Sphere.parse_radius(radius)
        self.register_buffer("base", uniform_init_sphere_(torch.empty(*size)))

    @staticmethod
    def parse_radius(radius):
        if radius <= 0.0:
            raise ValueError(
                "The radius has to be a positive real number. Got {}".format(radius)
            )
        return radius

    def frame(self, x, v):
        projection = (v.unsqueeze(-2) @ x.unsqueeze(-1)).squeeze(-1)
        v = v - projection * x
        return v

    def forward(self, v):
        x = self.base
        # Project v onto {<x,v> = 0}
        v = self.frame(x, v)
        vnorm = v.norm(dim=-1, keepdim=True)
        return self.radius * (torch.cos(vnorm) * x + sinc(vnorm) * v)

    def right_inverse(self, x, check_in_manifold=True):
        if check_in_manifold and not self.in_manifold(x):
            raise InManifoldError(x, self)
        with torch.no_grad():
            x = x / self.radius
            self.base.copy_(x)
        return torch.zeros_like(x)

    def in_manifold(self, x, eps=1e-5):
        r"""
        Checks that a vector is on the sphere.

        For tensors with more than 2 dimensions the first dimensions are
        treated as batch dimensions.

        Args:
            X (torch.Tensor): The vector to be checked.
            eps (float): Optional. Threshold at which the norm is considered
                    to be equal to `1`.
                    Default: ``1e-5``
        """
        return _in_sphere(x, self.radius, eps)

    def sample(self):
        r"""
        Returns a uniformly sampled vector on the sphere.
        """
        device = self.base.device
        dtype = self.base.dtype
        x = torch.empty(*(self.tensorial_size) + (self.n,), device=device, dtype=dtype)
        return uniform_init_sphere_(x, r=self.radius)

    def extra_repr(self):
        return _extra_repr(
            n=self.n, radius=self.radius, tensorial_size=self.tensorial_size
        )
