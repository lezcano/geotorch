import torch
from torch import nn

from .exceptions import InManifoldError
from .utils import base, _extra_repr


def project(x):
    return x / x.norm(dim=-1, keepdim=True)


def uniform_init_sphere_(x, r=1.0):
    r"""Samples a point uniformly on the sphere into x"""
    with torch.no_grad():
        x.normal_()
        x.data = r * project(x.data)
    return x


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
    trivializations = {"project": project}

    def __init__(self, size, triv="project", r=1.0):
        r"""
        Sphere as a map from :math:`\mathbb{R}^n` to :math:`\mathbb{S}^{n-1}`.
        By default it uses the orthogonal projection
        :math:`x \mapsto \frac{x}{\lVert x \rVert}`.

        Args:
            size (torch.size): Size of the tensor to be applied to
            triv (str or callable): Optional.
                A map that maps :math:`\mathbb{R}^n` onto the sphere surjectively.
                It can be either `"project"` or a custom callable. Default: `"project"`
            r (float): Optional.
                Radius of the sphere. It has to be positive. Default: 1.
        """
        super().__init__()
        self.triv = SphereEmbedded.parse_triv(triv)
        self.r = SphereEmbedded.parse_r(r)
        self.n = size[-1]
        self.tensorial_size = size[:-1]

    @staticmethod
    def parse_triv(triv):
        if triv in SphereEmbedded.trivializations.keys():
            return SphereEmbedded.trivializations[triv]
        elif callable(triv):
            return triv
        else:
            raise ValueError(
                "Argument triv was not recognized and is "
                "not callable. Should be one of {}. Found {}".format(
                    list(SphereEmbedded.trivializations.keys()), triv
                )
            )

    @staticmethod
    def parse_r(r):
        if r <= 0.0:
            raise ValueError(
                "The radius has to be a positive real number. Got {}".format(r)
            )
        return r

    def forward(self, x):
        return self.r * self.triv(x)

    def initialize_(self, X):
        if not SphereEmbedded.in_manifold(X, self.r):
            raise InManifoldError(X, self)
        return X / self.r

    @staticmethod
    def in_manifold(X, r, eps=1e-4):
        return Sphere.in_manifold(X, r, eps)

    def extra_repr(self):
        return _extra_repr(n=self.n, r=self.r, tensorial_size=self.tensorial_size)


class Sphere(nn.Module):
    def __init__(self, size, r=1.0):
        r"""
        Sphere as a map from the tangent space onto the sphere using the
        exponential map.

        Args:
            size (torch.size): Size of the tensor to be applied to
            r (float): Optional.
                Radius of the sphere. It has to be positive. Default: 1.
        """
        super().__init__()
        if r <= 0.0:
            raise ValueError(
                "The radius has to be a positive real number. Got {}".format(r)
            )
        self.n = size[-1]
        self.tensorial_size = size[:-1]
        self.r = r
        self.register_buffer("base", uniform_init_sphere_(torch.empty(*size)))

    def frame(self, x, v):
        projection = (v.unsqueeze(-2) @ x.unsqueeze(-1)).squeeze(-1)
        v = v - projection * x
        return v

    @base
    def forward(self, v):
        x = self.base
        # Project v onto {<x,v> = 0}
        v = self.frame(x, v)
        vnorm = v.norm(dim=-1, keepdim=True)
        return self.r * (torch.cos(vnorm) * x + sinc(vnorm) * v)

    def initialize_(self, X):
        if not Sphere.in_manifold(X, self.r):
            raise InManifoldError(X, self)
        with torch.no_grad():
            X = X / self.r
            self.base.data = X.data
        return torch.zeros_like(X)

    @staticmethod
    def in_manifold(X, r, eps=1e-4):
        norm = X.norm(dim=-1)
        rs = torch.full_like(norm, r)
        return (torch.norm(norm - rs, p=float("inf")) < eps).all()

    def extra_repr(self):
        return _extra_repr(n=self.n, r=self.r, tensorial_size=self.tensorial_size)
