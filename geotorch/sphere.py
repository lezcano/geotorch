import torch

from .constructions import Manifold


def project(x):
    return x / x.norm(dim=-1, keepdim=True)


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
        x, = ctx.saved_tensors
        ret = torch.cos(x) / x - torch.sin(x) / (x * x)
        ret[x.abs() < 1e-10] = 0.0
        return ret * grad_output


sinc = sinc_class.apply


class SphereEmbedded(Manifold):
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
        super().__init__(dimensions=1, size=size, dynamic=False)

        if triv not in SphereEmbedded.trivializations.keys() and not callable(triv):
            raise ValueError(
                "Argument triv was not recognized and is "
                "not callable. Should be one of {}. Found {}".format(
                    list(SphereEmbedded.trivializations.keys()), triv
                )
            )
        if callable(triv):
            self.triv = triv
        else:
            self.triv = SphereEmbedded.trivializations[triv]

        if r <= 0.0:
            raise ValueError(
                "The radius has to be a positive real number. Got {}".format(r)
            )
        self.r = r
        self.uniform_init_()

    def trivialization(self, x):
        return self.r * self.triv(x)

    def uniform_init_(self):
        r"""Samples a point uniformly on the sphere"""
        if self.is_registered():
            with torch.no_grad():
                x = self.original_tensor()
                x.normal_()
                x.data = project(x.data)

    def extra_repr(self):
        return "n={}, r={}, triv={}".format(self.n, self.r, self.triv.__name__)


class Sphere(Manifold):
    def __init__(self, size, r=1.0):
        r"""
        Sphere as a map from the tangent space onto the sphere using the
        exponential map.

        Args:
            size (torch.size): Size of the tensor to be applied to
            r (float): Optional.
                Radius of the sphere. It has to be positive. Default: 1.
        """
        super().__init__(
            dimensions=1,
            size=size
        )
        print("SIZE")
        print(size)
        if r <= 0.0:
            raise ValueError(
                "The radius has to be a positive real number. Got {}".format(r)
            )
        self.r = r

    def trivialization(self, v):
        # Project v onto {<x,v> = 0}
        x = self.base
        projection = (v.unsqueeze(-2) @ x.unsqueeze(-1)).squeeze(-1)
        print("PROJECTION")
        print(projection.size())
        v = v - projection * x
        vnorm = v.norm(dim=-1, keepdim=True)
        ret = torch.cos(vnorm) * x + sinc(vnorm) * v
        print(ret)
        return ret

    def uniform_init_(self):
        r"""Samples a point uniformly on the sphere"""
        self.base.data.normal_()
        self.base.data = project(self.base.data)
        if self.is_registered():
            self.original_tensor().zero_()

    def extra_repr(self):
        return "n={}, r={}".format(self.n, self.r)
