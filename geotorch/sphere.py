import torch

from .manifold import EmbeddedManifold, Fibration
from .stiefel import StiefelTall


def project(x):
    return x / x.norm(dim=-1, keepdim=True)


class SphereEmbedded(EmbeddedManifold):
    projections = {"project": project}

    def __init__(self, size, projection="project", r=1.0):
        r"""
        Sphere as a map from :math:`\mathbb{R}^n` to :math:`\mathbb{S}^{n-1}`.
        By default it uses the orthogonal projection
        :math:`x \mapsto \frac{x}{\lVert x \rVert}`.

        Args:
            size (torch.size): Size of the tensor to be applied to
            projection (str or callable): Optional.
                It can be `"projection"` or a map that from :math:`\mathbb{R}^n`
                onto the sphere :math`\mathbb{S}^{n-1}`.
            r (float): Optional.
                Radius of the sphere. It has to be positive. Default: 1.
        """
        super().__init__(dimensions=1, size=size)

        if projection not in SphereEmbedded.projections.keys() and not callable(
            projection
        ):
            raise ValueError(
                "Argument projection was not recognized and is "
                "not callable. Should be one of {}. Found {}".format(
                    list(SphereEmbedded.projections.keys()), projection
                )
            )
        if callable(projection):
            self.proj = projection
        else:
            self.proj = SphereEmbedded.projections[projection]

        if r <= 0.0:
            raise ValueError(
                "The radius has to be a positive real number. Got {}".format(r)
            )
        self.r = r
        self.uniform_init_()

    def projection(self, x):
        return self.r * self.proj(x)

    def uniform_init_(self):
        r"""Samples a point uniformly on the sphere"""
        if self.is_registered():
            with torch.no_grad():
                x = self.original_tensor()
                x.normal_()
                x.data = project(x.data)

    def extra_repr(self):
        return "n={}, r={}, projection={}".format(
            self.n, self.r, self.projection.__name__
        )


class Sphere(Fibration):
    def __init__(self, size, r=1.0):
        r"""
        Sphere as a map from the tangent space onto the sphere using the exponential map.

        Args:
            size (torch.size): Size of the tensor to be applied to
            r (float): Optional.
                Radius of the sphere. It has to be positive. Default: 1.
        """

        super().__init__(
            dimensions=1, size=size, total_space=StiefelTall(size + (1,), triv="expm"),
        )
        if r <= 0.0:
            raise ValueError(
                "The radius has to be a positive real number. Got {}".format(r)
            )
        self.r = r

    def embedding(self, x):
        return x.unsqueeze(-1)

    def fibration(self, x):
        return self.r * x.squeeze(dim=-1)

    def uniform_init_(self):
        r"""Samples a point uniformly on the sphere"""
        self.total_space.uniform_init_()

    def extra_repr(self):
        return "n={}, r={}".format(self.n, self.r)
