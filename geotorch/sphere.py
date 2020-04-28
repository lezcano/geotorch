import torch

from .manifold import EmbeddedManifold, Fibration
from .stiefel import StiefelTall


def project(x):
    return x / x.norm(dim=-1, keepdim=True)


class SphereEmbedded(EmbeddedManifold):
    projections = {"project": project}

    def __init__(self, size, projection="project", K=1.0):
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

        if K <= 0.0:
            raise ValueError(
                "The curvature has to be a positive real number. Got {}".format(K)
            )
        self.K = K
        self.uniform_init_()

    def projection(self, x):
        return self.K * self.proj(x)

    def uniform_init_(self):
        if self.is_registered():
            with torch.no_grad():
                x = self.original_tensor()
                x.normal_()
                x.data = project(x.data)

    def extra_repr(self):
        return "n={}, K={}, projection={}".format(
            self.n, self.K, self.projection.__name__
        )


class Sphere(Fibration):
    def __init__(self, size, K=1.0):
        super().__init__(
            dimensions=1, size=size, total_space=StiefelTall(size + (1,), triv="expm"),
        )
        if K <= 0.0:
            raise ValueError(
                "The curvature has to be a positive real number. Got {}".format(K)
            )
        self.K = K

    def embedding(self, x):
        return x.unsqueeze(-1)

    def fibration(self, x):
        return self.K * x.squeeze(dim=-1)

    def uniform_init_(self):
        self.total_space.uniform_init_()

    def extra_repr(self):
        return "n={}, K={}".format(self.n, self.K)
