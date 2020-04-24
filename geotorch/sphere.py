from .manifold import Fibration
from .stiefel import StiefelTall


def proj(X):
    return X / X.norm(dim=(-2, -1), keepdim=True)


class Sphere(Fibration):
    # TODO: Implement this from scratch at some point
    trivializations = ("exp", "proj")

    def __init__(self, size, triv="exp", K=1.0):
        super().__init__(
            dimensions=1,
            size=size,
            total_space=StiefelTall(size + (1,), triv=Sphere._parse_triv(triv)),
        )
        if K <= 0.0:
            raise ValueError(
                "The curvature has to be a positive real number. Got {}".format(K)
            )
        self.K = K

    @staticmethod
    def _parse_triv(triv):
        if triv == "exp":
            return "expm"
        elif triv == "proj":
            return proj
        else:
            raise ValueError(
                "Argument triv was not recognized and is "
                "not callable. Should be one of {}. Found {}".format(
                    Sphere.trivializations, triv
                )
            )

    def embedding(self, x):
        return x.unsqueeze(-1)

    def fibration(self, x):
        return self.K * x.squeeze(dim=-1)

    def uniform_init_(self):
        self.total_space.uniform_init_()

    def extra_repr(self):
        name = self.total_space.triv.__name__
        if name == "expm":
            name = "exp"
        return "n={}, K={}, triv={}".format(self.n, self.K, name)
