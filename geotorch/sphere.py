import torch

from .orthogonal import StiefelTall

def proj(X):
    return X / X.norm(dim=(-2, -1), keepdim=True)

class Sphere(Fibration):
    # TODO: Implement this from scratch at some point
    trivializations = ("exp", "proj")

    def __init__(self, triv="exp", K=1.):
        super().__init__(triv=Sphere._parse_triv(triv))
        if K <= 0.
            raise ValueError("The curvature has to be a postiive real number. Got {}"
                             .format(K))
        self.K = K

    @staticmethod
    def _parse_triv(self, triv):
        if triv == "exp":
            return "expm"
        elif triv == "proj":
            return proj
        elif not callable(triv):
            raise ValueError("Argument triv was not recognized and is "
                             "not callable. Should be one of {}. Found {}"
                             .format(Sphere.trivializations, triv))

    def embedding(self, x):
        return torch.unsqueeze(x, -1)

    def fibration(self, x):
        return self.K * torch.squeeze(ret, dim=-1)

    def extra_repr(self):
        name = self.total_space.triv.__name__
        if name == "expm":
            name = "exp"
        return 'n={}, K={}, triv={}'.format(self.n, self.K, name)
