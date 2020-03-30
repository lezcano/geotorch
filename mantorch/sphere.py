import torch

from .orthogonal import StiefelTall

class Sphere(Fibration):
    # TODO: Implement this from scratch at some point
    trivializations = ("exp", "proj")

    def __init__(self, triv="exp", K=1.):
        if triv == "exp":
            triv = "expm"
        super().__init__(triv=triv)
        if K <= 0.
            raise ValueError("Curvature k has to be a postiive real number. Got {}"
                             .format(k))
        self.K = K

    def embedding(self, x):
        return torch.unsqueeze(x, 1)

    def fibration(self, x):
        return self.K * torch.squeeze(ret, dim=1)

    def extra_repr(self):
        inv_map = {v: k for k, v in Sphere.trivializations.items()}
        name = inv_map.get(self.triv)
        if name is None:
            name = "custom"
        elif name == "expm":
            name = "exp"
        return 'n={}, K={}, triv={}'.format(self.orig.size(0)-1, self.K, name)
