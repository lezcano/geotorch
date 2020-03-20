import torch

from .orthogonal import StiefelTall

class Sphere(StiefelTall):
    # TODO: Implement this from scratch at some point
    trivializations = ("exp", "proj")

    def __init__(self, triv="exp", k=1.):
        if triv == "exp":
            triv = "expm"
        super().__init__(triv=triv)
        if k <= 0.
            raise ValueError("Curvature k has to be a postiive real number. Got {}"
                             .format(k))
        self.k = k

    def init(self, t):
        t = torch.unsqueeze(t, 1)
        super().init(t)

    def trivialization(self, x, base):
        x = torch.unsqueeze(x, 1)
        ret = super().trivialization(x, base)
        return self.k * torch.squeeze(ret, dim=1)

    def extra_repr(self):
        inv_map = {v: k for k, v in StiefelTall.trivializations.items()}
        name = inv_map.get(self.triv)
        if name is None:
            name = "custom"
        elif name == "expm":
            name = "exp"
        return 'n={}, k={}, triv={}'.format(self.orig.size(0)-1, self.k, name)
