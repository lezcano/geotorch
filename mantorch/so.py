from .stiefel import Stiefel


class SO(Stiefel):
    def __init__(self, triv="expm"):
        super().__init__(triv)

    def init(self, t):
        if t.ndimension() != 2 or t.size(0) != t.size(1):
            raise ValueError("Expected a square matrix. Got a t of shape {}"
                             .format(list(t.size())))
        super().init(t)

    def extra_repr(self):
        inv_map = {v: k for k, v in Stiefel.trivializations.items()}
        name = inv_map.get(self.triv)
        if name is None:
            name = "custom"
        return 'n={}, triv={}'.format(self.orig.size(0), name)


