import torch

from .manifold import Fibration
from .stiefel import Stiefel, StiefelTall


class Grassmanian(Fibration):
    r"""
    Implement everything as the fibration St(n,k) -> Gr(n,k)
    G(n,k) \iso St(n,k) / O(k)
    """
    def __init__(self, size, triv="expm"):
        size_st = Grassmanian.size_st(size)
        super().__init__(dimensions=2, size=size,
                         total_space=Stiefel(size_st, triv))
        self.triv = triv

    @staticmethod
    def size_st(size):
        if size[-2] < size[-1]:
            size = list(size)
            size[-1], size[-2] = size[-2], size[-1]
            size = tuple(size)
        return size

    def embedding(self, A):
        Z = A.new_zeros(self.k, self.k)
        return torch.cat([Z, A[self.k:]])

    def fibration(self, X):
        return X

    def extra_repr(self):
        return 'n={}, k={}, triv={}'.format(self.n, self.k, self.triv)


class GrassmanianTall(StiefelTall):
    def __init__(self, size, triv="expm"):
        super().__init__(size, triv)
        # Stiefel parametrization
        zeros = self.fibr_aux.new_zeros(self.k, self.k)
        delattr(self, "fibr_aux")
        self.register_buffer("fibr_aux", zeros)
