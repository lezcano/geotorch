import torch

from .manifold import Fibration
from .stiefel import Stiefel, StiefelTall


class Grassmannian(Fibration):
    r"""
    Implement everything as the fibration St(n,k) -> Gr(n,k)
    G(n,k) \iso St(n,k) / O(k)
    """

    def __init__(self, size, triv="expm"):
        size_st = Grassmannian.size_st(size)
        super().__init__(dimensions=2, size=size, total_space=Stiefel(size_st, triv))
        self.triv = triv

    @staticmethod
    def size_st(size):
        if size[-2] < size[-1]:
            size = list(size)
            size[-1], size[-2] = size[-2], size[-1]
            size = tuple(size)
        return size

    def embedding(self, A):
        size_z = self.tensorial_size + (self.k, self.k)
        Z = A.new_zeros(*size_z)
        return torch.cat([Z, A[..., self.k :, :]], dim=-2)

    def fibration(self, X):
        return X

    def uniform_init_(self):
        self.total_space.uniform_init_()

    def extra_repr(self):
        return "n={}, k={}, triv={}".format(self.n, self.k, self.triv)


class GrassmannianTall(StiefelTall):
    def __init__(self, size, triv="expm"):
        super().__init__(size, triv)
        # Stiefel parametrization
        zeros = self.fibr_aux.new_zeros(self.fibr_aux.size())
        delattr(self, "fibr_aux")
        self.register_buffer("fibr_aux", zeros)
