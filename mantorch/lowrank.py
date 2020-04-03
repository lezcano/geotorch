from copy import deepcopy

import torch
import torch.nn as nn

from .manifold import Fibration, ProductManifold
from .stiefel import Stiefel, StiefelTall
from .so import SO
from .reals import Rn


class LowRank(Fibration):
    def __init__(self, size, rank):
        size_u, size_s, size_v = LowRank.size_usv(size, rank)
        Stiefel_u = LowRank.cls_stiefel(size_u)
        Stiefel_v = LowRank.cls_stiefel(size_v)
        super().__init__(dimensions=2, size=size,
                         total_space=ProductManifold([Stiefel_u(size_u),
                                                      Rn(size_s),
                                                      Stiefel_v(size_v)]))
        self.rank = rank

    @staticmethod
    def size_usv(size, rank):
        n, k = size[-1], size[-2]
        if n < k:
            n, k = k, n
        if rank > min(n, k) or rank < 1:
            raise ValueError("The rank has to be 1 <= rank <= min({}, {}). Found {}"
                             .format(n, k, rank))
        size_u = list(size)
        size_u[-2], size_u[-1] = n, rank
        size_s = list(size[:-1])
        size_s[-1] = rank
        size_v = list(size)
        size_v[-2], size_v[-1] = k, rank
        return tuple(size_u), tuple(size_s), tuple(size_v)

    @staticmethod
    def cls_stiefel(size):
        return StiefelTall if size[-2] > 2*size[-1] else Stiefel

    def embedding(self, X):
        U = X.tril(-1)[..., :, :self.rank]
        S = X.diagonal(dim1=-2, dim2=-1)[..., :self.rank]
        V = X.triu(1).transpose(-2, -1)[..., :, :self.rank]
        return U, S, V

    def fibration(self, X):
        U, S, V = X
        Vt = V.transpose(-2, -1)
        # Multiply the three of them, S as a diagonal matrix
        return U @ (S.unsqueeze(-1).expand_as(Vt) * Vt)
