from .constructions import FiberedSpace, ProductManifold
from .so import SO
from .stiefel import Stiefel, StiefelTall
from .reals import Rn
from .exceptions import VectorError, RankError


class LowRank(FiberedSpace):
    def __init__(self, size, rank, triv="expm"):
        r"""
        Variety of the matrices of rank :math:`r` or less.

        Args:
            size (torch.size): Size of the tensor to be applied to
            rank (int): Rank of the matrices.
                It has to be less or equal to
                :math:`\min(\texttt{size}[-1], \texttt{size}[-2])`
            triv (str or callable): Optional.
                A map that maps :math:`\operatorname{Skew}(n)` onto the orthogonal
                matrices surjectively. This is used to optimize the U and V in the
                SVD. It can be one of `["expm", "cayley"]` or a custom
                callable. Default: `"expm"`
        """

        size_u, size_s, size_v = LowRank.size_usv(size, rank)
        Stiefel_u = LowRank.cls_stiefel(size_u)
        Stiefel_v = LowRank.cls_stiefel(size_v)
        super().__init__(
            dimensions=2,
            size=size,
            total_space=ProductManifold(
                [Stiefel_u(size_u, triv), Rn(size_s), Stiefel_v(size_v, triv)]
            ),
        )
        self.rank = rank

    @classmethod
    def size_usv(cls, size, rank):
        if len(size) < 2:
            raise VectorError(cls.__name__, size)
        # Split the size and transpose if necessary
        tensorial_size = size[:-2]
        n, k = size[-2:]
        if n < k:
            n, k = k, n
        if rank > min(n, k) or rank < 1:
            raise RankError(n, k, rank)
        size_u = tensorial_size + (n, rank)
        size_s = tensorial_size + (rank,)
        size_v = tensorial_size + (k, rank)
        return size_u, size_s, size_v

    @staticmethod
    def cls_stiefel(size):
        n, k = size[-2:]
        if n == k:
            return SO
        elif n > 4 * k:
            return StiefelTall
        else:
            return Stiefel

    def embedding(self, X):
        U = X.tril(-1)[..., :, : self.rank]
        S = X.diagonal(dim1=-2, dim2=-1)[..., : self.rank]
        V = X.triu(1).transpose(-2, -1)[..., :, : self.rank]
        return U, S, V

    def submersion(self, X):
        U, S, V = X
        Vt = V.transpose(-2, -1)
        # Multiply the three of them, S as a diagonal matrix
        return U @ (S.unsqueeze(-1).expand_as(Vt) * Vt)
