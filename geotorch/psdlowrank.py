from .constructions import Fibration, ProductManifold
from .so import SO
from .stiefel import Stiefel, StiefelTall
from .reals import Rn
from .exceptions import VectorError, NonSquareError, RankError


class PSDLowRank(Fibration):
    def __init__(self, size, rank):
        r"""
        Variety of the positive semidefinite square matrices of rank :math:`r` or less.

        Args:
            size (torch.size): Size of the tensor to be applied to
            rank (int): Rank of the matrices.
                It has to be less or equal to
                :math:`\min(\texttt{size}[-1], \texttt{size}[-2])`
        """

        size_q, size_l = PSDLowRank.size_ql(size, rank)
        Stiefel_q = PSDLowRank.cls_stiefel(size_q)
        super().__init__(
            dimensions=2,
            size=size,
            total_space=ProductManifold(
                [Stiefel_q(size_q), Rn(size_l)]
            ),
        )
        self.rank = rank

    @classmethod
    def size_ql(cls, size, rank):
        if len(size) < 2:
            raise VectorError(cls.__name__, size)
        # Split the size and transpose if necessary
        tensorial_size = size[:-2]
        n, k = size[-2:]
        if n != k:
            raise NonSquareError(cls.__name__, size)
        if rank > min(n, k) or rank < 1:
            raise RankError(n, k, rank)
        size_q = tensorial_size + (n, rank)
        size_l = tensorial_size + (rank,)
        return size_q, size_l

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
        L = X.diagonal(dim1=-2, dim2=-1)[..., : self.rank]
        return X, L

    def fibration(self, X):
        Q, L = X
        Qt = Q.transpose(-2, -1)
        # Multiply the three of them as Q\LambdaQ^T
        return Q @ (L.unsqueeze(-1).expand_as(Qt) * Qt)
