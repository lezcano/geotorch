from .constructions import Fibration, ProductManifold
from .stiefel import Stiefel, StiefelTall
from .reals import Rn


class LowRank(Fibration):
    def __init__(self, size, rank):
        r"""
        Variety of the matrices of rank :math:`r` or less.

        Args:
            size (torch.size): Size of the tensor to be applied to
            rank (int): Rank of the matrices.
                It has to be less than
                :math:`\min(\texttt{size}[-1], \texttt{size}[-2])`
        """

        size_u, size_s, size_v = LowRank.size_usv(size, rank)
        Stiefel_u = LowRank.cls_stiefel(size_u)
        Stiefel_v = LowRank.cls_stiefel(size_v)
        super().__init__(
            dimensions=2,
            size=size,
            total_space=ProductManifold(
                [Stiefel_u(size_u), Rn(size_s), Stiefel_v(size_v)]
            ),
        )
        self.rank = rank

    @staticmethod
    def size_usv(size, rank):
        # Split the size and transpose if necessary
        tensorial_size = size[:-2]
        n, k = size[-2:]
        if n < k:
            n, k = k, n
        if rank > min(n, k) or rank < 1:
            raise ValueError(
                "The rank has to be 1 <= rank <= min({}, {}). Found {}".format(
                    n, k, rank
                )
            )
        size_u = tensorial_size + (n, rank)
        size_s = tensorial_size + (rank,)
        size_v = tensorial_size + (k, rank)
        return size_u, size_s, size_v

    @staticmethod
    def cls_stiefel(size):
        n, k = size[-2:]
        return StiefelTall if n > 4 * k else Stiefel

    def embedding(self, X):
        U = X.tril(-1)[..., :, : self.rank]
        S = X.diagonal(dim1=-2, dim2=-1)[..., : self.rank]
        V = X.triu(1).transpose(-2, -1)[..., :, : self.rank]
        return U, S, V

    def fibration(self, X):
        U, S, V = X
        Vt = V.transpose(-2, -1)
        # Multiply the three of them, S as a diagonal matrix
        return U @ (S.unsqueeze(-1).expand_as(Vt) * Vt)
