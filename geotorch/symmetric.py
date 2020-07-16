from .constructions import AbstractManifold, Fibration, ProductManifold
from .so import SO
from .stiefel import Stiefel, StiefelTall
from .reals import Rn
from .exceptions import VectorError, NonSquareError, RankError


class Symmetric(AbstractManifold):
    def __init__(self, size, lower=True):
        r"""
        Vector space of symmetric matrices, parametrized in terms of the upper or lower
        triangular part of a matrix.

        Args:
            size (torch.size): Size of the tensor to be applied to
            lower (bool): Optional. Uses the lower triangular part of the matrix to
                parametrize the matrix. Default: `True`
        """
        super().__init__(dimensions=2, size=size)
        if self.n != self.k:
            raise NonSquareError(self.__class__.__name__, size)
        self.lower = lower

    def forward(self, X):
        if self.lower:
            return X.tril(0) + X.tril(-1).transpose(-2, -1)
        else:
            return X.triu(0) + X.triu(1).transpose(-2, -1)

    def extra_repr(self):
        return "n={}".format(self.n)


class SymF(Fibration):
    def __init__(self, size, rank, f, triv="expm"):
        r"""
        Space of the symmetric matrices of rank at most k with eigenvalues
        in the image of a given function

        Args:
            size (torch.size): Size of the tensor to be applied to
            rank (int): Rank of the matrices.
                It has to be less or equal to
                :math:`\min(\texttt{size}[-1], \texttt{size}[-2])`
            f (callable): Function parametrizing the space of possible
                eigenvalues of the matrix
            triv (str or callable): Optional.
                A map that maps :math:`\operatorname{Skew}(n)` onto the orthogonal
                matrices surjectively. This is used to optimize the Q in the eigenvalue
                decomposition. It can be one of `["expm", "cayley"]` or a custom
                callable. Default: `"expm"`
        """

        size_q, size_l = SymF.size_ql(size, rank)
        Stiefel_q = SymF.cls_stiefel(size_q)
        super().__init__(
            dimensions=2,
            size=size,
            total_space=ProductManifold([Stiefel_q(size_q, triv), Rn(size_l)]),
        )
        self.rank = rank
        if not callable(f):
            raise ValueError("f should be callable. Got {}".format(f))
        self.f = f

    @classmethod
    def size_ql(cls, size, rank):
        if len(size) < 2:
            raise VectorError(cls.__name__, size)
        # Split the size and transpose if necessary
        tensorial_size = size[:-2]
        n, k = size[-2:]
        if n != k:
            raise NonSquareError(cls.__name__, size)
        if rank > n or rank < 1:
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
        return X[..., : self.rank], L

    def fibration(self, X):
        Q, L = X
        L = self.f(L)
        Qt = Q.transpose(-2, -1)
        # Multiply the three of them as Q\LambdaQ^T
        return Q @ (L.unsqueeze(-1).expand_as(Qt) * Qt)
