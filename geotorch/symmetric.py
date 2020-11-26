import torch
import geotorch.parametrize as P

from .constructions import ProductManifold
from .so import SO
from .stiefel import Stiefel, StiefelTall
from .reals import Rn
from .exceptions import VectorError, NonSquareError, RankError
from .utils import _extra_repr


class Symmetric(P.Parametrization):
    def __init__(self, lower=True):
        r"""
        Vector space of symmetric matrices, parametrized in terms of the upper or lower
        triangular part of a matrix.

        Args:
            size (torch.size): Size of the tensor to be applied to
            lower (bool): Optional. Uses the lower triangular part of the matrix to
                parametrize the matrix. Default: `True`
        """
        super().__init__()
        self.lower = lower

    @staticmethod
    def frame(X, lower):
        if lower:
            return X.tril(0) + X.tril(-1).transpose(-2, -1)
        else:
            return X.triu(0) + X.triu(1).transpose(-2, -1)

    def forward(self, X):
        if len(X.size()) < 2:
            raise VectorError(type(self).__name__, X.size())
        if X.size(-2) != X.size(-1):
            raise NonSquareError(type(self).__name__, X.size())
        return self.frame(X, self.lower)


class SymF(ProductManifold):
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
        n, tensorial_size = SymF.parse_size(size)
        if rank > n or rank < 1:
            raise RankError(n, n, rank)
        super().__init__(SymF.manifolds(n, rank, tensorial_size, triv))
        self.n = n
        self.tensorial_size = tensorial_size
        self.rank = rank
        if not callable(f):
            raise ValueError("f should be callable. Got {}".format(f))
        self.f = f

    @classmethod
    def parse_size(cls, size):
        if len(size) < 2:
            raise VectorError(cls.__name__, size)
        n, k = size[-2:]
        tensorial_size = size[:-2]
        if n != k:
            raise NonSquareError(cls.__name__, size)
        return n, tensorial_size

    @staticmethod
    def manifolds(n, rank, tensorial_size, triv):
        size_q = tensorial_size + (n, rank)
        size_l = tensorial_size + (rank,)
        Stiefel_q = SymF.cls_stiefel(size_q)
        return Stiefel_q(size_q, triv), Rn(size_l)

    @staticmethod
    def cls_stiefel(size):
        if torch.__version__ >= "1.7.0":
            return Stiefel
        n, k = size[-2:]
        if n == k:
            return SO
        elif n > 4 * k:
            return StiefelTall
        else:
            return Stiefel

    def frame(self, X):
        L = X.diagonal(dim1=-2, dim2=-1)[..., : self.rank]
        X = X[..., : self.rank]
        return X, L

    def submersion(self, Q, L):
        L = self.f(L)
        Qt = Q.transpose(-2, -1)
        # Multiply the three of them as Q\LambdaQ^T
        return Q @ (L.unsqueeze(-1).expand_as(Qt) * Qt)

    def forward(self, X):
        X = self.frame(X)
        Q, L = super().forward(X)
        return self.submersion(Q, L)

    def extra_repr(self):
        return _extra_repr(n=self.n, rank=self.rank, tensorial_size=self.tensorial_size)
