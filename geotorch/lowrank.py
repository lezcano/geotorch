from collections.abc import Iterable
import torch
from .product import ProductManifold
from .so import SO
from .stiefel import Stiefel, StiefelTall
from .reals import Rn
from .exceptions import VectorError, RankError, InManifoldError
from .utils import transpose, _extra_repr


class LowRank(ProductManifold):
    def __init__(self, size, rank, triv="expm"):
        r"""
        Variety of the matrices of rank :math:`r` or less.

        Args:
            size (torch.size): Size of the tensor to be parametrized
            rank (int): Rank of the matrices.
                It has to be less or equal to
                :math:`\min(\texttt{size}[-1], \texttt{size}[-2])`
            triv (str or callable): Optional.
                A map that maps skew-symmetric matrices onto the orthogonal
                matrices surjectively. This is used to optimize the U and V in the
                SVD. It can be one of ``["expm", "cayley"]`` or a custom
                callable. Default: ``"expm"``
        """
        n, k, tensorial_size = LowRank.parse_size(size)
        if rank > min(n, k) or rank < 1:
            raise RankError(n, k, rank)
        super().__init__(LowRank.manifolds(n, k, rank, tensorial_size, triv))
        self.n = n
        self.k = k
        self.tensorial_size = tensorial_size
        self.rank = rank

    @classmethod
    def parse_size(cls, size):
        if len(size) < 2:
            raise VectorError(cls.__name__, size)
        n = max(size[-2:])
        k = min(size[-2:])
        tensorial_size = size[:-2]
        return n, k, tensorial_size

    @staticmethod
    def manifolds(n, k, rank, tensorial_size, triv):
        size_u = tensorial_size + (n, rank)
        size_s = tensorial_size + (rank,)
        size_v = tensorial_size + (k, rank)
        Stiefel_u = LowRank.cls_stiefel(size_u)
        Stiefel_v = LowRank.cls_stiefel(size_v)
        return Stiefel_u(size_u, triv), Rn(size_s), Stiefel_v(size_v, triv)

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
        U = X.tril(-1)[..., : self.rank]
        S = X.diagonal(dim1=-2, dim2=-1)[..., : self.rank]
        V = X.triu(1).transpose(-2, -1)[..., : self.rank]
        return U, S, V

    def submersion(self, U, S, V):
        Vt = V.transpose(-2, -1)
        # Multiply the three of them, S as a diagonal matrix
        return U @ (S.unsqueeze(-1).expand_as(Vt) * Vt)

    @transpose
    def forward(self, X):
        Xs = self.frame(X)
        U, S, V = super().forward(Xs)
        return self.submersion(U, S, V)

    @transpose
    def initialize_(self, X):
        USV = self.submersion_inv(X)
        X1, X2, X3 = super().initialize_(USV)
        return self.frame_inv(X1, X2, X3)

    def frame_inv(self, X1, X2, X3):
        # X1 is lower-triangular
        # X2 is a vector
        # X3 is upper-triangular
        size = self.tensorial_size + (self.n, self.k)
        ret = torch.zeros(*size, dtype=X1.dtype, device=X1.device)
        with torch.no_grad():
            ret[..., : self.rank] = X1
            ret[..., : self.rank, : self.rank] = torch.diag_embed(X2)
            ret[..., : self.rank, :] = X3
        return ret

    def submersion_inv(self, X):
        if isinstance(X, torch.Tensor):
            with torch.no_grad():
                U, S, V = X.svd()
        else:
            # We assume that we got he U S V factorised in a tuple / list
            U, S, V = X
        if not LowRank.in_manifold_singular_values(S, rank=self.rank):
            raise InManifoldError(X, self)
        return U[..., : self.rank], S[..., : self.rank], V[..., : self.rank]

    @staticmethod
    def in_manifold_singular_values(S, rank, eps=1e-3):
        # We compute the 1-norm of the vector normalising by the size of the vector
        D = S[..., rank:]
        if D.size(-1) == 0:
            return True
        avg_err = D.abs().sum(dim=-1) / len(D)
        return (avg_err < eps).all()

    @staticmethod
    def in_manifold(X, size, rank, eps=1e-3):
        _, S, _ = X.svd()
        return LowRank.in_manifold_svd(S, size, rank, eps)

    def extra_repr(self):
        return _extra_repr(
            n=self.n, k=self.k, rank=self.rank, tensorial_size=self.tensorial_size
        )
