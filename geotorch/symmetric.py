import torch
from torch import nn

from .product import ProductManifold
from .stiefel import Stiefel
from .reals import Rn
from .exceptions import (
    VectorError,
    NonSquareError,
    RankError,
    InManifoldError,
    InverseError,
)
from .utils import _extra_repr


class Symmetric(nn.Module):
    def __init__(self, lower=True):
        r"""
        Vector space of symmetric matrices, parametrized in terms of the upper or lower
        triangular part of a matrix.

        Args:
            size (torch.size): Size of the tensor to be parametrized
            lower (bool): Optional. Uses the lower triangular part of the matrix to
                parametrize the matrix. Default: ``True``
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

    @staticmethod
    def in_manifold(X, eps=1e-6):
        return (
            X.dim() >= 2
            and X.size(-2) == X.size(-1)
            and torch.allclose(X, X.transpose(-2, -1), atol=eps)
        )


class SymF(ProductManifold):
    def __init__(self, size, rank, f, triv="expm"):
        r"""
        Space of the symmetric matrices of rank at most k with eigenvalues
        in the image of a given function

        Args:
            size (torch.size): Size of the tensor to be parametrized
            rank (int): Rank of the matrices.
                It has to be less or equal to
                :math:`\min(\texttt{size}[-1], \texttt{size}[-2])`
            f (callable or pair of callables): Either:

                - A callable

                - A pair of callables such that the second is a (right)
                  inverse of the first
            triv (str or callable): Optional.
                A map that maps skew-symmetric matrices onto the orthogonal matrices
                surjectively. This is used to optimize the :math:`Q` in the eigenvalue
                decomposition. It can be one of ``["expm", "cayley"]`` or a custom
                callable. Default: ``"expm"``
        """
        n, tensorial_size = SymF.parse_size(size)
        if rank > n or rank < 1:
            raise RankError(n, n, rank)
        super().__init__(SymF.manifolds(n, rank, tensorial_size, triv))
        self.n = n
        self.tensorial_size = tensorial_size
        self.rank = rank
        f, inv = SymF.parse_f(f)
        self.f = f
        self.inv = inv

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
    def parse_f(f):
        if callable(f):
            return f, None
        elif isinstance(f, tuple) and callable(f[0]) and callable(f[1]):
            return f
        else:
            raise ValueError(
                "Argument f is not callable nor a pair of callables. "
                "Found {}".format(f)
            )

    @staticmethod
    def manifolds(n, rank, tensorial_size, triv):
        size_q = tensorial_size + (n, rank)
        size_l = tensorial_size + (rank,)
        return Stiefel(size_q, triv=triv), Rn(size_l)

    def frame(self, X):
        L = X.diagonal(dim1=-2, dim2=-1)[..., : self.rank]
        X = X[..., : self.rank]
        return X, L

    def submersion(self, Q, L):
        L = self.f(L)
        return (Q * L.unsqueeze(-2)) @ Q.transpose(-2, -1)

    def forward(self, X):
        X = self.frame(X)
        Q, L = super().forward(X)
        return self.submersion(Q, L)

    def frame_inv(self, X1, X2):
        size = self.tensorial_size + (self.n, self.n)
        ret = torch.zeros(*size, dtype=X1.dtype, device=X1.device)
        with torch.no_grad():
            ret[..., : self.rank] += X1
            ret[..., : self.rank, : self.rank] += torch.diag_embed(X2)
        return ret

    def submersion_inv(self, X, check_in_manifold=True):
        with torch.no_grad():
            L, Q = torch.linalg.eigh(X)
        if check_in_manifold and not self.in_manifold_eigen(L):
            raise InManifoldError(X, self)
        if self.inv is None:
            raise InverseError(self)
        with torch.no_grad():
            Q = Q[..., -self.rank :]
            L = L[..., -self.rank :]
            L = self.inv(L)
        return L, Q

    def right_inverse(self, X, check_in_manifold=True):
        L, Q = self.submersion_inv(X, check_in_manifold)
        X1, X2 = super().right_inverse([Q, L], check_in_manifold=False)
        return self.frame_inv(X1, X2)

    def in_manifold_eigen(self, L, eps=1e-6):
        r"""
        Checks that an ascending ordered vector of eigenvalues is in the manifold.

        Args:
            L (torch.Tensor): Vector of eigenvalues of shape `(*, rank)`
            eps (float): Optional. Threshold at which the eigenvalues are
                considered to be zero
                Default: ``1e-6``
        """
        if L.size()[:-1] != self.tensorial_size:
            return False
        if L.size(-1) > self.rank:
            # We compute the \infty-norm of the remaining dimension
            D = L[..., : -self.rank]
            infty_norm_err = D.abs().max(dim=-1).values
            if (infty_norm_err > 5.0 * eps).any():
                return False
        return (L[..., -self.rank :] >= -eps).all().item()

    def in_manifold(self, X, eps=1e-6):
        r"""
        Checks that a matrix is in the manifold.

        Args:
            X (torch.Tensor): The matrix or batch of matrices of shape ``(*, n, n)`` to check.
            eps (float): Optional. Threshold at which the singular values are
                    considered to be zero. Default: ``1e-6``
        """
        size = self.tensorial_size + (self.n, self.n)
        if X.size() != size or not Symmetric.in_manifold(X, eps):
            return False
        L = torch.linalg.eigvalsh(X)
        return self.in_manifold_eigen(L, eps)

    def sample(self, init_=torch.nn.init.xavier_normal_, factorized=False):
        r"""
        Returns a randomly sampled matrix on the manifold as

        .. math::

            WW^\intercal \qquad W_{i,j} \sim \texttt{init_}

        By default ``init\_`` is a (xavier) normal distribution, so that the
        returned matrix follows a Wishart distribution.

        The output of this method can be used to initialize a parametrized tensor
        that has been parametrized with this or any other manifold as::

            >>> layer = nn.Linear(20, 20)
            >>> M = PSSD(layer.weight.size())
            >>> geotorch.register_parametrization(layer, "weight", M)
            >>> layer.weight = M.sample()

        Args:
            init\_ (callable): Optional.
                    A function that takes a tensor and fills it in place according
                    to some distribution. See
                    `torch.init <https://pytorch.org/docs/stable/nn.init.html>`_.
                    Default: ``torch.nn.init.xavier_normal_``
        """
        with torch.no_grad():
            device = self[0].base.device
            dtype = self[0].base.dtype
            X = torch.empty(
                *(self.tensorial_size + (self.n, self.n)), device=device, dtype=dtype
            )
            init_(X)
            X = X @ X.transpose(-2, -1)
            L, Q = torch.linalg.eigh(X)
            L = L[..., -self.rank :]
            Q = Q[..., -self.rank :]
            if factorized:
                return L, Q
            else:
                return (Q * L.unsqueeze(-2)) @ Q.transpose(-2, -1)

    def extra_repr(self):
        return _extra_repr(
            n=self.n,
            rank=self.rank,
            tensorial_size=self.tensorial_size,
            f=self.f,
            no_inv=self.inv is None,
        )
