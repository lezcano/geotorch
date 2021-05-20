import torch
from functools import partial

try:
    from torch.linalg import svd

    svd = partial(svd, full_matrices=False)
except ImportError:
    from torch import svd


from .product import ProductManifold
from .stiefel import Stiefel
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
                A map that maps skew-symmetric matrices onto the orthogonal matrices
                surjectively. This is used to optimize the :math:`U` and :math:`V` in
                the SVD. It can be one of ``["expm", "cayley"]`` or a custom callable.
                Default: ``"expm"``
        """
        n, k, tensorial_size, transposed = LowRank.parse_size(size)
        if rank > min(n, k) or rank < 1:
            raise RankError(n, k, rank)
        super().__init__(LowRank.manifolds(n, k, rank, tensorial_size, triv))
        self.n = n
        self.k = k
        self.rank = rank
        self.tensorial_size = tensorial_size
        self.transposed = transposed

    @classmethod
    def parse_size(cls, size):
        if len(size) < 2:
            raise VectorError(cls.__name__, size)
        transposed = size[-2] < size[-1]
        n = max(size[-2:])
        k = min(size[-2:])
        tensorial_size = size[:-2]
        return n, k, tensorial_size, transposed

    @staticmethod
    def manifolds(n, k, rank, tensorial_size, triv):
        size_u = tensorial_size + (n, rank)
        size_s = tensorial_size + (rank,)
        size_v = tensorial_size + (k, rank)
        return Stiefel(size_u, triv), Rn(size_s), Stiefel(size_v, triv)

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
        X = self.frame(X)
        U, S, V = super().forward(X)
        return self.submersion(U, S, V)

    def frame_inv(self, X1, X2, X3):
        with torch.no_grad():
            # X1 is lower-triangular
            # X2 is a vector
            # X3 is upper-triangular
            size = self.tensorial_size + (self.n, self.k)
            ret = torch.zeros(*size, dtype=X1.dtype, device=X1.device)
            ret[..., : self.rank] += X1
            ret[..., : self.rank, : self.rank] += torch.diag_embed(X2)
            ret[..., : self.rank, :] += X3.transpose(-2, -1)
        return ret

    def submersion_inv(self, X, check_in_manifold=True):
        if isinstance(X, torch.Tensor):
            U, S, V = svd(X)
            if check_in_manifold and not self.in_manifold_singular_values(S):
                raise InManifoldError(X, self)
        else:
            # We assume that we got he U S V factorized in a tuple / list
            U, S, V = X
            if check_in_manifold and not self.in_manifold_tuple(U, S, V):
                raise InManifoldError(X, self)
        return U[..., : self.rank], S[..., : self.rank], V[..., : self.rank]

    @transpose
    def right_inverse(self, X, check_in_manifold=True):
        USV = self.submersion_inv(X, check_in_manifold)
        X1, X2, X3 = super().right_inverse(USV, check_in_manifold=False)
        return self.frame_inv(X1, X2, X3)

    def in_manifold_singular_values(self, S, eps=1e-5):
        r"""
        Checks that an ordered vector of singular values is in the manifold.

        For tensors with more than 1 dimension the first dimensions are
        treated as batch dimensions.

        Args:
            S (torch.Tensor): Vector of singular values
            eps (float): Optional. Threshold at which the singular values are
                considered to be zero
                Default: ``1e-5``
        """
        if S.size(-1) <= self.rank:
            return True
        # We compute the \infty-norm of the remaining dimension
        D = S[..., self.rank :]
        infty_norm_err = D.abs().max(dim=-1).values
        return (infty_norm_err < eps).all()

    def in_manifold_tuple(self, U, S, V, eps=1e-5):
        return (
            self.in_manifold_singular_values(S, eps)
            and self[0].in_manifold(U)
            and self[1].in_manifold(S)
            and self[2].in_manifold(V)
        )

    def in_manifold(self, X, eps=1e-5):
        r"""
        Checks that a matrix is in the manifold. The matrix may be given
        factorized in a `3`-tuple :math:`(U, \Sigma, V)` of a matrix, vector,
        and matrix representing an SVD of the matrix.


        For tensors with more than 2 dimensions the first dimensions are
        treated as batch dimensions.

        Args:
            X (torch.Tensor or tuple): The matrix to be checked or a tuple containing
                :math:`(U, \Sigma, V)` as returned by ``torch.linalg.svd`` or
                ``self.sample(factorized=True)``.
            eps (float): Optional. Threshold at which the singular values are
                    considered to be zero
                    Default: ``1e-5``
        """
        if isinstance(X, tuple):
            if len(X) == 3:
                return self.in_manifold_tuple(X[0], X[1], X[2])
            else:
                return False
        else:
            if X.size(-1) > X.size(-2):
                X = X.transpose(-2, -1)
            if X.size() != self.tensorial_size + (self.n, self.k):
                return False
            try:
                S = torch.linalg.svdvals(X)
            except AttributeError:
                S = svd(X).S
            return self.in_manifold_singular_values(S, eps)

    def project(self, X, factorized=True):
        r"""
        Project a matrix onto the manifold.

        If ``factorized==True``, it returns a tuple containing the SVD decomposition of
        the matrix.

        Args:
            X (torch.Tensor): Matrix to be projected onto the manifold
            factorized (bool): Optional. Return an SVD decomposition of the
                    sampled matrix as a tuple :math:`(U, \Sigma, V)`.
                    Using ``factorized=True`` is more efficient when the result is
                    used to initialize a parametrized tensor.
                    Default: ``True``
        """
        U, S, V = svd(X)
        U, S, V = U[..., : self.rank], S[..., : self.rank], V[..., : self.rank]
        if factorized:
            return U, S, V
        else:
            X = self.submersion(U, S, V)
            if self.transposed:
                X = X.transpose(-2, -1)
            return X

    def sample(self, init_=torch.nn.init.xavier_normal_, factorized=True):
        r"""
        Returns a randomly sampled matrix on the manifold by sampling a matrix according
        to ``init_`` and projecting it onto the manifold.

        The output of this method can be used to initialize a parametrized tensor
        that has been parametrized with this or any other manifold as::

            >>> layer = nn.Linear(20, 20)
            >>> M = LowRank(layer.weight.size(), rank=6)
            >>> geotorch.register_parametrization(layer, "weight", M)
            >>> layer.weight = M.sample()

        Args:
            init\_ (callable): Optional. A function that takes a tensor and fills it
                    in place according to some distribution. See
                    `torch.init <https://pytorch.org/docs/stable/nn.init.html>`_.
                    Default: ``torch.nn.init.xavier_normal_``
            factorized (bool): Optional. Return an SVD decomposition of the
                    sampled matrix as a tuple :math:`(U, \Sigma, V)`.
                    Using ``factorized=True`` is more efficient when the result is
                    used to initialize a parametrized tensor.
                    Default: ``True``
        """
        with torch.no_grad():
            device = self[0].base.device
            dtype = self[0].base.dtype
            X = torch.empty(
                *(self.tensorial_size + (self.n, self.k)), device=device, dtype=dtype
            )
            init_(X)
            U, S, V = svd(X)
            U, S, V = U[..., : self.rank], S[..., : self.rank], V[..., : self.rank]
            if factorized:
                return U, S, V
            else:
                Vt = V.transpose(-2, -1)
                # Multiply the three of them, S as a diagonal matrix
                X = U @ (S.unsqueeze(-1).expand_as(Vt) * Vt)
                if self.transposed:
                    X = X.transpose(-2, -1)
                return X

    def extra_repr(self):
        return _extra_repr(
            n=self.n,
            k=self.k,
            rank=self.rank,
            tensorial_size=self.tensorial_size,
            transposed=self.transposed,
        )
