import torch

try:
    from torch.linalg import qr
except ImportError:
    from torch import qr


from .utils import transpose, _extra_repr
from .so import SO, _has_orthonormal_columns

from .exceptions import VectorError, InManifoldError


class Stiefel(SO):
    def __init__(self, size, triv="expm"):
        r"""
        Manifold of rectangular orthogonal matrices parametrized as a projection
        onto the first :math:`k` columns from the space of square orthogonal matrices
        :math:`\operatorname{SO}(n)`. The metric considered is the canonical.

        Args:
            size (torch.size): Size of the tensor to be parametrized
            triv (str or callable): Optional.
                A map that maps skew-symmetric matrices onto the orthogonal matrices
                surjectively. It can be one of ``["expm", "cayley"]`` or a custom
                callable. Default: ``"expm"``
        """
        super().__init__(size=Stiefel.size_so(size), triv=triv, lower=True)
        self.k = min(size[-1], size[-2])
        self.transposed = size[-2] < size[-1]

    @classmethod
    def size_so(cls, size):
        if len(size) < 2:
            raise VectorError(cls.__name__, size)
        size_so = list(size)
        size_so[-1] = size_so[-2] = max(size[-1], size[-2])
        return tuple(size_so)

    def frame(self, X):
        n, k = X.size(-2), X.size(-1)
        size_z = X.size()[:-2] + (n, n - k)
        return torch.cat([X, X.new_zeros(*size_z)], dim=-1)

    @transpose
    def forward(self, X):
        X = self.frame(X)
        X = super().forward(X)
        return X[..., : self.k]

    @transpose
    def right_inverse(self, X, check_in_manifold=True):
        if check_in_manifold and not self.in_manifold(X):
            raise InManifoldError(X, self)
        if self.n != self.k:
            # N will be a completion of X to an orthogonal basis of R^n
            N = X.new_empty(*(self.tensorial_size + (self.n, self.n - self.k)))
            with torch.no_grad():
                N.normal_()
                # We assume for now that X is orthogonal.
                # This will be checked in super().right_inverse()
                # Project N onto the orthogonal complement to X
                # We iterate this twice for this algorithm to be numerically stable
                # This is standard, as done in some stochastic SVD algorithms
                for _ in range(2):
                    N = N - X @ (X.transpose(-2, -1) @ N)
                    # And make it an orthonormal base of the image
                    N = qr(N).Q
                X = torch.cat([X, N], dim=-1)
        return super().right_inverse(X, check_in_manifold=False)[..., : self.k]

    def in_manifold(self, X, eps=1e-4):
        r"""
        Checks that a matrix is in the manifold.

        For tensors with more than 2 dimensions the first dimensions are
        treated as batch dimensions.

        Args:
            X (torch.Tensor): The matrix to be checked
            eps (float): Optional. Tolerance to numerical errors.
                Default: ``1e-4``
        """
        if X.size(-1) > X.size(-2):
            X = X.transpose(-2, -1)
        if X.size() != self.tensorial_size + (self.n, self.k):
            return False
        return _has_orthonormal_columns(X, eps)

    def sample(self, distribution="uniform", init_=None):
        r"""
        Returns a randomly sampled orthogonal matrix according to the specified
        ``distribution``. The options are:

            - ``"uniform"``: Samples a tensor distributed according to the Haar measure
              on :math:`\operatorname{SO}(n)`

            - ``"torus"``: Samples a block-diagonal skew-symmetric matrix.
              The blocks are of the form
              :math:`\begin{pmatrix} 0 & b \\ -b & 0\end{pmatrix}` where :math:`b` is
              distributed according to ``init_``. This matrix will be then projected
              onto :math:`\operatorname{SO}(n)` using ``self.triv``

        .. note

            The ``"torus"`` initialization is particularly useful in recurrent kernels
            of RNNs

        Args:
            distribution (string): Optional. One of ``["uniform", "torus"]``.
                    Default: ``"uniform"``
            init\_ (callable): Optional. To be used with the ``"torus"`` option.
                    A function that takes a tensor and fills it in place according
                    to some distribution. See
                    `torch.init <https://pytorch.org/docs/stable/nn.init.html>`_.
                    Default: :math:`\operatorname{Uniform}(-\pi, \pi)`
        """
        X = super().sample(distribution, init_)
        if not self.transposed:
            return X[..., : self.k]
        else:
            return X[..., : self.k, :]

    def extra_repr(self):
        return _extra_repr(
            n=self.n,
            k=self.k,
            tensorial_size=self.tensorial_size,
            triv=self.triv,
            transposed=self.transposed,
        )
