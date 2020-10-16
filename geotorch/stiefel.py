import torch

from .constructions import Manifold, FibredSpace
from .so import SO, uniform_init_, torus_init_, cayley_map
from .linalg.expm import expm
from .exceptions import VectorError


class Stiefel(FibredSpace):
    def __init__(self, size, triv="expm"):
        r"""
        Manifold of rectangular orthogonal matrices parametrized as a projection from
        the square orthogonal matrices :math:`\operatorname{SO}(n)`.
        The metric considered is the canonical.

        .. note::

            This class is equivalent to :class:`StiefelTall`, but it is faster for the
            case when :math:`n` is of a similar size of `k`. For example,
            :math:`n \leq 4k`.

        Args:
            size (torch.size): Size of the tensor to be applied to
            triv (str or callable): Optional.
                A map that maps :math:`\operatorname{Skew}(n)` onto the orthogonal
                matrices surjectively. It can be one of `["expm", "cayley"]` or a custom
                callable. Default: `"expm"`
        """

        super().__init__(
            dimensions=2,
            size=size,
            total_space=SO(size=Stiefel.size_so(size), triv=triv, lower=True),
        )
        self.triv = triv

    @classmethod
    def size_so(cls, size):
        if len(size) < 2:
            raise VectorError(cls.__name__, size)
        size_so = list(size)
        size_so[-1] = size_so[-2] = max(size[-1], size[-2])
        return tuple(size_so)

    def embedding(self, X):
        # Returns the matrix
        # A = [ X 0 ] \in R^{n x n}
        # The skew symmetric embedding will make it into A - A^t, which is an
        # identification of T_pSt(n,k) as a subset of Skew(n) via left-invariant
        # vector fields
        size_z = self.tensorial_size + (self.n, self.n - self.k)
        return torch.cat([X, X.new_zeros(*size_z)], dim=-1)

    def submersion(self, X):
        return X[..., :, : self.k]

    def uniform_init_(self):
        r"""Samples an orthogonal matrix uniformly at random according
        to the Haar measure on :math:`\operatorname{St}(n,k)`."""
        self.total_space.uniform_init_()

    def torus_init_(self, init_=None, triv=expm):
        r"""Samples the 2D input `tensor` as a block-diagonal skew-symmetric matrix
        which is skew-symmetric in the main diagonal. The blocks are of the form
        :math:`\begin{pmatrix} 0 & b \\ -b & 0\end{pmatrix}` where :math:`b` is
        distributed according to `init_`. Then it is projected to the manifold using `triv`.

        .. warning::

            This initialization is just accessible whenever the underlying matrix is square

        .. note::

            This initialization is particularly useful for regularizing RNNs.

        Args:
            init_: Optional. A function that takes a tensor and fills
                    it in place according to some distribution. See
                    `torch.init <https://pytorch.org/docs/stable/nn.init.html?highlight=init>`_.
                    Default: :math:`\operatorname{Uniform}(-\pi, \pi)`
            triv: Optional. A function that maps skew-symmetric matrices
                    to orthogonal matrices.
        """
        if self.k != self.n:
            raise RuntimeError(
                "This initialization is just available in square matrices."
                "This matrix has dimensions ({}, {})".format(self.n, self.k)
            )

        with torch.no_grad():
            torus_init_(self.base, init_, triv)
            if self.is_registered():
                self.original_tensor().zero_()

    def extra_repr(self):
        return super().extra_repr() + ", triv={}".format(self.triv)


def stable_qr(X):
    # Make the QR decomposition unique provided X is non-singular
    # so that no subgradients are needed
    # This should be done with QR with pivoting...
    Q, R = torch.qr(X)
    d = R.diagonal(dim1=-2, dim2=-1).sign()
    return Q * d.unsqueeze(-2).expand_as(Q), R * d.unsqueeze(-1).expand_as(R)


def non_singular_(X):
    # This should be done with QR with pivoting...
    # If this works it's because the gradients of the QR in
    # PyTorch are not correctly implemented at zero... Check that
    with torch.no_grad():
        n, k = X.size()[-2:]
        eps = k * 1e-7
        # If it's close to zero, we give it a wiggle
        small = X.norm(dim=(-2, -1)) < eps
        if small.any():
            if X.ndimension() == 2:
                X[:k] += eps * torch.eye(k, k)
            else:
                size_e = X.size()[:-2] + (k, k)
                eye = (eps * torch.eye(k)).expand(*size_e)
                small = small.unsqueeze_(-1).unsqueeze_(-1).float().expand(*size_e)
                X[..., :k, :k] += small * eye
        return X


class StiefelTall(Manifold):
    trivializations = {"expm": expm, "cayley": cayley_map}

    def __init__(self, size, triv="expm"):
        r"""
        Manifold of rectangular orthogonal matrices parametrized using its tangent space.
        To parametrize this tangent space we use the orthogonal projection from the ambient
        space :math:`\mathbb{R}^{n \times k}`. The metric considered is the canonical.

        .. note::

            This class is equivalent to :class:`Stiefel`, but it is faster for the case
            when :math:`n` is of a much larger than `k`. For example, :math:`n > 4k`.

        Args:
            size (torch.size): Size of the tensor to be applied to
            triv (str or callable): Optional.
                A map that maps :math:`\operatorname{Skew}(n)` onto the orthogonal
                matrices surjectively. It can be one of `["expm", "cayley"]` or a custom
                callable. Default: `"expm"`
        """
        super().__init__(dimensions=2, size=size)
        if triv not in StiefelTall.trivializations.keys() and not callable(triv):
            raise ValueError(
                "Argument triv was not recognized and is "
                "not callable. Should be one of {}. Found {}".format(
                    list(StiefelTall.trivializations.keys()), triv
                )
            )

        if callable(triv):
            self.triv = triv
        else:
            self.triv = StiefelTall.trivializations[triv]
        self.uniform_init_()

    def trivialization(self, X):
        # We compute the exponential map as per Edelman
        # This also works for the Cayley
        # Note that this Cayley map is not the same as that of Wen & Yin

        if torch.is_grad_enabled():
            non_singular_(X)
        # Equivalent to (in the paper):
        # (Id - B @ B.t())X
        B = self.base
        BtX = B.transpose(-2, -1) @ X
        X = X - B @ BtX
        A = BtX.tril(-1)
        return self._expm_aux(X, A)

    def _expm_aux(self, X, A):
        r"""Forms
        :math:`X \in \mathbb{R}^{n x k}`
        :math:`Q, R = qr(X)`
        :math:`A` is lower-triangular
        :math:`hat{A} = A - A.t() \in \Skew(n)`
        :math:`Atilde = [[hat{A}, -R.t()],
                         [R,        0   ]] in Skew(2k)`
        and returns
        :math:`\pi([B, Q] expm(Atilde))`
        where `pi` is the projection of a matrix into its first :math:`k` columns
        """
        Q, R = stable_qr(X)
        z_size = self.tensorial_size + (2 * self.k, self.k)
        Atilde = torch.cat([torch.cat([A, R], dim=-2), X.new_zeros(*z_size)], dim=-1)
        Atilde = Atilde - Atilde.transpose(-2, -1)

        B = self.base
        BQ = torch.cat([B, Q], dim=-1)
        MN = self.triv(Atilde)[..., :, : self.k]
        return BQ @ MN

    def uniform_init_(self):
        r"""Samples an orthogonal matrix uniformly at random according
        to the Haar measure on :math:`\operatorname{St}(n,k)`."""
        with torch.no_grad():
            uniform_init_(self.base)
            if self.is_registered():
                self.original_tensor().zero_()

    def torus_init_(self, init_=None, triv=expm):
        r"""Samples the 2D input `tensor` as a block-diagonal skew-symmetric matrix
        which is skew-symmetric in the main diagonal. The blocks are of the form
        :math:`\begin{pmatrix} 0 & b \\ -b & 0\end{pmatrix}` where :math:`b` is
        distributed according to `init_`. Then it is projected to the manifold using `triv`.

        .. warning::

            This initialization is just accessible whenever the underlying matrix is square

        .. note::

            This initialization is particularly useful for regularizing RNNs.

        Args:
            init_: Optional. A function that takes a tensor and fills
                    it in place according to some distribution. See
                    `torch.init <https://pytorch.org/docs/stable/nn.init.html?highlight=init>`_.
                    Default: :math:`\operatorname{Uniform}(-\pi, \pi)`
            triv: Optional. A function that maps skew-symmetric matrices
                    to orthogonal matrices.
        """
        if self.k != self.n:
            raise RuntimeError(
                "This initialization is just available in square matrices."
                "This matrix has dimensions ({}, {})".format(self.n, self.k)
            )

        with torch.no_grad():
            torus_init_(self.base, init_, triv)
            if self.is_registered():
                self.original_tensor().zero_()

    def extra_repr(self):
        return super().extra_repr() + ", triv={}".format(self.triv.__name__)
