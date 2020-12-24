import torch
from torch import nn

from .utils import transpose, base, _extra_repr
from .so import SO, uniform_init_, cayley_map
from .skew import Skew

try:
    from torch import matrix_exp as expm
except ImportError:
    from .linalg.expm import expm
from .exceptions import VectorError, InManifoldError


class Stiefel(SO):
    def __init__(self, size, triv="expm"):
        r"""
        Manifold of rectangular orthogonal matrices parametrized as a projection onto the
        first :math:`k` columns from the space of square orthogonal matrices
        :math:`\operatorname{SO}(n)`. The metric considered is the canonical.

        .. note::

            This class is equivalent to :class:`StiefelTall`, but it is faster for the
            case when :math:`n` is of a similar size of :math:`k`. For example,
            :math:`n \leq 4k`.

        Args:
            size (torch.size): Size of the tensor to be parametrized
            triv (str or callable): Optional.
                A map that maps skew-symmetric matrices onto the orthogonal matrices
                surjectively. It can be one of ``["expm", "cayley"]`` or a custom
                callable. Default: ``"expm"``
        """
        super().__init__(size=Stiefel.size_so(size), triv=triv, lower=True)
        self.k = min(size[-1], size[-2])

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
    def initialize_(self, X):
        size = self.tensorial_size + (self.n , self.k)
        if not Stiefel.in_manifold(X, size):
            raise InManifoldError(X, self)
        if self.n != self.k:
            size_n = X.size()[:-2] + (self.n, self.n - self.k)
            # N will be a complement orthogonal to X
            N = X.new_empty(*size_n)
            with torch.no_grad():
                N.normal_()
                # We assume for now that X is orthogonal.
                # This will be checked in super().initialize_()
                # Project N onto the orthogonal complement to X
                # We iterate this twice for this algorithm to be numerically stable
                # This is standard, as done in some stochastic SVD algorithms
                for _ in range(2):
                    N = N - X @ (X.transpose(-2, -1) @ N)
                    # And make it an orthonormal base of the image
                    N = N.qr().Q
                X = torch.cat([X, N], dim=-1)
        return super().initialize_(X)[..., : self.n, : self.k]

    @staticmethod
    def in_manifold(X, size, eps=1e-4):
        return SO.in_manifold(X, size, eps)

    def extra_repr(self):
        return _extra_repr(
            n=self.n, k=self.k, tensorial_size=self.tensorial_size, triv=self.triv
        )


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
                X[:k] += eps * torch.eye(k, k, dtype=X.dtype, device=X.device)
            else:
                size_e = X.size()[:-2] + (k, k)
                eye = (eps * torch.eye(k, dtype=X.dtype, device=X.device)).expand(
                    *size_e
                )
                small = small.unsqueeze_(-1).unsqueeze_(-1).float().expand(*size_e)
                X[..., :k, :k] += small * eye
        return X


class StiefelTall(nn.Module):
    trivializations = {"expm": expm, "cayley": cayley_map}

    def __init__(self, size, triv="expm"):
        r"""
        Manifold of rectangular orthogonal matrices parametrized using its tangent space.
        To parametrize this tangent space we use the orthogonal projection from the ambient
        space :math:`\mathbb{R}^{n \times k}`. The metric considered is the canonical.

        .. note::

            This class is equivalent to :class:`Stiefel`, but it is faster for the case
            when :math:`n` is of a much larger than :math:`k`. For example, :math:`n > 4k`.

        Args:
            size (torch.size): Size of the tensor to be parametrized
            triv (str or callable): Optional.
                A map that maps :math:`\operatorname{Skew}(n)` onto the orthogonal
                matrices surjectively. It can be one of ``["expm", "cayley"]`` or a custom
                callable. Default: ``"expm"``
        """
        super().__init__()
        if torch.__version__ >= "1.7.0":
            cls = self.__class__.__name__
            raise RuntimeError(
                "{} not available in PyTorch 1.7.0, "
                "as it introduced a breaking change in the "
                "gradients of the QR decomposition. "
                "Use {} instead.".format(cls, cls[: -len("Tall")])
            )

        n, k, tensorial_size = StiefelTall.parse_size(size)
        self.n = n
        self.k = k
        self.tensorial_size = tensorial_size
        self.triv = StiefelTall.parse_triv(triv)
        self.register_buffer(
            "base", uniform_init_(torch.empty(*(tensorial_size + (n, k))))
        )

    @classmethod
    def parse_size(cls, size):
        if len(size) < 2:
            raise VectorError(cls.__name__, size)
        n = max(size[-2:])
        k = min(size[-2:])
        tensorial_size = size[:-2]
        return n, k, tensorial_size

    @staticmethod
    def parse_triv(triv):
        if triv in StiefelTall.trivializations.keys():
            return StiefelTall.trivializations[triv]
        elif callable(triv):
            return triv
        else:
            raise ValueError(
                "Argument triv was not recognized and is "
                "not callable. Should be one of {}. Found {}".format(
                    list(StiefelTall.trivializations.keys()), triv
                )
            )

    @transpose
    @base
    def forward(self, X):
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
        where :math:`pi` is the projection of a matrix into its first :math:`k` columns
        """
        Q, R = stable_qr(X)
        AR = torch.cat([A, R], dim=-2)
        Atilde = Stiefel._frame(AR)
        Atilde = Skew.frame(Atilde, lower=True)

        B = self.base
        BQ = torch.cat([B, Q], dim=-1)
        MN = self.triv(Atilde)[..., :, : self.k]
        return BQ @ MN

    @transpose
    def initialize_(self, X):
        if not Stiefel.in_manifold(X, self.base.size()):
            raise InManifoldError(X, self)
        with torch.no_grad():
            self.base.data = X.data
        return torch.zeros_like(X)

    @staticmethod
    def in_manifold(X, size, eps=1e-4):
        return SO.in_manifold(X, size, eps)

    def extra_repr(self):
        return _extra_repr(
            n=self.n, k=self.k, tensorial_size=self.tensorial_size, triv=self.triv
        )
