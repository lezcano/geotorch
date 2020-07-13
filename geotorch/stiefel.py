import torch
import torch.nn as nn

from .constructions import Manifold, Fibration
from .so import SO, uniform_init_, cayley_map
from .linalg.expm import expm


class Stiefel(Fibration):
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

    @staticmethod
    def size_so(size):
        if len(size) < 2:
            raise ValueError(
                "Cannot instantiate Stiefel on a tensor of less than 2 dimensions."
                "Got size {}".format(size)
            )
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

    def fibration(self, X):
        return X[..., :, : self.k]

    def uniform_init_(self):
        r""" Samples an orthogonal matrix uniformly at random according
        to the Haar measure on :math:`\operatorname{St}(n,k)`."""
        self.total_space.uniform_init_()

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
        eps = k * 1e-8
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

        size_z = size[:-2] + (self.k, self.k)
        self.register_parameter("fibr_aux", nn.Parameter(torch.zeros(*size_z)))
        self.uniform_init_()

    def trivialization(self, X):
        # We compute the exponential map as per Edelman
        # This also works for the Cayley
        # Note that this Cayley map is not the same as that of Wen & Yin

        if torch.is_grad_enabled():
            non_singular_(X)
        # Equivalent to (in the paper):
        # Id = torch.eye(n)
        # IBBt = Id - B @ B.t()
        # delta = B @ A + IBBt @ X
        # Q, R = torch.qr(IBBt @ delta)
        B = self.base
        Q, R = stable_qr(X - B @ (B.transpose(-2, -1) @ X))
        # Form
        # A \in Skew(k)
        # Atilde = [[A, -R.t()],
        #           [R,  0    ]] \in Skew(2k)
        A = self.fibr_aux.tril(-1)
        z_size = self.tensorial_size + (2 * self.k, self.k)
        Atilde = torch.cat([torch.cat([A, R], dim=-2), X.new_zeros(*z_size)], dim=-1)
        Atilde = Atilde - Atilde.transpose(-2, -1)

        BQ = torch.cat([B, Q], dim=-1)
        MN = self.triv(Atilde)[..., :, : self.k]
        return BQ @ MN

    def update_base(self, zero=True):
        super().update_base(zero)
        with torch.no_grad():
            self.fibr_aux.zero_()

    def uniform_init_(self):
        r""" Samples an orthogonal matrix uniformly at random according
        to the Haar measure on :math:`\operatorname{St}(n,k)`."""
        with torch.no_grad():
            uniform_init_(self.base)
            if self.is_registered():
                self.original_tensor().zero_()
            self.fibr_aux.zero_()

    def extra_repr(self):
        return super().extra_repr() + ", triv={}".format(self.triv.__name__)
