import math
import torch
import torch.nn as nn

from .manifold import Manifold
from .linalg.expm import expm


def cayley_map(X):
    n = X.size(0)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)
    return torch.solve(Id - X, Id + X)[0]


class Stiefel(Manifold):
    r"""
    Implement everything as the fibration SO(n) -> St(n,k)
    """
    trivializations = {"expm": expm,
                       "cayley": cayley_map}

    def __init__(self, triv="expm"):
        super().__init__()
        if triv not in Stiefel.trivializations.keys() and not callable(triv):
            raise ValueError("Argument triv was not recognized and is "
                             "not callable. Should be one of {}. Found {}"
                             .format(list(Stiefel.trivializations.keys()), triv))

        if callable(triv):
            self.triv = triv
        else:
            self.triv = Stiefel.trivializations[triv]

    def init(self, t):
        super().init(t)
        self.n, self.k = self.size()
        self.inverted = self.n < self.k
        if self.inverted:
            self.n, self.k = self.k, self.n
            t = t.t()
        self.base = t.new_empty(self.n, self.n)
        self.uniform_init_()

    def trivialization(self, x, base):
        if self.inverted:
            x = x.t()

        n, k = self.n, self.k

        A = x.tril(-1)
        A = torch.cat([A, A.new_zeros(n, n-k)], dim=1)
        A = A - A.t() # \in skew(n)


        ret = base.mm(self.triv(x))[:n,:k]
        if self.inverted:
            ret = ret.t()
        return ret

    def uniform_init_(self):
        with torch.no_grad():
            uniform_init_(self.base)
            self.orig.data.zero_()

    def torus_init_(self, init_=None, triv=expm):
        with torch.no_grad():
            torus_init_(self.base, init_, triv)
            self.orig.data.zero_()

    def extra_repr(self):
        inv_map = {v: k for k, v in Stiefel.trivializations.items()}
        name = inv_map.get(self.triv)
        if name is None:
            name = "custom"
        return 'n={}, k={}, triv={}'.format(self.orig.size(0), self.orig.size(1), name)


class StiefelTall(Manifold):
    """
    Implements St(n,k), 1 <= k <= n/2
    """
    trivializations = ("expm", "cayley")


    def __init__(self, triv="expm"):
        super().__init__()
        if triv not in StiefelTall.trivializations:
            raise ValueError("Argument triv should be one of {}. Found {}"
                             .format(*StiefelTall.trivializations, triv))
        self.triv = triv


    def init(self, t):
        super().init(t)
        self.n, self.k = self.size()
        self.inverted = self.n < self.k
        if self.inverted:
            self.n, self.k = self.k, self.n
            t = t.t()
        self.base = torch.empty_like(t)
        uniform_init_(self.base)
        # Parameters to parametrize a square k x k skew-symmetric matrix
        self.register_parameter("fibr_aux",
                                nn.Parameter(
                                    t.new_zeros(self.k, self.k)))

    def trivialization(self, x, base):
        if self.inverted:
            x = x.t()

        if self.triv == "expm":
            #  We compute the exponential map as per Edelman

            # Rename for convenience
            X, B = x, base
            # Equivalent to (in the paper):
            # Id = torch.eye(n)
            # IBBt = Id - B @ B.t()
            # delta = B @ A + IBBt @ X
            # Q, R = torch.qr(IBBt @ delta)
            Q, R = torch.qr(X - B @ (B.t() @ X))
            # Form
            # A \in Skew(k)
            # Atilde = [[A, -R.t()],
            #           [R,  0    ]] \in Skew(2k)
            A = self.fibr_aux.tril(1)
            Atilde = torch.cat([torch.cat([A, R]),torch.zeros(2*self.k, self.k)], dim=1)
            Atilde = Atilde - Atilde.t()
            BQ = torch.cat([B, Q], dim=1)
            MN = expm(Atilde)[:, :self.k]
            ret = BQ @ MN
        elif self.triv == "cayley":
            raise NotImplementedError()

        if self.inverted:
            ret = ret.t()
        return ret

    def uniform_init_(self):
        with torch.no_grad():
            uniform_init_(self.base)
            self.orig.data.zero_()
            self.fibr_aux.data.zero_()

    def torus_init_(self, init_=None, triv=expm):
        with torch.no_grad():
            torus_init_(self.base, init_, triv)
            self.orig.data.zero_()
            self.fibr_aux.data.zero_()

    def extra_repr(self):
        name = self.triv
        if name not in StiefelTall.trivializations:
            name = "custom"
        return 'n={}, k={}, triv={}'.format(self.orig.size(0), self.orig.size(1), name)


def uniform_init_(tensor):
    r"""Samples the 2D `tensor` uniformly distributed over the orthogonal matrices.
    If `tensor` is square, then it will be distributed over the orthogonal matrices
    with positive determinant

    Args:
        tensor (torch.nn.Tensor): a 2-dimensional matrix
    """
    torch.nn.init.orthogonal_(tensor)
    if tensor.size(0) == tensor.size(1) and torch.det(tensor) < 0.:
        with torch.no_grad():
            tensor.data[0] *= -1.
    return tensor


def torus_init_(tensor, init_=None, triv=expm):
    r"""Samples the 2D input `tensor` as a block-diagonal skew-symmetric matrix
    which is skew-symmetric in the main diagonal. The blocks are of the form
    :math:`\begin{pmatrix} 0 & b \\ -b & 0\end{pmatrix}` where :math:`b` is
    distributed according to `init_`. Then it is projected to the manifold using `triv`.

    Args:
        tensor (torch.nn.Tensor): a 2-dimensional matrix
        triv: Optional. A function that maps skew-symmetric matrices
                to orthogonal matrices.
        init_: Optional. A function that takes a tensor and fills
                it in place according to some distribution. Default:
               :math:`\mathcal{U}(-\pi, \pi)
    """
    if tensor.ndimension() != 2:
        raise ValueError("Expected a square matrix. Got a tensor of shape {}"
                         .format(list(tensor.size())))

    if init_ is None:
        init_ = lambda t: torch.nn.init.uniform_(t, -math.pi, math.pi)

    square = tensor.size(0) == tensor.size(1)
    n = max(tensor.size(0), tensor.size(1))

    # Non-zero elements that we are going to set on the diagonal
    n_diag = n // 2
    diag = tensor.new(n_diag)
    init_(diag)

    with torch.no_grad():
        if square:
            t = tensor.data
        else:
            t = torch.new_empty(n, n)

        # First non-central diagonal
        diag_z = torch.zeros(n-1)
        diag_z[::2] = diag
        torch.diag(diag_z, diagonal=1, out=t)
        t.copy_(triv(t - t.t())[:tensor.size(0), :tensor.size(1)])
        if not square:
            tensor.data = t
    return tensor
