import math
import torch
import torch.nn as nn

from manifold import Manifold
from linalg.expm import expm


def cayley_map(X):
    n = X.size(0)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)
    return torch.solve(Id - X, Id + X)[0]


class SO(Manifold):
    trivializations = {"expm": expm,
                       "cayley": cayley_map}

    def __init__(self, triv="expm"):
        super().__init__()
        if triv not in SO.trivializations.keys() and not callable(triv):
            raise ValueError("Argument triv was not recognized and is "
                             "not callable. Should be one of {}. Found {}"
                             .format(list(SO.trivializations.keys()), triv))

        if callable(triv):
            self.triv = triv
        else:
            self.triv = SO.trivializations[triv]

    def init(self, t):
        if t.ndimension() != 2 or t.size(0) != t.size(1):
            raise ValueError("Expected a square matrix. Got a t of shape {}"
                             .format(list(t.size())))

        self.base = torch.empty_like(t)
        self.uniform_init_()

    def trivialization(self, x, base):
        r"""
        Maps the skew-symmetric matrices into SO(n) surjectively
        """
        x = x.triu(diagonal=1)
        x = x - x.t()
        return base.mm(self.triv(x))

    def uniform_init_(self):
        uniform_init_(self.base)
        self.orig.data.zero_()

    def torus_init_(self, init_=None, triv=expm):
        torus_init_(self.base, init_, triv)
        self.orig.data.zero_()

    def extra_repr(self):
        inv_map = {v: k for k, v in SO.trivializations.items()}
        name = inv_map.get(self.triv)
        if name is None:
            name = "custom"
        return 'n={}, triv={}'.format(self.orig.size(0), name)


def _tall_to_skew(x):
    n = x.size(0)
    low = x[:, :n//2].tril(-1)
    up  = x[:, :n//2 + n%2].triu(1)
    # Compute the reflection of low
    low = low.flip(-1).flip(-2)
    # S is square upper triangular
    return torch.cat([up, low], dim=1)


class Stiefel(SO):
    r"""
    Implement everything as the fibration SO(n) -> St(n,k)
    """

    def init(self, t):
        self.n, self.k = self.size
        self.inverted = self.n < self.k
        if self.inverted:
            self.n, self.k = self.k, self.n
            t = t.t()

        self.small = self.k < self.n//2 + self.n%2
        if self.small:
            d = self.n//2 + self.n%2 - self.k
            self.register_parameter("fibr_aux", nn.Parameter(t.new_zeros(self.n, d)))
        super().init(t.new_empty(self.n, self.n))

    def trivialization(self, x, base):
        if self.inverted:
            x = x.t()
        if self.small:
            x = torch.cat([x, self.fibr_aux], dim=1)
        A = _tall_to_skew(x)

        # Size of the original matrix
        n, k = self.size
        return super().trivialization(A, base)[:n, :k]

    def update_base(self):
        if "orig" not in self._parameters:
            raise RuntimeError("Parametrization {} has to be applied to a tensor with "
                               "`module.register_parametrization`".format(type(self).__name__))
        x = super().trivialization(self.orig.data, self.base)
        self.base.copy_(x)
        self.orig.data.zero_()
        if self.small:
            self.fibr_aux.data.zero_()

    def uniform_init_(self):
        uniform_init_(self.base)
        self.orig.data.zero_()
        if self.small:
            self.fibr_aux.data.zero_()

    def torus_init_(self, init_=None, triv=expm):
        torus_init_(self.base, init_, triv)
        self.orig.data.zero_()
        if self.small:
            self.fibr_aux.data.zero_()

    def extra_repr(self):
        inv_map = {v: k for k, v in SO.trivializations.items()}
        name = inv_map.get(self.triv)
        if name is None:
            name = "custom"
        return 'n={}, k={}, triv={}'.format(self.orig.size(0), self.orig.size(1), name)

class StiefelTall(Manifold):
    """
    Implements St(n,k), 1 <= k <= n/2
    """
    trivializations = {"expm": expm,
                       "cayley": cayley_map,
                      }


    def __init__(self, triv="expm"):
        super().__init__()
        if triv not in StiefelTall.trivializations.keys() and not callable(triv):
            raise ValueError("Argument triv was not recognized and is "
                             "not callable. Should be one of {}. Found {}"
                             .format(list(SO.trivializations.keys()), triv))

        if callable(triv):
            self.triv = triv
        else:
            self.triv = StiefelTall.trivializations[triv]


    def init(self, t):
        self.n, self.k = self.size
        self.inverted = self.n < self.k
        if self.inverted:
            self.n, self.k = self.k, self.n
            t = t.t()
        self.base = torch.empty_like(t)
        uniform_init_(self.base)
        self.register_parameter("fibr_aux",
                                nn.Parameter(
                                    t.new_zeros(self.k, self.k//2 + self.k%2)))

    def trivialization(self, x, base):
        # TODO Implement Cayley
        #  We compute the exponential map as per Edelman

        # Equivalent to:
        # Id = torch.eye(n)
        # IBBt = Id - B @ B.t()
        # delta = B @ A + IBBt @ X
        # Q, R = torch.qr(IBBt @ delta)
        X, B = x, base
        A = _tall_to_skew(self.fibr_aux)
        T = base @ A + X
        Q, R = torch.qr(T - B @ (B.t() @ T))
        Z = torch.zeros(self.k, self.k)
        Atilde = torch.cat([
                   torch.cat([A, -R.t()], dim=1),
                   torch.cat([R,  Z],     dim=1)])
        BQ = torch.cat([B, Q], dim=1)
        MN = expm(Atilde)[:, :self.k]
        ret = BQ @ MN

        if self.inverted:
            ret = ret.t()
        return ret

    def uniform_init_(self):
        uniform_init_(self.base)
        self.orig.data.zero_()
        self.fibr_aux.data.zero_()

    def torus_init_(self, init_=None, triv=expm):
        torus_init_(self.base, init_, triv)
        self.orig.data.zero_()
        self.fibr_aux.data.zero_()

    def extra_repr(self):
        inv_map = {v: k for k, v in SO.trivializations.items()}
        name = inv_map.get(self.triv)
        if name is None:
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
        torch.diag(diag_z, diagonal=1, out=t.data)
        t = triv(t - t.t())[:tensor.size(0), :tensor.size(1)]
        if not square:
            tensor.data = t
    return tensor
