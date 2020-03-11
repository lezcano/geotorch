from abc import abstractmethod
import math
import random
import torch

from manifold import Manifold
from linalg.expm import expm


def cayley_map(X):
    n = X.size(0)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)
    return torch.solve(Id - X, Id + X)[0]


class SO(Manifold):
    trivializations = {
                       "expm": expm,
                       "cayley": cayley_map,
                      }


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

    def frame(self, x, base):
        r"""
        Parametrizes the skew-symmetric matrices in terms of the lower triangle of ``x``
        """
        x = x.triu(diagonal=1)
        x = x - x.t()
        return x

    def trivialization(self, x, base):
        r"""
        Maps the skew-symmetric matrices into SO(n) surjectively
        """
        return base.mm(self.triv(x))

    def uniform_init(self):
        uniform_init_(self.base)
        if "orig" not in self._parametrizations:
            self.orig.data.zero_()

    def torus_init(self, init_=None):
        torus_init_(self.base, init_)
        if "orig" not in self._parametrizations:
            self.orig.data.zero_()

    def extra_repr(self):
        inv_map = {v: k for k, v in SO.trivializations.items()}
        name = inv_map.get(self.triv)
        if name is None:
            name = "custom"
        return 'n={}, triv={}'.format(self.orig.size(0), name)


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

        super().__init__(t.new_empty(n, n))
        if self.k < self.n//2 + self.n%2:
            d = self.n//2 + self.n%2 - self-k
            self.register_parameter("fibr_aux", t.new_zeros(n, d))


    def frame(self, x, base):
        if self.inverted:
            x = x.t()
        low = x[:, :self.n//2].tril(-1)
        up =  x[:, :self.n//2 + self.n%2].triu(1)
        # Compute the reflection of low
        low = low.flip(-1).flip(-2)
        # S is square upper triangular
        S = torch.cat([up, low], dim=1)
        return super().frame(S, base)

    def trivialization(self, x, base):
        # Size of the original matrix
        n, k = self.size
        return super().trivialization(x, base)[:n, :k]

    def update_base(self):
        if "orig" not in self._parameters:
            raise RuntimeError("Parametrization {} has to be applied to a tensor with "
                    "`module.register_parametrization`".format(type(self).__name__))
        v = self.frame(self.orig.data, self.base)
        x = super().trivialization(v, self.base)
        self.base.copy_(x)
        self.orig.data.zero_()

    def uniform_init_(self):
        uniform_init_(self.base)
        self.orig.data.zero_()

    def torus_init_(self, init_=None):
        torus_init_(self.base, init_)
        self.orig.data.zero_()

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
        if self.n < 2 * self.k:
            raise NotImplementedError
        self.base = torch.empty_like(t)
        uniform_init_(self.base)
        # TODO Register

    def frame(self, x, base):
        # Eq. (2.3) in https://arxiv.org/pdf/physics/9806030.pdf
        btx = b.t().mm(x)
        # x - projection to the normal space
        return x - .5*base.mm(btx + btx.t())

    def trivialization(self, x, base):
        # TODO Implement Cayley
        # TODO Implement Exp even
        # We compute the exponential map
        # www.manoptjl.org/stable/manifolds/stiefel/#Base.exp
        # Eq before 2.14
        # https://arxiv.org/pdf/physics/9806030.pdf
        X1 = torch.cat([base, x], dim=1)
        bx = base.t().mm(x)
        xx = x.t().mm(x)
        Id = torch.eye(base.size(1), dtype=base.dtype, device=base.device)
        X2 = torch.cat([torch.cat([bx, -xx], dim=1), torch.cat([Id, bx], dim=1)])
        eX2 = expm(X2)
        embx = expm(-bx)
        zeros = torch.zeros_like(Id)
        X3 = torch.cat([embx, zeros])
        # Order matters
        ret = X1.mm(X2.mm(X3))
        if self.inverted:
            ret = ret.t()
        return ret


    def uniform_init_(self):
        uniform_init_(self.base)
        self.orig.data.zero_()
        raise NotImplementedError()

    def torus_init_(self, init_=None):
        torus_init_(self.base, init_)
        self.orig.data.zero_()
        raise NotImplementedError()

    def extra_repr(self):
        inv_map = {v: k for k, v in SO.trivializations.items()}
        name = inv_map.get(self.triv)
        if name is None:
            name = "custom"
        return 'n={}, k={}, triv={}'.format(self.orig.size(0), self.orig.size(1), name)


def torus_init_(tensor, triv=expm, init_=None):
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
            t = tensor
        else:
            t = torch.new_empty(n,n)

        # First non-central diagonal
        diag_z = torch.zeros(n-1)
        diag_z[::2] = diag
        torch.diag(diag_z, diagonal=1, out=t.data)
        t.data = triv(t.data - t.data.t())[:tensor.size(0), :tensor.size(1)]
        if not square:
            tensor = t.data
    return tensor
