import math
import torch
import torch.nn.utils.parametrization as P

from .manifold import Manifold
from .skew import Skew
from .linalg.expm import expm


def cayley_map(X):
    n = X.size(0)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)
    return torch.solve(Id - X, Id + X)[0]


class SO(Manifold):
    trivializations = {"expm": expm,
                       "cayley": cayley_map}

    def __init__(self, size, triv="expm", lower=True):
        super().__init__(dimensions=2, size=size)
        if self.n != self.k:
            raise ValueError("The SO parametrization can just be applied to square matrices. "
                             "Got a tensor of size {}"
                             .format(self.dim[::-1] if self.transpose else self.dim))

        if triv not in SO.trivializations.keys() and not callable(triv):
            raise ValueError("Argument triv was not recognized and is "
                             "not callable. Should be one of {}. Found {}"
                             .format(list(SO.trivializations.keys()), triv))
        if callable(triv):
            self.triv = triv
        else:
            self.triv = SO.trivializations[triv]

        self.skew = Skew(size=size, lower=lower)
        self.uniform_init_()

    def trivialization(self, X, B):
        X = self.skew(X)
        return B @ self.triv(X)

    def uniform_init_(self):
        with torch.no_grad():
            uniform_init_(self.base)

    def torus_init_(self, init_=None, triv=expm):
        with torch.no_grad():
            torus_init_(self.base, init_, triv)

    def extra_repr(self):
        inv_map = {v: k for k, v in SO.trivializations.items()}
        name = inv_map.get(self.triv, "custom")
        return 'n={}, triv={}'.format(self.n, name)


def uniform_init_(tensor):
    r"""Samples the 2D `tensor` uniformly distributed over the orthogonal matrices.
    If `tensor` is square, then it will be distributed over the orthogonal matrices
    with positive determinant

    Args:
        tensor (torch.nn.Tensor): a 2-dimensional tensor
    """
    if tensor.ndimension() != 2:
        raise ValueError("Expected a matrix. Got a tensor of shape {}"
                         .format(list(tensor.size())))
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
        tensor (torch.nn.Tensor): a 2-dimensional tensor
        triv: Optional. A function that maps skew-symmetric matrices
                to orthogonal matrices.
        init_: Optional. A function that takes a tensor and fills
                it in place according to some distribution. Default:
               :math:`\mathcal{U}(-\pi, \pi)
    """
    if tensor.ndimension() != 2:
        raise ValueError("Expected a matrix. Got a tensor of shape {}"
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
