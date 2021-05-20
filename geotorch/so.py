import math
import torch
from torch import nn

try:
    from torch.linalg import qr
except ImportError:
    from torch import qr

from .utils import _extra_repr
from .skew import Skew

try:
    from torch import matrix_exp as expm
except ImportError:
    from .linalg.expm import expm
from .exceptions import NonSquareError, VectorError, InManifoldError


def _has_orthonormal_columns(X, eps):
    k = X.size(-1)
    Id = torch.eye(k, dtype=X.dtype, device=X.device)
    if X.dim() > 2:
        Id = Id.repeat(*(X.size()[:-2] + (1, 1)))
    return torch.allclose(X.transpose(-2, -1) @ X, Id, atol=eps)


def cayley_map(X):
    n = X.size(-1)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)
    if X.ndimension() > 2:
        Id = Id.expand_as(X)
    halfX = 0.5 * X  # To make it into a retraction so that (d\phi)_0 = Id
    return torch.solve(Id + halfX, Id - halfX).solution


class SO(nn.Module):
    trivializations = {"expm": expm, "cayley": cayley_map}

    def __init__(self, size, triv="expm", lower=True):
        r"""
        Manifold of square orthogonal matrices with positive determinant parametrized
        in terms of its Lie algebra, the skew-symmetric matrices.

        Args:
            size (torch.size): Size of the tensor to be parametrized
            triv (str or callable): Optional.
                A map that maps skew-symmetric onto :math:`\operatorname{SO}(n)`
                surjectively. It can be one of ``["expm", "cayley"]`` or a custom
                callable. Default: ``"expm"``
            lower (bool): Optional. Uses the lower triangular part of the matrix to
                parametrize the skew-symmetric matrices. Default: ``True``
        """
        super().__init__()
        n, tensorial_size = SO.parse_size(size)
        self.n = n
        self.tensorial_size = tensorial_size
        self.lower = lower
        self.triv = SO.parse_triv(triv)
        self.register_buffer(
            "base", torch.empty(*(self.tensorial_size + (self.n, self.n)))
        )
        uniform_init_(self.base)

    @classmethod
    def parse_size(cls, size):
        if len(size) < 2:
            raise VectorError(cls.__name__, size)
        n = max(size[-2:])
        k = min(size[-2:])
        if n != k:
            raise NonSquareError(cls.__name__, size)
        tensorial_size = size[:-2]
        return n, tensorial_size

    @staticmethod
    def parse_triv(triv):
        if triv in SO.trivializations.keys():
            return SO.trivializations[triv]
        elif callable(triv):
            return triv
        else:
            raise ValueError(
                "Argument triv was not recognized and is "
                "not callable. Should be one of {}. Found {}".format(
                    list(SO.trivializations.keys()), triv
                )
            )

    def forward(self, X):
        X = Skew.frame(X, self.lower)
        return self.base @ self.triv(X)

    def right_inverse(self, X, check_in_manifold=True):
        if check_in_manifold and not self.in_manifold(X):
            raise InManifoldError(X, self)
        with torch.no_grad():
            self.base.copy_(X)
        return torch.zeros_like(X)

    def in_manifold(self, X, in_so=False, eps=1e-4):
        r"""
        Checks that a matrix is in the manifold.

        For tensors with more than 2 dimensions the first dimensions are
        treated as batch dimensions.

        Args:
            X (torch.Tensor): The matrix to be checked
            in_so (bool): Optional. Checks that the matrix is orthogonal and
                has positive determinant. Otherwise just orthogonality is checked.
                Default: ``False``
            eps (float): Optional. Tolerance to numerical errors.
                Default: ``1e-4``
        """
        if X.size() != self.base.size():
            return False
        is_orth = _has_orthonormal_columns(X, eps)
        X_in_correct_coset = not in_so or (X.det() > 0.0).all().item()
        return is_orth and X_in_correct_coset

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
        device = self.base.device
        dtype = self.base.dtype
        ret = torch.empty(
            *(self.tensorial_size + (self.n, self.n)), device=device, dtype=dtype
        )
        if distribution == "uniform":
            uniform_init_(ret)
        elif distribution == "torus":
            torus_init_(ret, init_, self.triv)
        else:
            raise ValueError(
                'The ditribution has to be one of ["uniform", "torus"]. '
                "Got {}".format(distribution)
            )
        return ret

    def extra_repr(self):
        return _extra_repr(n=self.n, tensorial_size=self.tensorial_size, triv=self.triv)


def uniform_init_(tensor):
    r"""Fills in the input ``tensor`` in place with an orthogonal matrix.
    If square, the matrix will have positive determinant.
    The tensor will be distributed according to the Haar measure.
    The input tensor must have at least 2 dimensions.
    For tensors with more than 2 dimensions the first dimensions are treated as
    batch dimensions.

    Args:
        tensor (torch.Tensor): a 2-dimensional tensor or a batch of them
    """
    # We re-implement torch.nn.init.orthogonal_, as their treatment of batches
    # is not in a per-matrix base
    if tensor.ndimension() < 2:
        raise ValueError(
            "Only tensors with 2 or more dimensions are supported. "
            "Got a tensor of shape {}".format(tuple(tensor.size()))
        )
    n, k = tensor.size()[-2:]
    transpose = n < k
    with torch.no_grad():
        x = torch.empty_like(tensor).normal_(0, 1)
        if transpose:
            x.transpose_(-2, -1)
        q, r = qr(x)

        # Make uniform (diag r >= 0)
        d = r.diagonal(dim1=-2, dim2=-1).sign()
        q *= d.unsqueeze(-2).expand_as(q)
        if transpose:
            q.transpose_(-2, -1)

        # Make them have positive determinant by multiplying the
        # first column by -1 (does not change the measure)
        if n == k:
            mask = (torch.det(q) > 0.0).float()
            mask[mask == 0.0] = -1.0
            mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(q)
            q[..., 0] *= mask[..., 0]
        tensor.copy_(q)
        return tensor


def torus_init_(tensor, init_=None, triv=expm):
    r"""Fills in the input ``tensor`` in place as a block-diagonal skew-symmetric matrix.
    The blocks are of the form
    :math:`\begin{pmatrix} 0 & b \\ -b & 0\end{pmatrix}` where :math:`b` is
    distributed according to ``init_``.
    This matrix is then projected onto the manifold using ``triv``.

    The input tensor must have at least 2 dimension. For tensors with more than 2 dimensions
    the first dimensions are treated as batch dimensions.

    Args:
        tensor (torch.Tensor): a 2-dimensional tensor
        init\_ (callable): Optional. A function that takes a tensor and fills
                it in place according to some distribution. See
                `torch.init <https://pytorch.org/docs/stable/nn.init.html>`_.
                Default: :math:`\operatorname{Uniform}(-\pi, \pi)`
        triv (callable): Optional. A function that maps skew-symmetric matrices
                to orthogonal matrices.
    """
    if tensor.ndimension() < 2 or tensor.size(-1) != tensor.size(-2):
        raise ValueError(
            "Only tensors with 2 or more dimensions which are square in "
            "the last two dimensions are supported. "
            "Got a tensor of shape {}".format(tuple(tensor.size()))
        )

    n = tensor.size(-2)
    tensorial_size = tensor.size()[:-2]

    # Non-zero elements that we are going to set on the diagonal
    n_diag = n // 2
    diag = tensor.new(tensorial_size + (n_diag,))
    if init_ is None:
        torch.nn.init.uniform_(diag, -math.pi, math.pi)
    else:
        init_(diag)

    with torch.no_grad():
        # First non-central diagonal
        diag_z = tensor.new_zeros(tensorial_size + (n - 1,))
        diag_z[..., ::2] = diag
        x = torch.diag_embed(diag_z, offset=-1)
        tensor.copy_(triv(x - x.transpose(-2, -1)))
    return tensor
