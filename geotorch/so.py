import math
import torch

from .constructions import Manifold
from .skew import Skew
from .linalg.expm import expm


def cayley_map(X):
    n = X.size(0)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)
    return torch.solve(Id - X, Id + X)[0]


class SO(Manifold):
    trivializations = {"expm": expm, "cayley": cayley_map}

    def __init__(self, size, triv="expm", lower=True):
        r"""
        Manifold of square orthogonal matrices with positive determinant parametrized
        in terms of its Lie algebra, the skew-symmetric matrices.

        Args:
            size (torch.size): Size of the tensor to be applied to
            triv (str or callable): Optional.
                A map that maps :math:`\operatorname{Skew}(n)` onto the orthogonal
                matrices surjectively. It can be one of `["expm", "cayley"]` or a custom
                callable. Default: `"expm"`
            lower (bool): Optional. Uses the lower triangular part of the matrix to parametrize
                the skew-symmetric matrices. Default: `True`
        """
        size = SO._parse_size(size)
        super().__init__(dimensions=2, size=size)
        if self.n != self.k:
            raise ValueError(
                "The SO parametrization can just be applied to square matrices. "
                "Got a tensor of size {}".format(self.orig_dim)
            )

        if triv not in SO.trivializations.keys() and not callable(triv):
            raise ValueError(
                "Argument triv was not recognized and is "
                "not callable. Should be one of {}. Found {}".format(
                    list(SO.trivializations.keys()), triv
                )
            )
        if callable(triv):
            self.triv = triv
        else:
            self.triv = SO.trivializations[triv]

        # Precompose with Skew
        self.chain(Skew(size=size, lower=lower))
        self.uniform_init_()

    @staticmethod
    def _parse_size(size):
        if isinstance(size, int):
            return (size, size)
        else:
            return size

    def trivialization(self, X):
        return self.base @ self.triv(X)

    def uniform_init_(self):
        r""" Samples an orthogonal matrix uniformly at random according
        to the Haar measure on :math:`\operatorname{SO}(n)`."""
        with torch.no_grad():
            uniform_init_(self.base)
            if self.is_registered():
                self.original_tensor().zero_()

    def torus_init_(self, init_=None, triv=expm):
        r"""Samples the 2D input `tensor` as a block-diagonal skew-symmetric matrix
        which is skew-symmetric in the main diagonal. The blocks are of the form
        :math:`\begin{pmatrix} 0 & b \\ -b & 0\end{pmatrix}` where :math:`b` is
        distributed according to `init_`. Then it is projected to the manifold using `triv`.

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
        with torch.no_grad():
            torus_init_(self.base, init_, triv)
            if self.is_registered():
                self.original_tensor().zero_()

    def extra_repr(self):
        return "n={}, triv={}".format(self.n, self.triv.__name__)


def uniform_init_(tensor):
    r"""Fills the input with an orthogonal matrix. If square, the matrix will have positive
    determinant.  The input tensor must have at least 2 dimensions, and for tensors with more
    than 2 dimensions the first dimensions are treated as batch dimensions.

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
        x = tensor.new(tensor.size()).normal_(0, 1)
        if transpose:
            x.transpose_(-2, -1)
        q, r = torch.qr(x)

        # Make uniform (diag r >= 0)
        d = r.diagonal(dim1=-2, dim2=-1).sign()
        q *= d.unsqueeze(-2).expand_as(q)
        if transpose:
            q.transpose_(-2, -1)

        if n == k:
            mask = (torch.det(q) > 0.0).float()
            mask[mask == 0.0] = -1.0
            if tensor.ndimension() > 2:
                mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(q)
            q *= mask
        tensor.copy_(q)
        return tensor


def torus_init_(tensor, init_=None, triv=expm):
    r"""Samples the 2D input `tensor` as a block-diagonal skew-symmetric matrix
    which is skew-symmetric in the main diagonal. The blocks are of the form
    :math:`\begin{pmatrix} 0 & b \\ -b & 0\end{pmatrix}` where :math:`b` is
    distributed according to `init_`. Then it is projected to the manifold using `triv`.

    Args:
        tensor (torch.Tensor): a 2-dimensional tensor
        init_: Optional. A function that takes a tensor and fills
                it in place according to some distribution. Default:
                :math:`\mathcal{U}(-\pi, \pi)`
        triv: Optional. A function that maps skew-symmetric matrices
                to orthogonal matrices.
    """
    if tensor.ndimension() < 2 or tensor.size(-1) != tensor.size(-2):
        raise ValueError(
            "Only tensors with 2 or more dimensions which are square in "
            "the last two dimensions are supported. "
            "Got a tensor of shape {}".format(tuple(tensor.size()))
        )

    n, k = tensor.size()[-2:]
    tensorial_size = tensor.size()[:-2]

    if init_ is None:

        def init_(t):
            return torch.nn.init.uniform_(t, -math.pi, math.pi)

    # Non-zero elements that we are going to set on the diagonal
    n_diag = n // 2
    diag = tensor.new(tensorial_size + (n_diag,))
    init_(diag)

    with torch.no_grad():
        # First non-central diagonal
        diag_z = tensor.new_zeros(tensorial_size + (n - 1,))
        diag_z[..., ::2] = diag
        tensor.data = torch.diag_embed(diag_z, offset=-1)
        tensor.data = triv(tensor - tensor.transpose(-2, -1))
    return tensor
