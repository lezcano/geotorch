import torch
from torch import nn
from .exceptions import VectorError, NonSquareError, InManifoldError


class Skew(nn.Module):
    def __init__(self, size, lower=True):
        r"""
        Vector space of skew-symmetric matrices, parametrized in terms of
        the upper or lower triangular part of a matrix.

        Args:
            size (torch.size): Size of the tensor to be parametrized
            lower (bool): Optional. Uses the lower triangular part of the matrix
                to parametrize the matrix. Default: ``True``
        """
        super().__init__()
        n, tensorial_size = Skew.parse_size(size)
        self.n = n
        self.tensorial_size = tensorial_size
        self.lower = lower

    @classmethod
    def parse_size(cls, size):
        if len(size) < 2:
            raise VectorError(cls.__name__, size)
        n, k = size[-2:]
        tensorial_size = size[:-2]
        if n != k:
            raise NonSquareError(cls.__name__, size)
        return n, tensorial_size

    @staticmethod
    def frame(X, lower):
        if lower:
            X = X.tril(-1)
        else:
            X = X.triu(1)
        return X - X.transpose(-2, -1)

    def forward(self, X):
        if len(X.size()) < 2:
            raise VectorError(type(self).__name__, X.size())
        if X.size(-2) != X.size(-1):
            raise NonSquareError(type(self).__name__, X.size())
        return self.frame(X, self.lower)

    def right_inverse(self, X, check_in_manifold=True, tol=1e-4):
        if check_in_manifold and not torch.allclose(X, -X.mT, atol=tol):
            raise InManifoldError(X, self)
        # We assume that X is skew_symmetric
        if self.lower:
            return X.tril(-1)
        else:
            return X.triu(1)

    @staticmethod
    def in_manifold(X):
        return (
            X.dim() >= 2
            and X.size(-2) == X.size(-1)
            and torch.allclose(X, -X.transpose(-2, -1))
        )

    def sample(self, init_=nn.init.xavier_normal_, lower=True):
        r"""
        Returns a randomly sampled matrix on the manifold as

        .. math::

            tril(W) \qquad W_{i,j} \sim \texttt{init_} if lower is set to True
            tiu(W)  \qquad W_{i,j} \sim \texttt{init_} otherwise

        By default ``init\_`` is a (xavier) normal distribution, so that the
        returned matrix follows a Wishart distribution.

        The output of this method can be used to initialize a parametrized tensor
        that has been parametrized with this or any other manifold as::

            >>> layer = nn.Linear(20, 20)
            >>> M = Skew(layer.weight.size())
            >>> geotorch.register_parametrization(layer, "weight", M)
            >>> layer.weight = M.sample()

        Args:
            init\_ (callable): Optional.
                    A function that takes a tensor and fills it in place according
                    to some distribution. See
                    `torch.init <https://pytorch.org/docs/stable/nn.init.html>`_.
                    Default: ``torch.nn.init.xavier_normal_``
        """
        with torch.no_grad():
            X = torch.empty(*(self.tensorial_size + (self.n, self.n)))
            init_(X)
            if lower:
                X.tril_(-1)
            else:
                X.triu_(1)
            return X - X.mT
