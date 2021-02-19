import torch
from torch import nn
from .exceptions import VectorError, NonSquareError


class Skew(nn.Module):
    def __init__(self, lower=True):
        r"""
        Vector space of skew-symmetric matrices, parametrized in terms of
        the upper or lower triangular part of a matrix.

        Args:
            size (torch.size): Size of the tensor to be parametrized
            lower (bool): Optional. Uses the lower triangular part of the matrix
                to parametrize the matrix. Default: ``True``
        """
        super().__init__()
        self.lower = lower

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

    @staticmethod
    def in_manifold(X):
        return (
            X.dim() >= 2
            and X.size(-2) == X.size(-1)
            and torch.allclose(X, -X.transpose(-2, -1))
        )
