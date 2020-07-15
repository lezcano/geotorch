from .constructions import AbstractManifold
from .exceptions import NonSquareError


class Symmetric(AbstractManifold):
    def __init__(self, size, lower=True):
        r"""
        Vector space of symmetric matrices, parametrized in terms of the upper or lower
        triangular part of a matrix.

        Args:
            size (torch.size): Size of the tensor to be applied to
            lower (bool): Optional. Uses the lower triangular part of the matrix to
                parametrize the matrix. Default: `True`
        """
        super().__init__(dimensions=2, size=size)
        if self.n != self.k:
            raise NonSquareError(self.__class__.__name__, size)
        self.lower = lower

    def forward(self, X):
        if self.lower:
            return X.tril(0) + X.tril(-1).transpose(-2, -1)
        else:
            return X.triu(0) + X.triu(1).transpose(-2, -1)

    def extra_repr(self):
        return "n={}".format(self.n)
