from torch import nn
from .utils import _extra_repr
from .exceptions import InManifoldError


class Rn(nn.Module):
    def __init__(self, size):
        r"""
        Vector space of unconstrained vectors.

        Args:
            size (torch.size): Size of the tensor to be parametrized
        """
        super().__init__()
        self.n = size[-1]
        self.tensorial_size = size[:-1]

    def forward(self, X):
        return X

    def right_inverse(self, X, check_in_manifold=True):
        if check_in_manifold and not self.in_manifold(X):
            raise InManifoldError(X, self)
        return X

    def in_manifold(self, X):
        return X.size() == self.tensorial_size + (self.n,)

    def extra_repr(self):
        return _extra_repr(n=self.n, tensorial_size=self.tensorial_size)
