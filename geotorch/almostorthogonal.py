import torch
from .lowrank import LowRank


def sigmoid(t):
    return 2.0 * (torch.sigmoid(t) - 0.5)


class AlmostOrthogonal(LowRank):
    fs = {"sigmoid": sigmoid, "tanh": torch.tanh, "sin": torch.sin}

    def __init__(self, size, r, f="sigmoid"):
        r"""
        Manifold of matrices with singular values in a radius :math:`r` around
        :math:`1`, that is, in the interval :math:`[1-r, 1+r]`.

        Args:
            size (torch.size): Size of the tensor to be applied to
            r (float): Radius. A float in the interval [0, 1]
            f (str or callable): Optional. One of `["sigmoid", "tanh", "sin"]`
                or a callable that maps real numbers to the interval [-1, 1].
                Default: `"sigmoid"`
        """
        super().__init__(self, size, AlmostOrthogonal.rank(size))
        if f not in AlmostOrthogonal.fs.keys() and not callable(f):
            raise ValueError(
                "Argument triv was not recognized and is "
                "not callable. Should be one of {}. Found {}".format(
                    list(AlmostOrthogonal.fs.keys()), f
                )
            )

        if r < 0.0 or r > 1.0:
            raise ValueError("The radius has to be between 0 and 1. Got {}".format(r))

        if callable(f):
            self.f = f
        else:
            self.f = AlmostOrthogonal.fs[f]

        self.r = r

    @staticmethod
    def rank(size):
        if len(size) < 2:
            raise ValueError(
                "Cannot instantiate AlmostOrthogonal on a tensor of less than 2 dimensions."
                "Got size {}".format(size)
            )
        return min(*size[-2:])

    def embedding(self, X):
        U, S, V = super().embedding(X)
        return U, 1.0 + self.r * self.f(S), V
