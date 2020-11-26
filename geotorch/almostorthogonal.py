import torch
from .lowrank import LowRank
from .exceptions import VectorError


def sigmoid(t):
    return 2.0 * (torch.sigmoid(t) - 0.5)


class AlmostOrthogonal(LowRank):
    fs = {"sigmoid": sigmoid, "tanh": torch.tanh, "sin": torch.sin}

    def __init__(self, size, lam, f="sin", triv="expm"):
        r"""
        Manifold of matrices with singular values in the interval :math:`(1-\lambda, 1+\lambda)`.

        The possible default maps are the :math:`\sin,\,\tanh` functions and a rescaled
        sigmoid. The sigmoid is rescaled as :math:`\operatorname{sigmoid}(x) = 2\sigma(x) - 1`
        where :math:`\sigma` is the usual sigmoid function.

        Args:
            size (torch.size): Size of the tensor to be applied to
            lam (float): Radius. A float in the interval [0, 1]
            f (str or callable): Optional. One of `["sigmoid", "tanh", "sin"]`
                or a callable that maps real numbers to the interval [-1, 1].
                Default: `"sin"`
            triv (str or callable): Optional.
                A map that maps :math:`\operatorname{Skew}(n)` onto the orthogonal
                matrices surjectively. This is used to optimize the Q in the eigenvalue
                decomposition. It can be one of `["expm", "cayley"]` or a custom
                callable. Default: `"expm"`

        """
        super().__init__(size, AlmostOrthogonal.rank(size), triv=triv)
        if lam < 0.0 or lam > 1.0:
            raise ValueError("The radius has to be between 0 and 1. Got {}".format(lam))
        self.lam = lam
        self.f = AlmostOrthogonal.parse_f(f)

    @staticmethod
    def parse_f(f):
        if f in AlmostOrthogonal.fs.keys():
            return AlmostOrthogonal.fs[f]
        elif callable(f):
            return f
        else:
            raise ValueError(
                "Argument f was not recognized and is "
                "not callable. Should be one of {}. Found {}".format(
                    list(AlmostOrthogonal.fs.keys()), f
                )
            )

    @classmethod
    def rank(cls, size):
        if len(size) < 2:
            raise VectorError(cls.__name__, size)
        return min(*size[-2:])

    def submersion(self, U, S, V):
        S = 1.0 + self.lam * self.f(S)
        return super().submersion(U, S, V)
