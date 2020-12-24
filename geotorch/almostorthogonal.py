import torch
from .lowrank import LowRank
from .exceptions import VectorError, InManifoldError
from .utils import _extra_repr


def scaled_sigmoid(t):
    return 2.0 * (torch.sigmoid(t) - 0.5)


def inv_scaled_sigmoid(t):
    y = 0.5 * t + 0.5
    return torch.log(y / (1.0 - y))


class AlmostOrthogonal(LowRank):
    fs = {
        "scaled_sigmoid": (scaled_sigmoid, inv_scaled_sigmoid),
        "tanh": (torch.tanh, torch.atanh),
        "sin": (torch.sin, torch.asin),
    }

    def __init__(self, size, lam, f="sin", inverse=None, triv="expm"):
        r"""Manifold of matrices with singular values in the interval
        :math:`(1-\lambda, 1+\lambda)`.

        The possible default maps are the :math:`\sin,\,\tanh` functions and a scaled
        sigmoid. The sigmoid is scaled as :math:`\operatorname{scaled\_sigmoid}(x) = 2\sigma(x) - 1`
        where :math:`\sigma` is the usual sigmoid function.
        This is dones so that the image of the scaled sigmoid is :math:`(-1, 1)`.

        Args:
            size (torch.size): Size of the tensor to be parametrized
            lam (float): Radius. A float in the interval :math:`[0, 1]`
            f (str or callable or tuple of callables): Optional. Either:

                - One of ``["scaled_sigmoid", "tanh", "sin"]``

                - A callable that maps real numbers to the interval :math:`(-1, 1)`.

                - A tuple of callables such that the first maps the real numbers to
                  :math:`(-1, 1)` and the second is a (right) inverse of the first
                Default: ``"sin"``
            triv (str or callable): Optional.
                A map that maps skew-symmetric matrices onto the orthogonal matrices
                surjectively. This is used to optimize the :math:`Q` in the eigenvalue
                decomposition. It can be one of ``["expm", "cayley"]`` or a custom
                callable. Default: ``"expm"``

        """
        super().__init__(size, AlmostOrthogonal.rank(size), triv=triv)
        if lam < 0.0 or lam > 1.0:
            raise ValueError("The radius has to be between 0 and 1. Got {}".format(lam))
        self.lam = lam
        f, inv = AlmostOrthogonal.parse_f(f)
        self.f = f
        self.inv = inv

    @staticmethod
    def parse_f(f):
        if f in AlmostOrthogonal.fs.keys():
            return AlmostOrthogonal.fs[f]
        elif callable(f):
            return f, None
        elif isinstance(f, tuple) and callable(f[0]) and callable(f[1]):
            return f
        else:
            raise ValueError(
                "Argument f was not recognized and is "
                "not callable or a tuple of callables. "
                "Should be one of {}. Found {}".format(
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

    def submersion_inv(self, X):
        U, S, V = super().submersion_inv(X)
        with torch.no_grad():
            S = (S - 1.0) / self.lam
        if (S.abs() > 1.0).any():
            raise InManifoldError(X, self)
        return U, self.inv(S), V

    def extra_repr(self):
        return _extra_repr(
            n=self.n,
            k=self.k,
            rank=self.rank,
            tensorial_size=self.tensorial_size,
            f=self.f,
            no_inv=self.inv is None,
        )
