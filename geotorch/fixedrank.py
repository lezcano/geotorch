import torch
from .lowrank import LowRank


def softplus_epsilon(x, epsilon=1e-6):
    return torch.nn.functional.softplus(x) + epsilon


def inv_softplus_epsilon(x, epsilon=1e-6):
    y = x - epsilon
    return torch.where(y > 20, y, y.expm1().log())


class FixedRank(LowRank):
    fs = {"softplus": (softplus_epsilon, inv_softplus_epsilon)}

    def __init__(self, size, rank, f="softplus", triv="expm"):
        r"""
        Manifold of non-square matrices of rank equal to ``rank``

        Args:
            size (torch.size): Size of the tensor to be parametrized
            rank (int): Rank of the matrices.
                It has to be less or equal to
                :math:`\min(\texttt{size}[-1], \texttt{size}[-2])`
            f (str or callable or tuple of callables): Optional. Either:

                - ``"softplus"``

                - A callable that maps real numbers to the interval :math:`(0, \infty)`.

                - A tuple of callables such that the first maps the real numbers to
                  :math:`(0, \infty)` and the second is a (right) inverse of the first
                Default: ``"softplus"``
            triv (str or callable): Optional.
                A map that maps skew-symmetric matrices onto the orthogonal matrices
                surjectively. This is used to optimize the :math:`U` and :math:`V` in the
                SVD. It can be one of ``["expm", "cayley"]`` or a custom
                callable. Default: ``"expm"``
        """
        super().__init__(size, rank, triv=triv)
        f, inv = FixedRank.parse_f(f)
        self.f = f
        self.inv = inv

    @staticmethod
    def parse_f(f):
        if f in FixedRank.fs.keys():
            return FixedRank.fs[f]
        elif callable(f):
            return f, None
        elif isinstance(f, tuple) and callable(f[0]) and callable(f[1]):
            return f
        else:
            raise ValueError(
                "Argument f was not recognized and is "
                "not callable. Should be one of {}. Found {}".format(
                    list(FixedRank.fs.keys()), f
                )
            )

    def submersion(self, U, S, V):
        return super().submersion(U, self.f(S), V)
