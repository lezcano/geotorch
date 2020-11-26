import torch
from .lowrank import LowRank


def softplus_epsilon(x, epsilon=1e-6):
    return torch.nn.functional.softplus(x) + epsilon


class FixedRank(LowRank):
    fs = {"softplus": softplus_epsilon}

    def __init__(self, size, rank, f="softplus", triv="expm"):
        r"""
        Manifold of non-square matrices of rank equal to `rank`

        Args:
            size (torch.size): Size of the tensor to be applied to
            rank (int): Rank of the matrices.
                It has to be less or equal to
                :math:`\min(\texttt{size}[-1], \texttt{size}[-2])`
            f (str or callable): Optional. The string `"softplus"` or a callable
                that maps real numbers to the interval :math:`(0, \infty)`.
                Default: `"softplus"`
            triv (str or callable): Optional.
                A map that maps :math:`\operatorname{Skew}(n)` onto the orthogonal
                matrices surjectively. This is used to optimize the U and V in the
                SVD. It can be one of `["expm", "cayley"]` or a custom
                callable. Default: `"expm"`
        """
        super().__init__(size, rank, triv=triv)
        self.f = FixedRank.parse_f(f)

    @staticmethod
    def parse_f(f):
        if f in FixedRank.fs.keys():
            return FixedRank.fs[f]
        elif callable(f):
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
