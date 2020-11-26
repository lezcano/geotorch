from .symmetric import SymF
from .fixedrank import softplus_epsilon


class PSSDFixedRank(SymF):
    fs = {"softplus": softplus_epsilon}

    def __init__(self, size, rank, f="softplus", triv="expm"):
        r"""
        Manifold of symmetric positive semidefinite matrices of rank :math:`r`.

        Args:
            size (torch.size): Size of the tensor to be applied to
            rank (int): Rank of the matrices.
                It has to be less or equal to
                :math:`\min(\texttt{size}[-1], \texttt{size}[-2])`
            f (str or callable): Optional. The string `"softplus"` or a callable
                that maps real numbers to the interval :math:`(0, infty)`. Default: `"softplus"`
            triv (str or callable): Optional.
                A map that maps :math:`\operatorname{Skew}(n)` onto the orthogonal
                matrices surjectively. This is used to optimize the Q in the eigenvalue
                decomposition. It can be one of `["expm", "cayley"]` or a custom
                callable. Default: `"expm"`
        """
        if f in PSSDFixedRank.fs.keys():
            f = PSSDFixedRank.fs[f]
        elif not callable(f):
            raise ValueError(
                "Argument f was not recognized and is "
                "not callable. Should be one of {}. Found {}".format(
                    list(PSSDFixedRank.fs.keys()), f
                )
            )
        super().__init__(size, rank, f, triv)
