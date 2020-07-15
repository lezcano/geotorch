import torch
from .lowrank import LowRank


class FixedRank(LowRank):
    fs = {"softplus": torch.nn.functional.softplus}

    def __init__(self, size, rank, f="softplus"):
        r"""
        Manifold of full-rank matrices

        Args:
            size (torch.size): Size of the tensor to be applied to
            rank (int): Rank of the matrices.
                It has to be less or equal to
                :math:`\min(\texttt{size}[-1], \texttt{size}[-2])`
            f (str or callable): Optional. The string `"softplus"` or a callable
                that maps real numbers to the interval (0, infty). Default: `"softplus"`
        """
        super().__init__(size, rank)
        if f not in FixedRank.fs.keys() and not callable(f):
            raise ValueError(
                "Argument triv was not recognized and is "
                "not callable. Should be one of {}. Found {}".format(
                    list(FixedRank.fs.keys()), f
                )
            )

        if callable(f):
            self.f = f
        else:
            self.f = FixedRank.fs[f]

    def embedding(self, X):
        U, S, V = super().embedding(X)
        return U, self.f(S), V
