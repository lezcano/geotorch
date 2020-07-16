from .fixedrank import FixedRank
from .exceptions import VectorError, NonSquareError


class GLp(FixedRank):
    def __init__(self, size, f="softplus", triv="expm"):
        r"""
        Manifold of invertible matrices

        Args:
            size (torch.size): Size of the tensor to be applied to
            f (str or callable): Optional. The string `"softplus"` or a callable
                that maps real numbers to the interval (0, infty). Default: `"softplus"`
            triv (str or callable): Optional.
                A map that maps :math:`\operatorname{Skew}(n)` onto the orthogonal
                matrices surjectively. This is used to optimize the U and V in the
                SVD. It can be one of `["expm", "cayley"]` or a custom
                callable. Default: `"expm"`
        """
        super().__init__(size, GLp.rank(size), f, triv)

    @classmethod
    def rank(cls, size):
        if len(size) < 2:
            raise VectorError(cls.__name__, size)
        n, k = size[-2:]
        if n != k:
            raise NonSquareError(cls.__name__, size)
        return n
