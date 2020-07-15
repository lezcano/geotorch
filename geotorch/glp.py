from .fixedrank import FixedRank
from .exceptions import VectorError, NonSquareError


class GLp(FixedRank):
    def __init__(self, size, f="softplus"):
        r"""
        Manifold of invertible matrices

        Args:
            size (torch.size): Size of the tensor to be applied to
            f (str or callable): Optional. The string `"softplus"` or a callable
                that maps real numbers to the interval (0, infty). Default: `"softplus"`
        """
        super().__init__(size, GLp.rank(size))

    @classmethod
    def rank(cls, size):
        if len(size) < 2:
            raise VectorError(cls.__name__, size)
        n, k = size[-2:]
        if n != k:
            raise NonSquareError(cls.__name__, size)
        return n
