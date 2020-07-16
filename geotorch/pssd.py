from .pssdlowrank import PSSDLowRank
from .exceptions import VectorError, NonSquareError


class PSSD(PSSDLowRank):
    def __init__(self, size, triv="expm"):
        r"""
        Manifold of symmetric positive semidefinite matrices

        Args:
            size (torch.size): Size of the tensor to be applied to
            triv (str or callable): Optional.
                A map that maps :math:`\operatorname{Skew}(n)` onto the orthogonal
                matrices surjectively. This is used to optimize the Q in the eigenvalue
                decomposition. It can be one of `["expm", "cayley"]` or a custom
                callable. Default: `"expm"`
        """
        super().__init__(size, PSSD.rank(size), triv)

    @classmethod
    def rank(cls, size):
        if len(size) < 2:
            raise VectorError(cls.__name__, size)
        n, k = size[-2:]
        if n != k:
            raise NonSquareError(cls.__name__, size)
        return n
