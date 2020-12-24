from .pssdlowrank import PSSDLowRank
from .exceptions import VectorError, NonSquareError
from .utils import _extra_repr


class PSSD(PSSDLowRank):
    def __init__(self, size, triv="expm"):
        r"""
        Manifold of symmetric positive semidefinite matrices

        Args:
            size (torch.size): Size of the tensor to be parametrized
            triv (str or callable): Optional.
                A map that maps skew-symmetric matrices onto the orthogonal matrices
                matrices surjectively. This is used to optimize the :math:`Q` in the eigenvalue
                decomposition. It can be one of ``["expm", "cayley"]`` or a custom
                callable. Default: ``"expm"``
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

    def extra_repr(self):
        return _extra_repr(n=self.n, tensorial_size=self.tensorial_size)
