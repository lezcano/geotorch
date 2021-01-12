from .pssdfixedrank import PSSDFixedRank
from .exceptions import VectorError, NonSquareError
from .utils import _extra_repr


class PSD(PSSDFixedRank):
    def __init__(self, size, f="softplus", triv="expm"):
        r"""
        Manifold of symmetric positive definite matrices

        Args:
            size (torch.size): Size of the tensor to be parametrized
            f (str or callable or pair of callables): Optional. Either:

                - ``"softplus"``

                - A callable that maps real numbers to the interval :math:`(0, \infty)`

                - A pair of callables such that the first maps the real numbers to
                  :math:`(0, \infty)` and the second is a (right) inverse of the first

                Default: ``"softplus"``
            triv (str or callable): Optional.
                A map that maps skew-symmetric matrices onto the orthogonal matrices
                surjectively. This is used to optimize the :math:`Q` in the eigenvalue
                decomposition. It can be one of ``["expm", "cayley"]`` or a custom
                callable. Default: ``"expm"``
        """
        super().__init__(size, PSD.rank(size), f, triv=triv)

    @classmethod
    def rank(cls, size):
        if len(size) < 2:
            raise VectorError(cls.__name__, size)
        n, k = size[-2:]
        if n != k:
            raise NonSquareError(cls.__name__, size)
        return n

    def extra_repr(self):
        return _extra_repr(
            n=self.n,
            tensorial_size=self.tensorial_size,
            f=self.f,
            no_inv=self.inv is None,
        )
