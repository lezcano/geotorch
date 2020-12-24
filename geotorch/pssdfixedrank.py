from .symmetric import SymF
from .fixedrank import softplus_epsilon, inv_softplus_epsilon
from .utils import _extra_repr


class PSSDFixedRank(SymF):
    fs = {"softplus": (softplus_epsilon, inv_softplus_epsilon)}

    def __init__(self, size, rank, f="softplus", triv="expm"):
        r"""
        Manifold of symmetric positive semidefinite matrices of rank :math:`r`.

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
                surjectively. This is used to optimize the :math:`Q` in the eigenvalue
                decomposition. It can be one of ``["expm", "cayley"]`` or a custom
                callable. Default: ``"expm"``
        """
        super().__init__(size, rank, PSSDFixedRank.parse_f(f), triv)

    @staticmethod
    def parse_f(f):
        if f in PSSDFixedRank.fs.keys():
            return PSSDFixedRank.fs[f]
        elif callable(f):
            return f, None
        elif isinstance(f, tuple) and callable(f[0]) and callable(f[1]):
            return f
        else:
            raise ValueError(
                "Argument f was not recognized and is "
                "not callable or a tuple of callables. "
                "Should be one of {}. Found {}".format(list(PSSDFixedRank.fs.keys()), f)
            )

    def extra_repr(self):
        return _extra_repr(
            n=self.n,
            tensorial_size=self.tensorial_size,
            f=self.f,
            no_inv=self.inv is None,
        )
