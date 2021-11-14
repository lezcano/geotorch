import torch

from .symmetric import SymF
from .fixedrank import softplus_epsilon, inv_softplus_epsilon


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
                "not callable or a pair of callables. "
                "Should be one of {}. Found {}".format(list(PSSDFixedRank.fs.keys()), f)
            )

    def in_manifold_eigen(self, L, eps=1e-6):
        r"""
        Checks that an ascending ordered vector of eigenvalues is in the manifold.

        Args:
            L (torch.Tensor): Vector of eigenvalues of shape `(*, rank)`
            eps (float): Optional. Threshold at which the eigenvalues are
                considered to be zero
                Default: ``1e-6``
        """
        return (
            super().in_manifold_eigen(L, eps)
            and (L[..., -self.rank :] >= eps).all().item()
        )

    def sample(self, init_=torch.nn.init.xavier_normal_, eps=5e-6):
        r"""
        Returns a randomly sampled matrix on the manifold as

        .. math::

            WW^\intercal \qquad W_{i,j} \sim \texttt{init_}

        If the sampled matrix has more than `self.rank` small singular values, the
        smallest ones are clamped to be at least ``eps`` in absolute value.


        The output of this method can be used to initialize a parametrized tensor as::

            >>> layer = nn.Linear(20, 20)
            >>> M = PSSD(layer.weight.size())
            >>> geotorch.register_parametrization(layer, "weight", M)
            >>> layer.weight = M.sample()

        Args:
            init\_ (callable): Optional.
                    A function that takes a tensor and fills it in place according
                    to some distribution. See
                    `torch.init <https://pytorch.org/docs/stable/nn.init.html>`_.
                    Default: ``torch.nn.init.xavier_normal_``
            eps (float): Optional. Minimum eigenvalue of the sampled matrix.
                    Default: ``5e-6``
        """
        L, Q = super().sample(factorized=True, init_=init_)
        with torch.no_grad():
            # L >= 0, as given by torch.linalg.eigvalsh()
            L[L < eps] = eps
        return (Q * L.unsqueeze(-2)) @ Q.transpose(-2, -1)
