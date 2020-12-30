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

    def in_manifold_eigen(self, L, eps=1e-6):
        r"""
        Checks that an ordered vector of eigenvalues values is in the manifold.

        For tensors with more than 1 dimension the first dimensions are
        treated as batch dimensions.

        Args:
            L (torch.Tensor): Vector of eigenvalues
            eps (float): Optional. Threshold at which the eigenvalues are
                considered to be zero
                Default: ``1e-6``
        """
        return (
            super().in_manifold_eigen(L, eps)
            and (L[..., : self.rank] >= eps).all().item()
        )

    def sample(self, factorized=True, init_=torch.nn.init.xavier_normal_, eps=5e-6):
        r"""
        Returns a randomly sampled matrix on the manifold as

        ..math::

            WW^\intercal \qquad W_{i,j} \sim \texttt{init\_}

        and then projected onto the manifold.

        If the sampled matrix has more than `self.rank` small singular values, the
        smallest ones are clamped to be at least ``eps`` in absolute value.

        Args:
            factorized (bool): Optional. Return the tuple :math:`(\Lambda, Q)` with an
                    eigen-decomposition of the sampled matrix. This can also be used
                    to initialize the layer.
                    Default: ``True``
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
            # S >= 0, as given by torch.symeig()
            small = L < eps
            L[small] = eps
        if factorized:
            return L, Q
        else:
            # Project onto the manifold
            Qt = Q.transpose(-2, -1)
            # Multiply the three of them as Q\LambdaQ^T
            return Q @ (L.unsqueeze(-1).expand_as(Qt) * Qt)
