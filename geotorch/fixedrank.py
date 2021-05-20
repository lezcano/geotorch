import torch
from .lowrank import LowRank
from .exceptions import InverseError


def softplus_epsilon(x, epsilon=1e-6):
    return torch.nn.functional.softplus(x) + epsilon


def inv_softplus_epsilon(x, epsilon=1e-6):
    y = x - epsilon
    return torch.where(y > 20, y, y.expm1().log())


class FixedRank(LowRank):
    fs = {"softplus": (softplus_epsilon, inv_softplus_epsilon)}

    def __init__(self, size, rank, f="softplus", triv="expm"):
        r"""
        Manifold of non-square matrices of rank equal to ``rank``

        Args:
            size (torch.size): Size of the tensor to be parametrized
            rank (int): Rank of the matrices.
                It has to be less or equal to
                :math:`\min(\texttt{size}[-1], \texttt{size}[-2])`
            f (str or callable or pair of callables): Optional. Either:

                - ``"softplus"``

                - A callable that maps real numbers to the interval :math:`(0, \infty)`

                - A pair of callables such that the first maps the real numbers onto
                  :math:`(0, \infty)` and the second is a (right) inverse of the first

                Default: ``"softplus"``
            triv (str or callable): Optional.
                A map that maps skew-symmetric matrices onto the orthogonal matrices
                surjectively. This is used to optimize the :math:`U` and :math:`V` in
                the SVD. It can be one of ``["expm", "cayley"]`` or a custom callable.
                Default: ``"expm"``
        """
        super().__init__(size, rank, triv=triv)
        f, inv = FixedRank.parse_f(f)
        self.f = f
        self.inv = inv

    @staticmethod
    def parse_f(f):
        if f in FixedRank.fs.keys():
            return FixedRank.fs[f]
        elif callable(f):
            return f, None
        elif isinstance(f, tuple) and callable(f[0]) and callable(f[1]):
            return f
        else:
            raise ValueError(
                "Argument f was not recognized and is "
                "not callable or a pair of callables. "
                "Should be one of {}. Found {}".format(list(FixedRank.fs.keys()), f)
            )

    def submersion(self, U, S, V):
        return super().submersion(U, self.f(S), V)

    def submersion_inv(self, X, check_in_manifold=True):
        U, S, V = super().submersion_inv(X, check_in_manifold)
        if self.inv is None:
            raise InverseError(self)
        return U, self.inv(S), V

    def in_manifold_singular_values(self, S, eps=1e-5):
        r"""
        Checks that a vector of singular values is in the manifold.

        For tensors with more than 1 dimension the first dimensions are
        treated as batch dimensions.

        Args:
            S (torch.Tensor): Vector of singular values
            eps (float): Optional. Threshold at which the singular values are
                considered to be zero
                Default: ``1e-5``
        """
        if not super().in_manifold_singular_values(S, eps):
            return False
        # We compute the \infty-norm of the eigenvalues
        D = S[..., : self.rank]
        infty_norm = D.abs().max(dim=-1).values
        return (infty_norm > eps).all().item()

    def sample(self, init_=torch.nn.init.xavier_normal_, factorized=True, eps=5e-6):
        r"""
        Returns a randomly sampled matrix on the manifold by sampling a matrix according
        to ``init_`` and projecting it onto the manifold.

        If the sampled matrix has more than `self.rank` small singular values, the
        smallest ones are clamped to be at least ``eps`` in absolute value.

        The output of this method can be used to initialize a parametrized tensor
        that has been parametrized with this or any other manifold as::

            >>> layer = nn.Linear(20, 20)
            >>> M = FixedRank(layer.weight.size(), rank=6)
            >>> geotorch.register_parametrization(layer, "weight", M)
            >>> layer.weight = M.sample()

        Args:
            init\_ (callable): Optional. A function that takes a tensor and fills it
                    in place according to some distribution. See
                    `torch.init <https://pytorch.org/docs/stable/nn.init.html>`_.
                    Default: ``torch.nn.init.xavier_normal_``
            factorized (bool): Optional. Return an SVD decomposition of the
                    sampled matrix as a tuple :math:`(U, \Sigma, V)`.
                    Using ``factorized=True`` is more efficient when the result is
                    used to initialize a parametrized tensor.
                    Default: ``True``
            eps (float): Optional. Minimum singular value of the sampled matrix.
                    Default: ``5e-6``
        """
        U, S, V = super().sample(factorized=True, init_=init_)
        with torch.no_grad():
            # S >= 0, as given by torch.linalg.eigvalsh()
            S[S < eps] = eps
        if factorized:
            return U, S, V
        else:
            Vt = V.transpose(-2, -1)
            # Multiply the three of them, S as a diagonal matrix
            X = U @ (S.unsqueeze(-1).expand_as(Vt) * Vt)
            if self.transposed:
                X = X.transpose(-2, -1)
            return X
