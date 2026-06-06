import torch
from .glp import GLp
from .fixedrank import FixedRank
from .exceptions import InManifoldError


class SL(GLp):
    def __init__(self, size, f="softplus", triv="expm"):
        r"""
        Manifold of special linear matrices

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
                surjectively. This is used to optimize the :math:`U` and :math:`V` in the
                SVD. It can be one of ``["expm", "cayley"]`` or a custom
                callable. Default: ``"expm"``
        """
        super().__init__(size, SL.parse_f(f), triv)

    @staticmethod
    def parse_f(f_name):
        if f_name in FixedRank.fs.keys():
            f, inv = FixedRank.parse_f(f_name)

            def f_sl(x):
                y = f(x)
                log_y = y.log()
                return (log_y - log_y.mean(dim=-1, keepdim=True)).exp()

            return (f_sl, inv)
        else:
            return f_name

    def in_manifold_singular_values(self, S, eps=5e-3):
        rank_eps = torch.finfo(S.dtype).eps * max(self.n, self.k)
        if not super().in_manifold_singular_values(S, rank_eps):
            return False
        eps = max(eps, 8 * rank_eps**0.5)
        logabsdet = S.log().sum(dim=-1).abs()
        return (logabsdet < eps).all().item()

    def submersion_inv(self, X, check_in_manifold=True):
        if check_in_manifold and not self.in_manifold(X):
            raise InManifoldError(X, self)
        return super().submersion_inv(X, check_in_manifold=False)

    def in_manifold(self, X, eps=5e-3):
        r"""
        Checks that a given matrix is in the manifold.

        Args:
            X (torch.Tensor or tuple): The input matrix or matrices of shape ``(*, n, k)``.
            eps (float): Optional. Threshold at which the singular values are
                    considered to be zero
                    Default: ``5e-3``
        """
        if X.size() != self.tensorial_size + (self.n, self.k):
            return False
        sign, logabsdet = torch.linalg.slogdet(X)
        eps = max(
            eps,
            8 * (torch.finfo(X.dtype).eps * max(self.n, self.k)) ** 0.5,
        )
        return ((sign > 0) & (logabsdet.abs() < eps)).all().item()

    def sample(self, init_=torch.nn.init.xavier_normal_, eps=5e-6, factorized=False):
        r"""
        Returns a randomly sampled matrix on the manifold by sampling a matrix according
        to ``init_`` and projecting it onto the manifold.

        The output of this method can be used to initialize a parametrized tensor
        that has been parametrized with this or any other manifold as::

            >>> layer = nn.Linear(20, 20)
            >>> M = SL(layer.weight.size(), rank=6)
            >>> torch.nn.utils.parametrize.register_parametrization(layer, "weight", M)
            >>> layer.weight = M.sample()

        Args:
            init\_ (callable): Optional. A function that takes a tensor and fills it
                    in place according to some distribution. See
                    `torch.init <https://pytorch.org/docs/stable/nn.init.html>`_.
                    Default: ``torch.nn.init.xavier_normal_``
            eps (float): Optional. Minimum singular value of the sampled matrix.
                    Default: ``5e-6``
        """
        U, S, V = super().sample(factorized=True, init_=init_, eps=eps)
        with torch.no_grad():
            min_singular_value = max(
                eps,
                (torch.finfo(S.dtype).eps * max(self.n, self.k)) ** 0.5,
            )
            S.clamp_min_(min_singular_value)
            log_S = S.log()
            S = (log_S - log_S.mean(dim=-1, keepdim=True)).exp()
            X = (U * S.unsqueeze(-2)) @ V.transpose(-2, -1)
            sign, logabsdet = torch.linalg.slogdet(X)
            X[..., :, 0] *= sign.unsqueeze(-1)
            X *= (-logabsdet / self.n).exp().unsqueeze(-1).unsqueeze(-1)
        return X
