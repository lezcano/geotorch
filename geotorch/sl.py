import torch
from .glp import GLp
from .fixedrank import FixedRank


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
                return y / y.prod(dim=-1, keepdim=True).pow(1.0 / y.shape[-1])

            return (f_sl, inv)
        else:
            return f_name

    def in_manifold_singular_values(self, S, eps=5e-3):
        print("A")
        if not super().in_manifold_singular_values(S, eps):
            return False
        # We compute the \infty-norm of the determinant minus 1 and should be about zero
        print(S.prod(dim=-1) -1)
        infty_norm = (S.prod(dim=-1) - 1).abs().max(dim=-1).values
        print(eps)
        print(infty_norm)
        print(infty_norm < eps)
        return (infty_norm < eps).all().item()

    def in_manifold(self, X, eps=5e-3):
        r"""
        Checks that a given matrix is in the manifold.

        Args:
            X (torch.Tensor or tuple): The input matrix or matrices of shape ``(*, n, k)``.
            eps (float): Optional. Threshold at which the singular values are
                    considered to be zero
                    Default: ``1e-4``
        """
        # The purpose of this function is just to have a more lax default eps value
        return super().in_manifold(X, eps)

    def sample(self, init_=torch.nn.init.xavier_normal_, eps=5e-6, factorized=False):
        r"""
        Returns a randomly sampled matrix on the manifold by sampling a matrix according
        to ``init_`` and projecting it onto the manifold.

        The output of this method can be used to initialize a parametrized tensor
        that has been parametrized with this or any other manifold as::

            >>> layer = nn.Linear(20, 20)
            >>> M = SL(layer.weight.size(), rank=6)
            >>> geotorch.register_parametrization(layer, "weight", M)
            >>> layer.weight = M.sample()

        Args:
            init\_ (callable): Optional. A function that takes a tensor and fills it
                    in place according to some distribution. See
                    `torch.init <https://pytorch.org/docs/stable/nn.init.html>`_.
                    Default: ``torch.nn.init.xavier_normal_``
            eps (float): Optional. Minimum singular value of the sampled matrix.
                    Default: ``5e-6``
        """
        U, S, V = super().sample(factorized=True, init_=init_)
        with torch.no_grad():
            # S >= 0, as given by torch.linalg.eigvalsh()
            S = S / S.prod(dim=-1, keepdim=True).pow(1.0 / S.shape[-1])
        return (U * S.unsqueeze(-2)) @ V.transpose(-2, -1)
