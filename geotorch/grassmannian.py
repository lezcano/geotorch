from torch.nn import functional as F

from .stiefel import Stiefel


class Grassmannian(Stiefel):
    def __init__(self, size, triv="expm"):
        r"""
        Grassmannian manifold as a projection from the orthogonal
        matrices :math:`\operatorname{St}(n,k)`.
        The metric considered is the canonical.

        Args:
            size (torch.size): Size of the tensor to be parametrized
            triv (str or callable): Optional.
                A map that maps skew-symmetric matrices onto the orthogonal matrices
                surjectively. It can be one of ``["expm", "cayley"]`` or a custom
                callable. Default: ``"expm"``
        """
        super().__init__(size=size, triv=triv)

    def frame(self, X):
        n, k = X.size(-2), X.size(-1)
        return F.pad(X[..., k:, :], (0, n - k, k, 0))
