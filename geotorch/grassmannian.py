import torch

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
        k = X.size(-1)
        size_z = X.size()[:-2] + (k, k)
        Z = X.new_zeros(*size_z)
        X = torch.cat([Z, X[..., k:, :]], dim=-2)
        return super().frame(X)
