import torch

from .stiefel import Stiefel, StiefelTall, non_singular_
from .utils import base, transpose


class Grassmannian(Stiefel):
    def __init__(self, size, triv="expm"):
        r"""
        Grassmannian manifold as a projection from the orthogonal
        matrices :math:`\operatorname{St}(n,k)`.
        The metric considered is the canonical.

        .. note::

            This class is equivalent to :class:`GrassmannianTall`, but it is faster
            for the case when :math:`n` is of a similar size of :math:`k`. For example,
            :math:`n \leq 4k`.

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


class GrassmannianTall(StiefelTall):
    def __init__(self, size, triv="expm"):
        r"""
        Grassmannian manifold parametrized from is tangent space using the orthogonal
        projection from the ambient space :math:`\mathbb{R}^{n \times k}`.
        The metric considered is the canonical.

        .. note::

            This class is equivalent to :class:`Grassmannian`, but it is faster for the case
            when :math:`n` is of a much larger than :math:`k`. For example, :math:`n > 4k`.

        Args:
            size (torch.size): Size of the tensor to be parametrized
            triv (str or callable): Optional.
                A map that maps skew-symmetric matrices onto the orthogonal matrices
                surjectively. It can be one of ``["expm", "cayley"]`` or a custom
                callable. Default: ``"expm"``
        """
        super().__init__(size, triv)

    @transpose
    @base
    def forward(self, X):
        # TODO This could surely be cleaner
        if torch.is_grad_enabled():
            non_singular_(X)
        # We project onto \hlie^\perp so that X_\hlie = B.t() @ X = 0
        B = self.base
        Bt = B.transpose(-2, -1)
        X = X - B @ (Bt @ X)
        # This line is the difference between the Grassmannian and the Stiefel manifold
        # In the Grassmannian A = 0, but we may still propagate the gradient along it
        A = (Bt @ X).tril(-1)

        return self._expm_aux(X, A)
