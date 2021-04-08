import torch.nn as nn


class ProductManifold(nn.ModuleList):
    def __init__(self, manifolds):
        r"""
        Product manifold :math:`M_1 \times \dots \times M_k`. It can be indexed like a
        regular Python list.

        .. note::

            This is an abstract manifold. It may be used by composing it on the
            left and the right by an appropriate linear immersion / submersion.
            See for example the implementation in :class:`~geotorch.LowRank`

        Args:
            manifolds (iterable): An iterable of manifolds
        """
        super().__init__(manifolds)

    def forward(self, Xs):
        return tuple(mani(X) for mani, X in zip(self, Xs))

    def right_inverse(self, Xs, check_in_manifold=True):
        return tuple(
            mani.right_inverse(X, check_in_manifold) for mani, X in zip(self, Xs)
        )
