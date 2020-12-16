import torch.nn as nn


class ProductManifold(nn.ModuleList):
    def __init__(self, manifolds):
        r"""
        Product manifold :math:`M_1 \times \dots \times M_k`. It can be indexed like a
        regular Python list.

        .. note::

            This is an abstract manifold. It may be used by precomposing and
            postcomposing by an apropriate linear immersion / submersion.
            See for example :class:`~geotorch.LowRank`

        Args:
            manifolds (iterable): an iterable of manifolds
        """
        super().__init__(manifolds)

    def forward(self, Xs):
        return tuple(mani(X) for mani, X in zip(self, Xs))
