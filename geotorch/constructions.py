import torch.nn as nn


class ProductManifold(nn.Module):
    def __init__(self, manifolds):
        r"""
        Product manifold :math:`M_1 \times \dots \times M_k`. It can be indexed like a
        regular Python list, but it cannot be modified

        This class automatically implements the framework of dynamic trivializations
        described in `Trivializations for Gradient-Based Optimization on Manifolds
        - Lezcano-Casado, M. (NeurIPS 2019) <https://arxiv.org/abs/1909.09501>`_
        allowing to perform Riemannian gradient descent by calling after each SGD
        step the method :meth:`update_base`.

        .. note::

            This is an abstract manifold. It may be used by precomposing and
            postcomposing by an apropriate linear immersion / submersion.
            See for example :class:`~geotorch.LowRank`

        Args:
            manifolds (iterable): an iterable of manifolds
        """
        super().__init__()
        self._manifolds = nn.ModuleList(manifolds)

    def forward(self, Xs):
        return tuple(mani(X) for mani, X in zip(self, Xs))

    def initialize_(self, Xs):
        return tuple(mani.initialize_(X) for mani, X in zip(self, Xs))

    def __getitem__(self, idx):
        return self._manifolds.__getitem__(idx)

    def __len__(self):
        return self._manifolds.__len__()

    def __iter__(self):
        return self._manifolds.__iter__()

    def __dir__(self):
        return self._manifolds.__dir__()
