import itertools
from collections.abc import Iterable

import torch
import torch.nn.utils.parametrization as P
import torch.nn as nn

class AbstractManifold(P.Parametrization):
    def __init__(self, dimensions, size):
        super().__init__()
        self.transpose = False
        self.dimensions = dimensions
        self.tensorial_size = tuple(size[:-dimensions])
        if self.dimensions == "product":
            self.dim = size
        elif self.dimensions == 1:
            self.n = size[-1]
            self.dim = (self.n,)
        elif self.dimensions == 2:
            self.n, self.k = size[-2], size[-1]
            self.transpose = self.n < self.k
            if self.transpose:
                self.n, self.k = self.k, self.n
            self.dim = (self.n, self.k)
        elif self.dimensions >= 3:
            self.dim = tuple(size[-(i+1)] for i in reversed(range(self.dimensions)))
        else:
            raise ValueError("Range {} not supported. Expected a positive integer or "
                             "`product`".format(self.dimensions))

    @property
    def orig_dim(self):
        self.dim[::-1] if self.transpose else self.dim

    def extra_repr(self):
        if self.dimensions == 1:
            ret = "n={}".format(self.n)
        elif self.dimensions == 2:
            ret = "n={}, k={}".format(self.n, self.k)
        else:
            ret = "size={}".format(self.size)
        if len(self.tensorial_size) != 0:
            ret += ", tensorial_size={}".format(self.tensorial_size)
        return ret


class Manifold(AbstractManifold):
    def __init__(self, dimensions, size):
        super().__init__(dimensions, size)
        self.register_buffer("base", torch.empty(*size))
        if self.transpose:
            self.base = self.base.transpose(-2, -1)

    def trivialization(self, X, B):
        r"""
        Parametrizes the manifold in terms of a tangent space
        Args:
            X (torch.nn.Tensor): A tensor, usually living in T_B M
            B (torch.nn.Tensor): Point on M at whose tangent space we are trivializing
        Returns:
            tensor (torch.nn.Tensor): A tensor on the manifold
        Note:
            This function should be surjective, otherwise not all the manifold
            will be explored
        """
        raise NotImplementedError()

    def forward(self, X):
        if self.transpose:
            X = X.transpose(-2, -1)
        X = self.trivialization(X, self.base)
        if self.transpose:
            X = X.transpose(-2, -1)
        return X


def update_base(module, tensor_name):
    if not is_parametrized(module, tensor_name):
        raise ValueError("Tensor '{}' in module '{}' is not parametrized."
                         .format(module, tensor_name))
    orig = getattr(module, tensor_name + "_orig")

    def _update_base(module):
        if isinstance(module, Manifold):
            with torch.no_grad():
                X = module.evaluate(orig)
                if module.transpose:
                    X = X.transpose(-2, -1)
                module.base.data.copy_(X)
        elif isinstance(module, Fibration):
            for p in module.parameters():
                p.zero_()

    module.apply(_update_base)
    orig.zero_()


class Fibration(AbstractManifold):
    def __init__(self, dimensions, size, total_space):
        super().__init__(dimensions, size)
        if not isinstance(total_space, AbstractManifold):
            raise TypeError("Expecting total_space to be a subclass "
                            "'mantorch.AbstractManifold'. Got '{}''."
                            .format(type(total_space).__name__))

        f_embedding = self.embedding
        if self.transpose:
            f_embedding = lambda _, X: self.embedding(X.transpose(-2, -1))

        Embedding = type("Embedding" + self.__class__.__name__,
                        (P.Parametrization,),
                        {"forward": f_embedding})

        total_space.chain(Embedding())
        self.chain(total_space)

    def embedding(self, X):
        raise NotImplementedError()

    def fibration(self, X):
        raise NotImplementedError()

    def forward(self, X):
        X = self.fibration(X)
        if self.transpose:
            X = X.transpose(-2, -1)
        return X

    # Expose the parameters from total_space
    @property
    def total_space(self):
        return self.orig_param

    @property
    def base(self):
        return self.total_space.base

class ProductManifold(AbstractManifold):
    def __init__(self, manifolds):
        super().__init__(dimensions="product",
                         size=ProductManifold._size(manifolds))
        self.manifolds = nn.ModuleList(manifolds)

    @staticmethod
    def _size(manifolds):
        for mani in manifolds:
            if not isinstance(mani, AbstractManifold):
                raise TypeError("Expecting all elements in a ProductManifold to be "
                                "mantorch.AbstractManifold. Found a {}."
                                .format(type(mani).__name__))

        return tuple(m.dim for m in manifolds)

    def forward(self, Xs):
        return [mani.evaluate(X) for mani, X in zip(self, Xs)]

    def __getitem__(self, idx):
        return self.manifolds.__getitem__(idx)

    def __len__(self):
        return self.manifolds.__len__()

    def __iter__(self):
        return self.manifolds.__iter__()

    def __dir__(self):
        return self.manifolds.__dir__()
