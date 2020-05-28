import torch
import torch.nn as nn

import geotorch.parametrize as P


class AbstractManifold(P.Parametrization):
    def __init__(self, dimensions, size):
        super().__init__()
        if dimensions != "product" and (
            not isinstance(dimensions, int) or dimensions < 0
        ):
            raise ValueError(
                "dimensions should be a non-negative integer or 'product'. Got {}".format(
                    dimensions
                )
            )

        self.transpose = False
        self.dimensions = dimensions
        if self.dimensions != "product":
            self.tensorial_size = tuple(size[:-dimensions])
            self.dim = tuple(size[-dimensions:])
            if self.dimensions == 1:
                self.n = self.dim[0]
            elif self.dimensions == 2:
                self.transpose = self.dim[0] < self.dim[1]
                if self.transpose:
                    self.dim = tuple(reversed(self.dim))
                self.n, self.k = self.dim

    @property
    def orig_dim(self):
        self.dim[::-1] if self.transpose else self.dim

    def extra_repr(self):
        if self.dimensions == "product":
            return ""
        if self.dimensions == 1:
            ret = "n={}".format(self.n)
        elif self.dimensions == 2:
            ret = "n={}, k={}".format(self.n, self.k)
        else:
            ret = "dim={}".format(self.dim)
        if len(self.tensorial_size) != 0:
            ret += ", tensorial_size={}".format(self.tensorial_size)
        return ret


class EmbeddedManifold(AbstractManifold):
    def projection(self, X):  # pragma: no cover
        r"""
        Parametrizes the manifold in terms of a projection from the ambient space
        Args:
            X (torch.nn.Tensor): A tensor in the ambient space
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
        X = self.projection(X)
        if self.transpose:
            X = X.transpose(-2, -1)
        return X


class Manifold(AbstractManifold):
    def __init__(self, dimensions, size):
        super().__init__(dimensions, size)
        self.register_buffer("base", torch.empty(*size))
        if self.transpose:
            self.base = self.base.transpose(-2, -1)

    def trivialization(self, X, B):  # pragma: no cover
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

    def update_base(self, zero=True):
        if not self.is_registered():
            raise ValueError(
                "Cannot update the base before registering the Parametrization"
            )
        with torch.no_grad():
            X = self.evaluate()
            if self.transpose:
                X = X.transpose(-2, -1)
            self.base.data.copy_(X)
            if zero:
                self.original_tensor().zero_()


def parametrization_from_function(f, name):
    return type(name, (P.Parametrization,), {"forward": f})


class Fibration(AbstractManifold):
    def __init__(self, dimensions, size, total_space):
        super().__init__(dimensions, size)
        if not isinstance(total_space, AbstractManifold):
            raise TypeError(
                "Expecting total_space to be a subclass "
                "'geotorch.AbstractManifold'. Got '{}''.".format(
                    type(total_space).__name__
                )
            )

        f_embedding = self.embedding
        if self.transpose:

            def f_embedding(_, X):
                return self.embedding(X.transpose(-2, -1))

        Embedding = parametrization_from_function(
            f_embedding, name="Embedding" + self.__class__.__name__
        )

        total_space.chain(Embedding())
        self.chain(total_space)

    def embedding(self, X):  # pragma: no cover
        raise NotImplementedError()

    def fibration(self, X):  # pragma: no cover
        raise NotImplementedError()

    def forward(self, X):
        X = self.fibration(X)
        if self.transpose:
            X = X.transpose(-2, -1)
        return X

    # Expose the parameters from total_space
    @property
    def total_space(self):
        return self.parametrizations.original

    @property
    def base(self):
        return self.total_space.base

    def update_base(self, zero=True):
        self.total_space.update_base(zero)


class ProductManifold(AbstractManifold):
    def __init__(self, manifolds):
        super().__init__(dimensions="product", size=ProductManifold._size(manifolds))
        self.manifolds = nn.ModuleList(manifolds)

    @staticmethod
    def _size(manifolds):
        for mani in manifolds:
            if not isinstance(mani, AbstractManifold):
                raise TypeError(
                    "Expecting all elements in a ProductManifold to be "
                    "geotorch.AbstractManifold. Found a {}.".format(type(mani).__name__)
                )

        return tuple(m.dim for m in manifolds)

    def chain(self, parametrization):
        is_chained = self.is_chained()
        super().chain(parametrization)
        # We do this just the first time
        if not is_chained:
            for i, mani in enumerate(self):
                projection = parametrization_from_function(
                    lambda _, X, i=i: X[i], name="Projection{}".format(i)
                )()
                projection.chain(parametrization)
                mani.chain(projection)

    def forward(self, X):
        # This is not quite right, but I don't think we can do better with the current API
        # In practice it's really not a problem
        return tuple(mani.evaluate() for mani in self)

    def update_base(self, zero=True):
        if not self.is_registered():
            raise ValueError(
                "Cannot update the base before registering the Parametrization"
            )
        for mani in self:
            mani.update_base(False)
        if zero:
            with torch.no_grad():
                self.original_tensor().zero_()

    def __getitem__(self, idx):
        return self.manifolds.__getitem__(idx)

    def __len__(self):
        return self.manifolds.__len__()

    def __iter__(self):
        return self.manifolds.__iter__()

    def __dir__(self):
        return self.manifolds.__dir__()
