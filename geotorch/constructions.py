import torch
import torch.nn as nn

import geotorch.parametrize as P


class AbstractManifold(P.Parametrization):
    def __init__(self, dimensions, size):
        r"""
        Base class for all the manifolds. This class implements some basic printing
        and creates some helper attributes to handle the dimensions of the mainfolds
        such as `self.n` and `self.k` for matrix manifolds

        Args:
            dimensions (int): Number of dimensions of the manifold as a tensor.
                For example, a matrix manifold would have 2 dimensions, while a
                vector would have 1. It should be a positive number
            size (torch.size): Size of the tensor to be applied to
        """

        super().__init__()
        if dimensions != "product" and (
            not isinstance(dimensions, int) or dimensions < 1
        ):
            raise ValueError(
                "dimensions should be a non-negative integer or 'product'. Got {}".format(
                    dimensions
                )
            )
        if dimensions != "product" and len(size) < dimensions:
            raise ValueError(
                "Cannot instantiate {} on a tensor of less than {} dimensions. Got size {}".format(
                    type(self), dimensions, size
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


class Manifold(AbstractManifold):
    def __init__(self, dimensions, size, dynamic=True):
        r"""
        Base class for a manifold. The class that implements the manifold
        should implement :meth:`trivialization`.

        This class automatically implements the framework of dynamic trivializations
        described in `Trivializations for Gradient-Based Optimization on Manifolds
        - Lezcano-Casado, M. (NeurIPS 2019) <https://arxiv.org/abs/1909.09501>`_ allowing
        to perform Riemannian gradient descent by calling after each SGD step to
        :meth:`update_base`.

        For matrix manifolds, this class identifies :math:`\mathbb{R}^{k \times n}`
        with :math:`\mathbb{R}^{n \times k}` for :math:`k > n` by transposing
        the matrix before and after applying the trivialization.

        Args:
            dimensions (int): Number of dimensions of the manifold as a tensor.
                For example, a matrix manifold would have 2 dimensions, while a
                vector would have 1. It should be a positive number
            size (torch.size): Size of the tensor to be applied to
            dynamic (bool): Registers a tensor called `base` that is used to implement
                the dynamic trivialization framework
        """

        super().__init__(dimensions, size)
        self.register_buffer("base", None)
        if dynamic:
            self.base = torch.empty(*size)
            if self.transpose:
                self.base = self.base.transpose(-2, -1)

    def trivialization(self, X):  # pragma: no cover
        r"""
        Parametrizes the manifold in terms of a tangent space at the point
        `self.base`, or any given vector space in the case of the static
        trivializations

        .. note::

            This function should be surjective, otherwise not all the manifold
            will be explored

        Args:
            X (torch.Tensor): A tensor from the tangent space :math:`T_B M`
        Returns:
            tensor (torch.Tensor): A tensor in the manifold
        """
        raise NotImplementedError()

    def forward(self, X):
        if self.transpose:
            X = X.transpose(-2, -1)
        X = self.trivialization(X)
        if self.transpose:
            X = X.transpose(-2, -1)
        return X

    def update_base(self, zero=True):
        r""" Changes `self.base` to the current output of `self.original`. It allows
        for the implementation of dynamic trivializations and RGD. If `zero == True`
        it also zeros-out the original parametrized tensor

        Args:
            zero (bool,Optional): If `True`, this method will zero-out the parametrized
                tensor. Default: `True`.
        """
        if not self.is_registered():
            raise ValueError(
                "Cannot update the base before registering the Parametrization"
            )
        if self.base is not None:
            with torch.no_grad():
                X = self(self.original)
                if self.transpose:
                    X = X.transpose(-2, -1)
                self.base.data.copy_(X)
                if zero:
                    self.original_tensor().zero_()


def parametrization_from_function(f, name):
    return type(name, (P.Parametrization,), {"forward": f})


class Fibration(AbstractManifold):
    def __init__(self, dimensions, size, total_space):
        r"""
        Base class for a fibration for parametrizing a manifold :math:`M` in terms
        of another manifold :math:`E` through :math:`\pi \colon E \to M`. The class
        that implements this one should implement :meth:`embedding` and
        :meth:`fibration`

        .. note::

            This class is a bit abstract at first sight, so it might be best to understand
            it through exmaples.
            The simplest non-trivial example of this construction is the parametrization
            of the Stiefel manifold :math:`\operatorname{St}(n,k)` in terms of
            :math:`\operatorname{SO}(n)` where
            :math:`\pi \colon \operatorname{SO}(n) \to \operatorname{St}(n,k)` is the
            map that returns the first :math:`k` columns of a given orthogonal matrix
            (see :class:`geotorch.Stiefel` and :class:`geotorch.LowRank`).

        This class automatically implements the framework of dynamic trivializations
        described in `Trivializations for Gradient-Based Optimization on Manifolds
        - Lezcano-Casado, M. (NeurIPS 2019) <https://arxiv.org/abs/1909.09501>`_ allowing
        to perform Riemannian gradient descent by calling after each SGD step to
        :meth:`update_base`.

        For matrix manifolds, this class identifies :math:`\mathbb{R}^{k \times n}`
        with :math:`\mathbb{R}^{n \times k}` for :math:`k > n` by transposing
        the matrix before and after applying the trivialization

        Args:
            dimensions (int): Number of dimensions of the manifold as a tensor.
                For example, a matrix manifold would have 2 dimensions, while a
                vector would have 1. It should be a positive number
            size (torch.size): Size of the tensor to be applied to
            total_space (geotorch.AbstractManifold): The :class:`geotorch.Manifold`,
                :class:`geotorch.Fibration` or :class:`geotorch.ProductManifold`
                object that acts as a total space for the fibration. More generally,
                it could be any :class:`geotroch.AbstractManifold` that implements
                :meth:`forward`.
        """

        super().__init__(dimensions, size)
        if not isinstance(total_space, AbstractManifold):
            raise TypeError(
                "Expecting total_space to be a subclass "
                "'geotorch.AbstractManifold'. Got '{}''.".format(
                    type(total_space).__name__
                )
            )

        def f_embedding(_, X):
            if self.transpose:
                X = X.transpose(-2, -1)
            return self.embedding(X)

        Embedding = parametrization_from_function(
            f_embedding, name="Embedding" + self.__class__.__name__
        )

        total_space.chain(Embedding())
        self.chain(total_space)

    def embedding(self, X):  # pragma: no cover
        r""" Embeds a vector :math:`X \in T_B M` where

        .. math::

            B = \pi(\texttt{self.base}) = \texttt{self.projection}(\texttt{self.base})

        into the tangent space of the total space :math:`T_{\texttt{self.base}} E`
        via a local section of :math:`\mathrm{d}\pi`.

        Args:
            X (torch.Tensor): A tensor from :math:`T_B M`
        Returns:
            tensor (torch.Tensor): A tensor from :math:`T_{\texttt{self.base}} E`
        """
        raise NotImplementedError()

    def fibration(self, X):  # pragma: no cover
        r"""
        Parametrizes the manifold in terms of the total space via a mapping
        :math:`\pi`. This map should, when possible, be a submersion (surjective
        with full rank differential)

        Args:
            X (torch.Tensor): A point in the total space
        Returns:
            tensor (torch.Tensor): A tensor in the manifold
        """
        raise NotImplementedError()

    def forward(self, X):
        X = self.fibration(X)
        if self.transpose:
            X = X.transpose(-2, -1)
        return X

    # Expose the parameters from total_space
    @property
    def total_space(self):
        """
        The total space of the fibration

        Returns:
            manifold (geotorch.AbstractManifold)
        """
        return self.parametrizations.original

    @property
    def base(self):
        """
        The base of the total space of the fibration

        Returns:
            tensor (torch.Tensor)
        """
        return self.total_space.base

    def update_base(self, zero=True):
        r""" Updates the base of total space

        Args:
            zero (bool,Optional): If `True`, this method will zero-out the parametrized
                tensor. Default: `True`.
        """
        self.total_space.update_base(zero)


class ProductManifold(AbstractManifold):
    def __init__(self, manifolds):
        r"""
        Product manifold :math:`M_1 \times \dots \times M_k`. It can be indexed like a
        regular Python list, but it cannot be modified

        This class automatically implements the framework of dynamic trivializations
        described in `Trivializations for Gradient-Based Optimization on Manifolds
        - Lezcano-Casado, M. (NeurIPS 2019) <https://arxiv.org/abs/1909.09501>`_ allowing
        to perform Riemannian gradient descent by calling after each SGD step to
        :meth:`update_base`.

        .. note::

            This manifold is mostly useful in combination with :class:`~geotorch.Fibration`,
            to create quotients of product manifolds, such as :class:`~geotorch.LowRank`

        Args:
            manifolds (iterable): an iterable of manifolds
        """
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
        # Note that this product manifold always has to be chained first, as
        # the parametrization of multiple elements with one parametrization is not
        # allowed
        if not is_chained:
            for i, mani in enumerate(self):
                projection = parametrization_from_function(
                    lambda _, X, i=i: X[i], name="Projection{}".format(i)
                )()
                projection.chain(parametrization)
                mani.chain(projection)

    def forward(self, X):
        # TODO Fix it via some property injection
        # This re-evaluates X for each manifold (it shouldn't)
        return tuple(mani.evaluate() for mani in self)

    def update_base(self, zero=True):
        r""" Updates the base of all the manifolds in the product

        Args:
            zero (bool,Optional): If `True`, this method will zero-out the parametrized
                tensor. Default: `True`.
        """
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
