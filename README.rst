GeoTorch
========

|Build| |Docs| |Codecov| |Codestyle Black| |License|

    A library for constrained optimization and manifold optimization for deep learning in PyTorch

Overview
--------

GeoTorch provides a simple way to perform constrained optimization and optimization on manifolds in PyTorch.
It is compatible out of the box with any optimizer, layer, and model implemented in PyTorch without having to reimplement
the layers or optimizers and without any kind of boilerplate.

.. code:: python

    import torch
    import torch.nn as nn
    import geotorch

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear = nn.Linear(64, 128)
            self.cnn = nn.Conv2d(16, 32, 3)
            # Make the linear layer into a low rank layer with rank at most 10
            geotorch.lowrank(self.linear, "weight", rank=10)
            # Also works on tensors. Makes every kernel orthogonal
            geotorch.orthogonal(self.cnn, "weight")

        def forward(self, x):
            # self.linear is has rank at most 10 and every 3x3 kernel in the CNN is orthogonal

    # Nothing fancy from here on. Use the model as you'd normally do.
    model = Model()

    # Use your optimizer of choice. Any optimizer works out of the box with any parametrization
    optim = torch.optim.Adam(model.parameters(), lr=lr)

Constraints
-----------

The following constraints are implemented and may be used as in the example above:

- ``geotorch.skew``. Skew-symmetric matrices
- ``geotorch.symmetric``. Symmetric matrices
- ``geotorch.sphere``. Vectors of norm ``1``
- ``geotorch.orthogonal``. Matrices with orthogonal columns
- ``geotorch.grassmannian``. Skew-symmetric matrices
- ``geotorch.lowrank(r)``. Matrices of rank at most ``r``

Each of these constraints have some extra parameters which can be used to tailor the
behavior of each constraint to the problem in hand. For more on this, see the constructions
section in the documentation.

Supported Spaces
----------------

The constraints in GeoTorch are implemented as manifolds. These give the user more flexibility
on the options that they choose for each parametrization. All these support Riemannian Gradient
Descent by default (more on this `here`_), but they also support optimization via any other optimizer.

GeoTorch currently supports the following spaces:

- ``Rn(n)``: Rⁿ. Unrestricted optimization
- ``Sym(n)``: Vector space of symmetric matrices
- ``Skew(n)``: Vector space of skew-symmetric matrices
- ``Sphere(n)``: Sphere in Rⁿ. It is Sⁿ⁻¹ = { x ∈ Rⁿ | ||x|| = 1 }
- ``SO(n)``: Manifold of n×n orthogonal matrices
- ``Stiefel(n,k)``: Manifold of n×k matrices with orthonormal columns
- ``Grassmannian(n,k)``: Manifold of k-dimensional subspaces in Rⁿ
- ``LowRank(n,k,r)``: Variety of n×k matrices of rank r or less

And the following spaces are planed to be implemented in the near future:

- ``AlmostOrthogonal(n,k,t)``: Manifold of n×k matrices with singular values in the interval (1-t, 1+t)
- ``FixedRank(n,k,r)``: Manifold of n×k matrices of rank r
- ``PD(n)``: Cone of n×n symmetric positive definite matrices
- ``PSD(n)``: Cone of n×n symmetric positive semi-definite matrices
- ``PSDLowRank(n,k)``: Variety of n×n symmetric positive semi-definite matrices of rank k or less
- ``PSDFixedRank(n,k)``: Manifold of n×n symmetric positive semi-definite matrices of rank k
- ``SymF(n, f)``: Symmetric positive definite matrices with eigenvalues in the image of a map ``f``. If the map ``f`` is an embedding, this is a manifold

Every manifold of dimension ``(n, k)`` can be applied to tensors of shape ``(*, n, k)``, so we also get efficient parallel implementations of product manifolds such as

- ``ObliqueManifold(n,k)``: Matrix with unit length columns, Sⁿ⁻¹ × ...ᵏ⁾ × Sⁿ⁻¹

It also implements the following constructions:

- ``Manifold``: Manifold that supports Riemannian Gradient Descent and trivializations
- ``Fibration``: Fibred space π : E → M, constructed from a ``Manifold`` E, a submersion π and local sections of dπ
- ``ProductManifold``: M₁ × ... × Mₖ

Sharing Weights, Parametrizations, and Normalizing Flows
--------------------------------------------------------

If one wants to use a parametrized tensor in different places in their model, or uses one parametrized layer many times, for example in an RNN, it is recommended to wrap the forward pass as follows to avoid each parametrization to be computed many times:

.. code:: python

    with geotorch.parametrize.cached():
        logits = model(input_)

Of course, this ``with`` statement may be used simply inside the forward function where the parametrized layer is used several times.

These ideas fall in the context of parametrized optimization, where one wraps a tensor ``X`` with a function ``f``, and rather than using ``X``, we use ``f(X)``. Particular examples of this idea are pruning, weight normalization, and spectral normalization among others. This repository implements a framework to approach this kind of problems. The framework is currently `PR #33344`_ in PyTorch. All the functionality of this PR is located in `geotorch/parametrize.py`_.

As every space in GeoTorch is, at its core, a map from a flat space into a manifold, the tools implemented here also serve as a building block in normalizing flows. Using a factorized space such as LowRank it is direct to compute the determinant of the transformation it defines, as we have direct access to the signular values of the layer.

Try GeoTorch!
-------------

If you have installed PyTorch v1.5 at least, you may try GeoTorch installing it via

.. code:: bash

    pip install git+https://github.com/Lezcano/geotorch/

GeoTorch is tested in Linux, Mac, and Windows environments for Python >= 3.6.

Bibliography
------------

Please cite the following work if you found GeoTorch useful. This paper exposes a simplified mathematical explanation of part of the inner-workings of GeoTorch.

.. code:: bibtex

    @inproceedings{lezcano2019trivializations,
        title = {Trivializations for gradient-based optimization on manifolds},
        author = {Lezcano-Casado, Mario},
        booktitle={Advances in Neural Information Processing Systems, NeurIPS},
        pages = {9154--9164},
        year = {2019},
    }


.. |Build| image:: https://github.com/lezcano/geotorch/workflows/Build/badge.svg
   :target: https://github.com/lezcano/geotorch/workflows/Build/badge.svg
   :alt: Build
.. |Docs| image:: https://readthedocs.org/projects/geotorch/badge/?version=latest
   :target: https://geotorch.readthedocs.io/en/latest/?badge=latest
.. |Codecov| image:: https://codecov.io/gh/Lezcano/geotorch/branch/master/graph/badge.svg?token=1AKM2EQ7RT
   :target: https://codecov.io/gh/Lezcano/geotorch/branch/master/graph/badge.svg?token=1AKM2EQ7RT
   :alt: Code coverage
.. |Codestyle Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black
   :alt: Codestyle Black
.. |License| image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://github.com/Lezcano/geotorch/blob/master/LICENSE
   :alt: License

.. _here: https://github.com/Lezcano/geotorch/blob/master/examples/copying_problem.py#L16
.. _PR #33344: https://github.com/pytorch/pytorch/pull/33344
.. _geotorch/parametrize.py: https://github.com/Lezcano/geotorch/blob/master/geotorch/parametrize.py

