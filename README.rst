GeoTorch
========

|Build| |Docs| |Codecov| |Codestyle Black| |License|

    A library for constrained optimization and manifold optimization for deep learning in PyTorch

Overview
--------

GeoTorch provides a simple way to perform constrained optimization and optimization on manifolds in PyTorch.
It is compatible out of the box with any optimizer, layer, and model implemented in PyTorch without having to reimplement the layers or optimizers and without any kind of boilerplate in the training code.

.. code:: python

    import torch
    import torch.nn as nn
    import geotorch

    class Net(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear = nn.Linear(64, 128)
            self.cnn = nn.Conv2d(16, 32, 3)
            # Make the linear layer into a low rank layer with rank at most 10
            geotorch.low_rank(self.linear, "weight", rank=10)
            # Also works on tensors. Makes every kernel orthogonal
            geotorch.orthogonal(self.cnn, "weight")

        def forward(self, x):
            # self.linear has rank at most 10 and every 3x3 kernel in the CNN is orthogonal

    # Nothing fancy from here on. Use the model as you'd normally do.
    model = Net()

    # Use your optimizer of choice. Any optimizer works out of the box with any parametrization
    optim = torch.optim.Adam(model.parameters(), lr=lr)

Constraints
-----------

The following constraints are implemented and may be used as in the example above:

- |geotorch.symmetric|_. Symmetric matrices
- |geotorch.skew|_. Skew-symmetric matrices
- |geotorch.sphere|_. Vectors of norm ``1``
- |geotorch.orthogonal|_. Matrices with orthogonal columns
- |geotorch.grassmannian|_. Skew-symmetric matrices
- |geotorch.almost_orthogonal|_. Matrices with singular values in  the interval ``[1-λ, 1+λ]``
- |geotorch.low_rank|_. Matrices of rank at most ``r``
- |geotorch.fixed_rank|_. Matrices of rank ``r``
- |geotorch.positive_definite|_. Positive definite matrices
- |geotorch.positive_semidefinite|_. Positive semidefinite matrices
- |geotorch.positive_semidefinite_low_rank|_. Positive semidefinite matrices of rank at most ``r``
- |geotorch.positive_semidefinite_fixed_rank|_. Positive semidefinite matrices of rank ``r``

.. |geotorch.symmetric| replace:: ``geotorch.symmetric``
.. _geotorch.symmetric: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.symmetric
.. |geotorch.skew| replace:: ``geotorch.skew``
.. _geotorch.skew: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.skew
.. |geotorch.sphere| replace:: ``geotorch.sphere``
.. _geotorch.sphere: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.sphere
.. |geotorch.orthogonal| replace:: ``geotorch.orthogonal``
.. _geotorch.orthogonal: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.orthogonal
.. |geotorch.grassmannian| replace:: ``geotorch.grassmannian``
.. _geotorch.grassmannian: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.grassmannian
.. |geotorch.almost_orthogonal| replace:: ``geotorch.almost_orthogonal(λ)``
.. _geotorch.almost_orthogonal: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.almost_orthogonal
.. |geotorch.low_rank| replace:: ``geotorch.low_rank(r)``
.. _geotorch.low_rank: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.low_rank
.. |geotorch.fixed_rank| replace:: ``geotorch.fixed_rank(r)``
.. _geotorch.fixed_rank: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.fixed_rank
.. |geotorch.positive_definite| replace:: ``geotorch.positive_definite``
.. _geotorch.positive_definite: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.positive_definite
.. |geotorch.positive_semidefinite| replace:: ``geotorch.positive_semidefinite``
.. _geotorch.positive_semidefinite: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.positive_semidefinite
.. |geotorch.positive_semidefinite_low_rank| replace:: ``geotorch.positive_semidefinite_low_rank(r)``
.. _geotorch.positive_semidefinite_low_rank: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.positive_semidefinite_low_rank
.. |geotorch.positive_semidefinite_fixed_rank| replace:: ``geotorch.positive_semidefinite_fixed_rank(r)``
.. _geotorch.positive_semidefinite_fixed_rank: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.positive_semidefinite_fixed_rank

Each of these constraints have some extra parameters which can be used to tailor the
behavior of each constraint to the problem in hand. For more on this, see the constructions
section in the documentation.

These constraint functions are a convenient umbrella for the families of spaces listed below.

Supported Spaces
----------------

Each constraint in GeoTorch is implemented as a manifold. These give the user more flexibility
on the options that they choose for each parametrization. All these support Riemannian Gradient
Descent by default (more on this `here`_), but they also support optimization via any other PyTorch
optimizer.

GeoTorch currently supports the following spaces:

- |reals|_: Rⁿ. Unrestricted optimization
- |sym|_: Vector space of symmetric matrices
- |skew|_: Vector space of skew-symmetric matrices
- |sphere|_: Sphere in Rⁿ. It is Sⁿ⁻¹ = { x ∈ Rⁿ | ||x|| = 1 }
- |so|_: Manifold of n×n orthogonal matrices
- |st|_: Manifold of n×k matrices with orthonormal columns
- |almost|_: Manifold of n×k matrices with singular values in the interval (1-λ, 1+λ)
- |grass|_: Manifold of k-dimensional subspaces in Rⁿ
- |low|_: Variety of n×k matrices of rank r or less
- |fixed|_: Manifold of n×k matrices of rank r
- |glp|_: Manifold of invertible n×n matrices with positive determinant
- |psd|_: Cone of n×n symmetric positive definite matrices
- |pssd|_: Cone of n×n symmetric positive semi-definite matrices
- |pssdlow|_: Variety of n×n symmetric positive semi-definite matrices of rank r or less
- |pssdfixed|_: Manifold of n×n symmetric positive semi-definite matrices of rank r

.. |reals| replace:: ``Rn(n)``
.. _reals: https://geotorch.readthedocs.io/en/latest/reals.html
.. |sym| replace:: ``Sym(n)``
.. _sym: https://geotorch.readthedocs.io/en/latest/symmetric.html
.. |skew| replace:: ``Skew(n)``
.. _skew: https://geotorch.readthedocs.io/en/latest/skew.html
.. |sphere| replace:: ``Sphere(n)``
.. _sphere: https://geotorch.readthedocs.io/en/latest/sphere.html
.. |so| replace:: ``SO(n)``
.. _so: https://geotorch.readthedocs.io/en/latest/so.html
.. |st| replace:: ``St(n,k)``
.. _st: https://geotorch.readthedocs.io/en/latest/stiefel.html
.. |almost| replace:: ``AlmostOrthogonal(n,k,λ)``
.. _almost: https://geotorch.readthedocs.io/en/latest/almostorthogonal.html
.. |grass| replace:: ``Gr(n,k)``
.. _grass: https://geotorch.readthedocs.io/en/latest/grassmannian.html
.. |low| replace:: ``LowRank(n,k,r)``
.. _low: https://geotorch.readthedocs.io/en/latest/lowrank.html
.. |fixed| replace:: ``FixedRank(n,k,r)``
.. _fixed: https://geotorch.readthedocs.io/en/latest/fixedrank.html
.. |glp| replace:: ``GLp(n)``
.. _glp: https://geotorch.readthedocs.io/en/latest/glp.html
.. |psd| replace:: ``PSD(n)``
.. _psd: https://geotorch.readthedocs.io/en/latest/psd.html
.. |pssd| replace:: ``PSSD(n)``
.. _pssd: https://geotorch.readthedocs.io/en/latest/pssd.html
.. |pssdlow| replace:: ``PSSDLowRank(n,r)``
.. _pssdlow: https://geotorch.readthedocs.io/en/latest/pssdlowrank.html
.. |pssdfixed| replace:: ``PSSDFixedRank(n,r)``
.. _pssdfixed: https://geotorch.readthedocs.io/en/latest/pssdfixedrank.html


Every space of dimension ``(n, k)`` can be applied to tensors of shape ``(*, n, k)``, so we also get efficient parallel implementations of product spaces such as

- ``ObliqueManifold(n,k)``: Matrix with unit length columns, Sⁿ⁻¹ × ...ᵏ⁾ × Sⁿ⁻¹

GeoTorch also provides the following constructions which help the user to implement other spaces:

- |manif|_: Manifold that supports Riemannian Gradient Descent and trivializations
- |fib|_: Fibred space π : E → M, constructed from a ``Manifold`` E, a submersion π and local sections of dπ
- |prod|_: M₁ × ... × Mₖ


.. |manif| replace:: ``Manifold``
.. _manif: https://geotorch.readthedocs.io/en/latest/constructions.html#geotorch.constructions.Manifold
.. |fib| replace:: ``Fibration``
.. _fib: https://geotorch.readthedocs.io/en/latest/constructions.html#geotorch.constructions.Fibration
.. |prod| replace:: ``ProductManifold``
.. _prod: https://geotorch.readthedocs.io/en/latest/constructions.html#geotorch.constructions.ProductManifold

Sharing Weights, Parametrizations, and Normalizing Flows
--------------------------------------------------------

If one wants to use a parametrized tensor in different places in their model, or uses one parametrized layer many times, for example in an RNN, it is recommended to wrap the forward pass as follows to avoid each parametrization to be computed many times:

.. code:: python

    with geotorch.parametrize.cached():
        logits = model(input_)

Of course, this ``with`` statement may be used simply inside the forward function where the parametrized layer is used several times.

These ideas fall in the context of parametrized optimization, where one wraps a tensor ``X`` with a function ``f``, and rather than using ``X``, uses ``f(X)``. Particular examples of this idea are pruning, weight normalization, and spectral normalization among others. This repository implements a framework to approach this kind of problems. The framework is currently `PR #33344`_ in PyTorch. All the functionality of this PR is located in `geotorch/parametrize.py`_.

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

