GeoTorch
========

|Build| |Docs| |Codecov| |Codestyle Black| |License|

    A library for constrained optimization and manifold optimization for deep learning in PyTorch

Overview
--------

GeoTorch provides a simple way to perform constrained optimization and optimization on manifolds in PyTorch.
It is compatible out of the box with any optimizer, layer, and model implemented in PyTorch without any boilerplate in the training code. Just state the constraints when you construct the model and you are ready to go!

.. code:: python

    import torch
    import torch.nn as nn
    import geotorch

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            # One line suffices: Make a linear layer with orthonormal columns
            self.linear = nn.Linear(64, 128)
            geotorch.orthogonal(self.linear, "weight")

            # Works with tensors: Make a CNN with kernels of rank 1
            self.cnn = nn.Conv2d(16, 32, 3)
            geotorch.low_rank(self.cnn, "weight", rank=1)

            # Weights are initialized to a random value when you put the constraints, but
            # you may re-initialize them to a different value by assigning to them
            self.linear.weight = torch.eye(128, 64)
            # And that's all you need to do. The rest is regular PyTorch code

        def forward(self, x):
            # self.linear is orthogonal and every 3x3 kernel in self.cnn is of rank 1

    # Use the model as you would normally do. Everything just works
    model = Model().cuda()

    # Use your optimizer of choice. Any optimizer works out of the box with any parametrization
    optim = torch.optim.Adam(model.parameters(), lr=lr)

Constraints
-----------

The following constraints are implemented and may be used as in the example above:

- |symmetric|_. Symmetric matrices
- |skew_constr|_. Skew-symmetric matrices
- |sphere_constr|_. Vectors of norm ``1``
- |orthogonal|_. Matrices with orthogonal columns
- |grassmannian|_. Skew-symmetric matrices
- |almost_orthogonal|_. Matrices with singular values in  the interval ``[1-λ, 1+λ]``
- |invertible|_. Invertible matrices with positive determinant
- |low_rank|_. Matrices of rank at most ``r``
- |fixed_rank|_. Matrices of rank ``r``
- |positive_definite|_. Positive definite matrices
- |positive_semidefinite|_. Positive semidefinite matrices
- |positive_semidefinite_low_rank|_. Positive semidefinite matrices of rank at most ``r``
- |positive_semidefinite_fixed_rank|_. Positive semidefinite matrices of rank ``r``

.. |symmetric| replace:: ``geotorch.symmetric``
.. _symmetric: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.symmetric
.. |skew_constr| replace:: ``geotorch.skew``
.. _skew_constr: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.skew
.. |sphere_constr| replace:: ``geotorch.sphere``
.. _sphere_constr: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.sphere
.. |orthogonal| replace:: ``geotorch.orthogonal``
.. _orthogonal: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.orthogonal
.. |grassmannian| replace:: ``geotorch.grassmannian``
.. _grassmannian: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.grassmannian
.. |almost_orthogonal| replace:: ``geotorch.almost_orthogonal(λ)``
.. _almost_orthogonal: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.almost_orthogonal
.. |invertible| replace:: ``geotorch.invertible``
.. _invertible: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.invertible
.. |low_rank| replace:: ``geotorch.low_rank(r)``
.. _low_rank: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.low_rank
.. |fixed_rank| replace:: ``geotorch.fixed_rank(r)``
.. _fixed_rank: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.fixed_rank
.. |positive_definite| replace:: ``geotorch.positive_definite``
.. _positive_definite: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.positive_definite
.. |positive_semidefinite| replace:: ``geotorch.positive_semidefinite``
.. _positive_semidefinite: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.positive_semidefinite
.. |positive_semidefinite_low_rank| replace:: ``geotorch.positive_semidefinite_low_rank(r)``
.. _positive_semidefinite_low_rank: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.positive_semidefinite_low_rank
.. |positive_semidefinite_fixed_rank| replace:: ``geotorch.positive_semidefinite_fixed_rank(r)``
.. _positive_semidefinite_fixed_rank: https://geotorch.readthedocs.io/en/latest/constraints.html#geotorch.positive_semidefinite_fixed_rank

Each of these constraints have some extra parameters which can be used to tailor the
behavior of each constraint to the problem in hand. For more on this, see the documentation.

These constraints are a fronted for the families of spaces listed below.

Supported Spaces
----------------

Each constraint in GeoTorch is implemented as a manifold. These give the user more flexibility
on the options that they choose for each parametrization. All these support Riemannian Gradient
Descent by default (more on this `here`_), but they also support optimization via any other PyTorch
optimizer.

GeoTorch currently supports the following spaces:

- |reals|_: ``Rⁿ``. Unrestricted optimization
- |sym|_: Vector space of symmetric matrices
- |skew|_: Vector space of skew-symmetric matrices
- |sphere|_: Sphere in ``Rⁿ``. ``{ x ∈ Rⁿ | ||x|| = 1 } ⊂ Rⁿ``
- |so|_: Manifold of ``n×n`` orthogonal matrices
- |st|_: Manifold of ``n×k`` matrices with orthonormal columns
- |almost|_: Manifold of ``n×k`` matrices with singular values in the interval ``[1-λ, 1+λ]``
- |grass|_: Manifold of ``k``-dimensional subspaces in ``Rⁿ``
- |glp|_: Manifold of invertible ``n×n`` matrices with positive determinant
- |low|_: Variety of ``n×k`` matrices of rank ``r`` or less
- |fixed|_: Manifold of ``n×k`` matrices of rank ``r``
- |psd|_: Cone of ``n×n`` symmetric positive definite matrices
- |pssd|_: Cone of ``n×n`` symmetric positive semi-definite matrices
- |pssdlow|_: Variety of ``n×n`` symmetric positive semi-definite matrices of rank ``r`` or less
- |pssdfixed|_: Manifold of ``n×n`` symmetric positive semi-definite matrices of rank ``r``
- |product|_: Product of manifolds ``M₁ × ... × Mₖ``

Every space of dimension ``(n, k)`` can be applied to tensors of shape ``(*, n, k)``, so we also get efficient parallel implementations of product spaces such as

- ``ObliqueManifold(n,k)``: Matrix with unit length columns, ``Sⁿ⁻¹ × ...ᵏ⁾ × Sⁿ⁻¹``

Using GeoTorch in your Code
---------------------------

The files in `examples/copying_problem.py`_ and `examples/sequential_mnist.py`_ serve as tutorials to see how to handle the initialization and usage of GeoTorch in some real code. They also show how to implement Riemannian Gradient Descent and some other tricks. For an introduction to how the library is actually implemented, see the Jupyter Notebook `examples/parametrisations.ipynb`_.

You may try GeoTorch installing it with

.. code:: bash

    pip install git+https://github.com/Lezcano/geotorch/

GeoTorch is tested in Linux, Mac, and Windows environments for Python >= 3.6.

Sharing Weights, Parametrizations, and Normalizing Flows
--------------------------------------------------------

If one wants to use a parametrized tensor in different places in their model, or uses one parametrized layer many times, for example in an RNN, it is recommended to wrap the forward pass as follows to avoid each parametrization to be computed many times:

.. code:: python

    with geotorch.parametrize.cached():
        logits = model(input_)

Of course, this ``with`` statement may be used simply inside the forward function where the parametrized layer is used several times.

These ideas fall in the context of parametrized optimization, where one wraps a tensor ``X`` with a function ``f``, and rather than using ``X``, uses ``f(X)``. Particular examples of this idea are pruning, weight normalization, and spectral normalization among others. This repository implements a framework to approach this kind of problems. The framework is currently `PR #33344`_ in PyTorch. All the functionality of this PR is located in `geotorch/parametrize.py`_.

As every space in GeoTorch is, at its core, a map from a flat space into a manifold, the tools implemented here also serve as a building block in normalizing flows. Using a factorized space such as |low|_ it is direct to compute the determinant of the transformation it defines, as we have direct access to the singular values of the layer.

.. |reals| replace:: ``Rn(n)``
.. _reals: https://geotorch.readthedocs.io/en/latest/vector_spaces/reals.html
.. |sym| replace:: ``Sym(n)``
.. _sym: https://geotorch.readthedocs.io/en/latest/vector_spaces/symmetric.html
.. |skew| replace:: ``Skew(n)``
.. _skew: https://geotorch.readthedocs.io/en/latest/vector_spaces/skew.html
.. |sphere| replace:: ``Sphere(n)``
.. _sphere: https://geotorch.readthedocs.io/en/latest/orthogonal/sphere.html
.. |so| replace:: ``SO(n)``
.. _so: https://geotorch.readthedocs.io/en/latest/orthogonal/so.html
.. |st| replace:: ``St(n,k)``
.. _st: https://geotorch.readthedocs.io/en/latest/orthogonal/stiefel.html
.. |almost| replace:: ``AlmostOrthogonal(n,k,λ)``
.. _almost: https://geotorch.readthedocs.io/en/latest/orthogonal/almostorthogonal.html
.. |grass| replace:: ``Gr(n,k)``
.. _grass: https://geotorch.readthedocs.io/en/latest/orthogonal/grassmannian.html
.. |glp| replace:: ``GLp(n)``
.. _glp: https://geotorch.readthedocs.io/en/latest/invertibility/glp.html
.. |low| replace:: ``LowRank(n,k,r)``
.. _low: https://geotorch.readthedocs.io/en/latest/lowrank/lowrank.html
.. |fixed| replace:: ``FixedRank(n,k,r)``
.. _fixed: https://geotorch.readthedocs.io/en/latest/lowrank/fixedrank.html
.. |psd| replace:: ``PSD(n)``
.. _psd: https://geotorch.readthedocs.io/en/latest/psd/psd.html
.. |pssd| replace:: ``PSSD(n)``
.. _pssd: https://geotorch.readthedocs.io/en/latest/psd/pssd.html
.. |pssdlow| replace:: ``PSSDLowRank(n,r)``
.. _pssdlow: https://geotorch.readthedocs.io/en/latest/psd/pssdlowrank.html
.. |pssdfixed| replace:: ``PSSDFixedRank(n,r)``
.. _pssdfixed: https://geotorch.readthedocs.io/en/latest/psd/pssdfixedrank.html
.. |product| replace:: ``ProductManifold(M₁, ..., Mₖ)``
.. _product: https://geotorch.readthedocs.io/en/latest/product.html


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
.. _examples/sequential_mnist.py: https://github.com/Lezcano/geotorch/blob/master/examples/sequential_mnist.py
.. _examples/copying_problem.py: https://github.com/Lezcano/geotorch/blob/master/examples/copying_problem.py
.. _examples/parametrisations.ipynb: https://github.com/Lezcano/geotorch/blob/master/examples/parametrisations.ipynb

