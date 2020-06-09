Constructions
=============

.. currentmodule:: geotorch.manifold

These are the baseclasses that conform `GeoTorch`. In particular,
every manifold is either a :class:`Manifold` or a :class:`Fibration`
or an :class:`EmbeddedManifold`. They can be used to easily implement
new manifolds in terms a function that parametrizes the particular manifold in
terms of another manifold or an unconstrained space.
Their implementation is in terms of `Parametrizations` in `PyTorch`. This directly
allows for the composition of these functions, as done in :class:`Fibration`.

From a more abstract perspective, the constructions given here can be regarded
as objects and smooth functors in the differentiable category. In particular,
these classes can be thought as an implementation of the category :math:`\operatorname{Diff}`
of differentiable mainfolds as a `cartesian monoidal category
<https://en.wikipedia.org/wiki/Cartesian_monoidal_category>`_ with `pullbacks along submersions
<https://ncatlab.org/nlab/show/submersion#pullbacks>`_.


.. autoclass:: Manifold

   .. automethod:: trivialization
   .. automethod:: update_base

.. autoclass:: EmbeddedManifold

   .. automethod:: projection

.. autoclass:: Fibration

   .. automethod:: embedding
   .. automethod:: fibration
   .. automethod:: update_base
   .. automethod:: total_space
   .. automethod:: base

.. autoclass:: ProductManifold

   .. automethod:: update_base

.. autoclass:: AbstractManifold

