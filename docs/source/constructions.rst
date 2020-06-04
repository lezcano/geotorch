Constructions
=============

.. currentmodule:: geotorch.manifold

These are the baseclasses that conform `GeoTorch`. In particular,
every manifold is either a :class:`Manifold` or a :class:`Fibration`
or an :class:`EmbeddedManifold`. They can be used to easily implement
new manifolds in terms of them and a function that parametrizes the
wanted manifold.


.. autoclass:: Manifold

   .. automethod:: trivialization
   .. automethod:: update_base

.. autoclass:: EmbeddedManifold

   .. automethod:: projection

.. autoclass:: Fibration

   .. automethod:: embedding
   .. automethod:: fibration
   .. automethod:: update_base

.. autoclass:: ProductManifold

.. autoclass:: AbstractManifold

