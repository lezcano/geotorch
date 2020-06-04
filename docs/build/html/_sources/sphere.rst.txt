Sphere: :math:`\mathbb{S}^n`
=============================

.. currentmodule:: geotorch.sphere

:math:`\operatorname{Sphere}(n, r)` is the sphere in :math:`\mathbb{R}^n`
with radius :math:`r`:

.. math::

    \operatorname{Sphere}(n,r) = \{x \in \mathbb{R}^n\:\mid\:\lVert x \rVert = r\}

.. warning::

    In mathematics, :math:`\mathbb{S}^n` represents the :math:`n`-dimensional sphere.
    With this notation, :math:`\operatorname{Sphere}(n, 1.) = \mathbb{S}^{n-1}`.

.. autoclass:: Sphere

   .. automethod:: uniform_init_

.. autoclass:: SphereEmbedded

   .. automethod:: uniform_init_
