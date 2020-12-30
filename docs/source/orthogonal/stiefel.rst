.. _RST Stiefel:

Stiefel Manifold
================

.. currentmodule:: geotorch

:math:`\operatorname{St}(n,k)` is the Stiefel manifold, that is, the rectangular matrices with
orthonormal columns for :math:`n \geq k`:

.. math::

    \operatorname{St}(n,k) = \{X \in \mathbb{R}^{n\times k}\:\mid\:X^\intercal X = \mathrm{I}_k\}

If :math:`n < k`, then we consider the space of matrices with orthonormal rows, that is,
:math:`X^\intercal \in \operatorname{St}(n,k)`.



.. autoclass:: Stiefel
