Stiefel Manifold: :math:`\operatorname{St}(n,k)`
======================================================

.. currentmodule:: geotorch.stiefel

:math:`\operatorname{St}(n,k)` is the Stiefel manifold, that is, the rectangular matrices with
orthonormal columns for :math:`n \geq k`.

.. math::

    \operatorname{St}(n,k) = \{X \in \mathbb{R}^{n\times k}\:\mid\:X^\intercal X = \mathrm{I}\}

If :math:`n < k`, then we consider the space of matrices with orthonormal rows, that is,
:math:`X^\intercal \in \operatorname{St}(n,k)`.



.. autoclass:: Stiefel

   .. automethod:: uniform_init_

.. autoclass:: StiefelTall

   .. automethod:: uniform_init_
