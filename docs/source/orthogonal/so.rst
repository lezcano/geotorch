Special Orthogonal Group
========================

.. currentmodule:: geotorch

:math:`\operatorname{SO}(n)` is the special orthogonal group, that is, the square matrices with
orthonormal columns and positive determinant:

.. math::

    \operatorname{SO}(n) = \{X \in \mathbb{R}^{n\times n}\:\mid\:X^\intercal X = \mathrm{I},\,\det(X) = 1\}


.. autoclass:: SO

   .. automethod:: uniform_init_
   .. automethod:: torus_init_
