Special Orthogonal Group: :math:`\operatorname{SO}(n)`
======================================================

.. currentmodule:: geotorch.so

:math:`\operatorname{SO}(n)` is the special orthogonal group, that is, the square matrices with
orthonormal columns and positive determinant

.. math::

    \operatorname{SO}(n) = \{B \in \mathbb{R}^{n\times n}\:\mid\:B^\intercal B = \mathrm{I},\,\det(B) = 1\}


.. autoclass:: SO

   .. automethod:: torus_init_
   .. automethod:: uniform_init_
