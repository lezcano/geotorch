Special Orthogonal Group
========================

.. currentmodule:: geotorch

:math:`\operatorname{SO}(n)` is the special orthogonal group, that is, the square matrices with
orthonormal columns and positive determinant:

.. math::

    \operatorname{SO}(n) = \{X \in \mathbb{R}^{n\times n}\:\mid\:X^\intercal X = \mathrm{I}_n,\,\det(X) = 1\}


.. autoclass:: SO
.. autofunction:: geotorch.so.uniform_init_
.. autofunction:: geotorch.so.torus_init_
