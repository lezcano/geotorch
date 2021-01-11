Almost Orthogonal Matrices
==========================

.. currentmodule:: geotorch

:math:`\operatorname{AlmostOrthogonal}(n,k,\lambda)` is the manifold matrices with singular values in the interval :math:`(1-\lambda, 1+\lambda)` for a :math:`\lambda \in [0,1]`.

.. math::

    \operatorname{AlmostOrthogonal}(n,k,\lambda) = \{X \in \mathbb{R}^{n\times k}\:\mid\:\left|1-\sigma_i(X)\right| < \lambda,\  i=1, \dots, k\}

It is realized via an SVD-like factorization:

.. math::

    \begin{align*}
        \pi \colon \operatorname{St}(n,k) \times \mathbb{R}^k \times \operatorname{SO}(k)
                &\to \operatorname{AlmostOrthogonal}(n,k,\lambda) \\
            (U, \Sigma, V) &\mapsto Uf_\lambda(\Sigma) V^\intercal
    \end{align*}

where we have identified the vector :math:`\Sigma` with a diagonal matrix in :math:`\mathbb{R}^{k \times k}`.
The function :math:`f_\lambda\colon \mathbb{R} \to (1-\lambda, 1+\lambda)` takes a function :math:`f\colon \mathbb{R} \to (-1, +1)` and rescales it to be a function on :math:`(1-\lambda, 1+\lambda)` as

.. math::

    f_\lambda(x) = 1+\lambda f(x).

The function :math:`f_\lambda` is then applied element-wise to the diagonal of :math:`\Sigma`.

If :math:`\lambda = 1` is chosen, the resulting space is not a manifold, although this should not hurt optimization in practice.

.. warning::

    In the limit :math:`\lambda = 0`, the resulting manifold is exactly :ref:`sec-so`. For this reason, we discourage the use of small values of :math:`\lambda` as the algorithm in this class becomes numerically unstable for very small :math:`\lambda`. We recommend to use :class:`geotorch.SO` rather than this one in this scenario.

.. note::

    There are no restrictions in place for the image of the function :math:`f`. For a function :math:`f` with image :math:`[a,b]`, the function :math:`f_\lambda` will take values in :math:`[\lambda (1+a), \lambda (1+b)]`. As such, rescaling the function :math:`f`, one may use this class to perform optimization with singular values constrained to any prescribed interval of :math:`\mathbb{R}_{\geq 0}`.


.. autoclass:: AlmostOrthogonal

    .. automethod:: sample
    .. automethod:: in_manifold
