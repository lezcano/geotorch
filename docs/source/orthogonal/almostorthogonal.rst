Almost Orthogonal Matrices
==========================

.. currentmodule:: geotorch

:math:`\operatorname{AlmostOrthogonal}(n,k,\lambda)` is the manifold matrices with singular values in the interval :math:`[1-\lambda, 1+\lambda]` for a :math:`\lambda \in [0,1]`.

.. math::

    \operatorname{AlmostOrthogonal}(n,k,\lambda) = \{X \in \mathbb{R}^{n\times k}\:\mid\:\left|1-\sigma_i(X)\right| \leq \lambda,\  i=1, \dots, k\}

It is realized via an SVD-like factorization. In particular, it is implemented via the projection

.. math::

    \begin{align*}
        \pi \colon \operatorname{St}(n,k) \times \mathbb{R}^k \times \operatorname{SO}(k)
                &\to \operatorname{AlmostOrthogonal}(n,k,\lambda) \\
            (U, \Sigma, V) &\mapsto Uf_\lambda(\Sigma) V^\intercal
    \end{align*}

where we have identified :math:`\mathbb{R}^k` with a diagonal matrix in :math:`\mathbb{R}^{k \times k}`. The function :math:`f_\lambda\colon \mathbb{R} \to (1-\lambda, 1+\lambda)` takes a function :math:`f\colon \mathbb{R} \to (-1, +1)` and rescales it to be a function on :math:`(1-\lambda, 1+\lambda)` as

.. math::

    f_\lambda(x) = 1+\lambda f(x).

The function :math:`f_\lambda` is then applied element-wise to the diagonal of :math:`\Sigma`.

If :math:`\lambda = 1` is chosen, the resulting space is not a manifold, although this should not hurt optimization in practice.

If :math:`\lambda = 0` is chosen, then the resulting manifold is exactly the :ref:`Special Orthogonal group <RST SO>`. Furthermore, for small values of :math:`\lambda` the algorithm in this class becomes numerically unstable, so we would recommend to choose :class:`geotorch.SO` over this one in that scenario.

.. note::

    There are no restrictions in place for the image of the function :math:`f`. For a function :math:`f` with image :math:`[a,b]`, the function :math:`f_\lambda` will take values in :math:`[\lambda (1+a), \lambda (1+b)]`. As such, rescaling the function :math:`f`, one may use this class to perform optimization with singular values constrained to any prescribed interval of :math:`\mathbb{R}_{\geq 0}`.


.. autoclass:: AlmostOrthogonal
