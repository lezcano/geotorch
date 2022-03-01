General Linear Group
====================

.. currentmodule:: geotorch

:math:`\operatorname{GL^+}(n)` is the manifold of invertible matrices of positive determinant

.. math::

    \operatorname{GL^+}(n) = \{X \in \mathbb{R}^{n\times n}\:\mid\:\det(X) > 0\}

It is realized via an SVD-like factorization:

.. math::

    \begin{align*}
        \pi \colon \operatorname{SO}(n) \times \mathbb{R}^n \times \operatorname{SO}(n)
                &\to \operatorname{GL^+}(n) \\
            (U, \Sigma, V) &\mapsto Uf(\Sigma)V^\intercal
    \end{align*}

where we have identified the vector :math:`\Sigma` with a diagonal matrix in :math:`\mathbb{R}^{n \times n}`. The function :math:`f\colon \mathbb{R} \to (0, \infty)` is applied element-wise to the diagonal. By default, the `softplus` function is used

.. math::

    \begin{align*}
        \operatorname{softplus} \colon \mathbb{R} &\to (0, \infty) \\
            x &\mapsto \log(1+\exp(x)) + \varepsilon
    \end{align*}

where we use a small :math:`\varepsilon > 0` for numerical stability.

.. autoclass:: GLp

    .. automethod:: sample
    .. automethod:: in_manifold
