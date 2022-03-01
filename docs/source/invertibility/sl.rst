General Linear Group
====================

.. currentmodule:: geotorch

:math:`\operatorname{SL}(n)` is the manifold of matrices of determinant equal to 1.

.. math::

    \operatorname{SL}(n) = \{X \in \mathbb{R}^{n\times n}\:\mid\:\det(X) = 1\}

It is realized via an SVD-like factorization:

.. math::

    \begin{align*}
        \pi \colon \operatorname{SO}(n) \times \mathbb{R}^n \times \operatorname{SO}(n)
                &\to \operatorname{SL}(n) \\
            (U, \Sigma, V) &\mapsto Uf(\Sigma)V^\intercal
    \end{align*}

where we have identified the vector :math:`\Sigma` with a diagonal matrix in :math:`\mathbb{R}^{n \times n}`. The function :math:`f\colon \mathbb{R} \to (\varepsilon, \infty)` is applied element-wise to the diagonal for a small :math:`\varepsilon > 0`. By default, a combination of the `softplus` function

.. math::

    \begin{align*}
        \operatorname{softplus} \colon \mathbb{R} &\to (\varepsilon, \infty) \\
            x &\mapsto \log(1+\exp(x)) + \varepsilon
    \end{align*}

composed with the normalization function

.. math::

    \begin{align*}
        \operatorname{g} \colon \mathbb{R}^n &\to (\varepsilon, \infty)^n \\
            (x_1, \dots, x_n) &\mapsto \left(\frac{x_i}{\sqrt[\leftroot{-2}\uproot{2}n]{\prod_i x_i}}\right)_i
    \end{align*}

to ensure that the product of all the singular values is equal to 1.

.. autoclass:: SL

    .. automethod:: sample
    .. automethod:: in_manifold
