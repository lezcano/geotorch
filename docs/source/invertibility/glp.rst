General Linear Group
====================

.. currentmodule:: geotorch

:math:`\operatorname{GL^+}(n)` is the manifold of invertible matrices of positive determinant

.. math::

    \operatorname{GL^+}(n) = \{X \in \mathbb{R}^{n\times n}\:\mid\:\det(X) > 0\}

It is realized via an SVD-like factorization. In particular, it is implemented via the projection

.. math::

    \begin{align*}
        \pi \colon \operatorname{SO}(n) \times \mathbb{R}^n \times \operatorname{SO}(n)
                &\to \operatorname{GL^+}(n) \\
            (U, \Sigma, V) &\mapsto Uf(\Sigma)V^\intercal
    \end{align*}

where we have identified :math:`\mathbb{R}^r` with a diagonal matrix in :math:`\mathbb{R}^{r \times r}`. The function :math:`f\colon \mathbb{R} \to (0, \infty)` is applied element-wise to the diagonal. By default, the `softmax` function is used

.. math::

    \begin{align*}
        \operatorname{softmax} \colon \mathbb{R} &\to (0, \infty) \\
            x &\mapsto \log(1+\exp(x)) + \varepsilon
    \end{align*}

for a small :math:`\varepsilon > 0`.

.. autoclass:: GLp
