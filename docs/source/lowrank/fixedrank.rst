Fixed Rank Matrices
===================

.. currentmodule:: geotorch

:math:`\operatorname{FixedRank}(n,k,r)` is the manifold of matrices of rank equal
to :math:`r`, for a given :math:`r \leq \min\{n, k\}`:

.. math::

    \operatorname{FixedRank}(n,k,r) = \{X \in \mathbb{R}^{n\times k}\:\mid\:\operatorname{rank}(X) = r\}

It is realized via an SVD-like factorization. In particular, it is implemented via the projection

.. math::

    \begin{align*}
        \pi \colon \operatorname{St}(n,r) \times \mathbb{R}^r \times \operatorname{St}(k, r)
                &\to \operatorname{FixedRank}(n,k,r) \\
            (U, \Sigma, V) &\mapsto Uf(\Sigma)V^\intercal
    \end{align*}

where we have identified :math:`\mathbb{R}^r` with a diagonal matrix in :math:`\mathbb{R}^{r \times r}`. The function :math:`f\colon \mathbb{R} \to (0, \infty)` is applied element-wise to the diagonal. By default, the `softmax` function is used

.. math::

    \begin{align*}
        \operatorname{softmax} \colon \mathbb{R} &\to (0, \infty) \\
            x &\mapsto \log(1+\exp(x)) + \varepsilon
    \end{align*}

for a small :math:`\varepsilon > 0`.

.. note::

    For practical applications, it will be almost always more convenient to use the class :class:`LowRank`, as it is less restrictive, and most of the times it will converge to a max-rank solution anyway.

.. autoclass:: FixedRank
