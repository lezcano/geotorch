Fixed Rank Matrices
===================

.. currentmodule:: geotorch

:math:`\operatorname{FixedRank}(n,k,r)` is the manifold of matrices of rank equal
to :math:`r`, for a given :math:`r \leq \min\{n, k\}`:

.. math::

    \operatorname{FixedRank}(n,k,r) = \{X \in \mathbb{R}^{n\times k}\:\mid\:\operatorname{rank}(X) = r\}

It is realized via an SVD-like factorization:

.. math::

    \begin{align*}
        \pi \colon \operatorname{St}(n,r) \times \mathbb{R}^r \times \operatorname{St}(k, r)
                &\to \operatorname{FixedRank}(n,k,r) \\
            (U, \Sigma, V) &\mapsto Uf(\Sigma)V^\intercal
    \end{align*}

where we have identified the vector :math:`\Sigma` with a diagonal matrix in :math:`\mathbb{R}^{r \times r}`. The function :math:`f\colon \mathbb{R} \to (0, \infty)` is applied element-wise to the diagonal. By default, the `softmax` function is used

.. math::

    \begin{align*}
        \operatorname{softmax} \colon \mathbb{R} &\to (0, \infty) \\
            x &\mapsto \log(1+\exp(x)) + \varepsilon
    \end{align*}

where we use a small :math:`\varepsilon > 0` for numerical stability.

.. note::

    For practical applications, it will be almost always more convenient to use the class :class:`LowRank`, as it is less restrictive, and most of the times it will converge to a max-rank solution anyway.

.. autoclass:: FixedRank

    .. automethod:: sample
    .. automethod:: in_manifold
