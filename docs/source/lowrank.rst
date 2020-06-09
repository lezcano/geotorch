Variety of Low Rank Matrices: :math:`\operatorname{LowRank}(n,k,r)`
===================================================================

.. currentmodule:: geotorch

:math:`\operatorname{LowRank}(n,k,r)` is the algebraic variety of matrices of rank less or equal
to :math:`r`, with :math:`r \leq \min\{n, k\}`:

.. math::

    \operatorname{LowRank}(n,k,r) = \{X \in \mathbb{R}^{n\times k}\:\mid\:\operatorname{rank}(X) \leq r\}

It is realized via a SVD-like factorization. In particular, it is implemented via the projection

.. math::

    \begin{align*}
        \pi \colon \operatorname{St}(n,r) \times \mathbb{R}^r \times \operatorname{St}(k, r)
                &\to \operatorname{LowRank}(n,k,r) \\
            (U, \Sigma, V) &\mapsto U\Sigma V^\intercal
    \end{align*}

where we have identified :math:`\mathbb{R}^r` with a diagonal matrix in :math:`\mathbb{R}^{r \times r}`.

.. autoclass:: LowRank
