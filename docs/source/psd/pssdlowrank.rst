Positive Semidefinite Low Rank Matrices
========================================

.. currentmodule:: geotorch

:math:`\operatorname{PSSDLowRank}(n,r)` is the algebraic variety of positive semidefinite matrices
of rank less or equal to :math:`r`, for a given :math:`r \leq n`:

.. math::

    \operatorname{PSSDLowRank}(n,r) = \{X \in \mathbb{R}^{n\times n}\:\mid\:X \succeq 0,\,\operatorname{rank}(X) \leq r\}

It is realized via an eigen-like factorization. In particular, it is implemented via the projection

.. math::

    \begin{align*}
        \pi \colon \operatorname{St}(n,r) \times \mathbb{R}^r
                &\to \operatorname{PSSDLowRank}(n,r) \\
            (Q, \Lambda) &\mapsto Q\left|\Lambda\right| Q^\intercal
    \end{align*}

where we have identified :math:`\mathbb{R}^r` with a diagonal matrix in :math:`\mathbb{R}^{r \times r}` and :math:`\left|\Lambda\right|` denotes the absolute value of the diagonal entries.

.. autoclass:: PSSDLowRank
