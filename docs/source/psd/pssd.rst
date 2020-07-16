Positive Semidefinite Matrices
==============================

.. currentmodule:: geotorch

:math:`\operatorname{PSSD}(n)` is the algebraic variety of positive semidefinite matrices.

.. math::

    \operatorname{PSSD}(n,r) = \{X \in \mathbb{R}^{n\times n}\:\mid\:X \succeq 0\}

It is realized via an eigen-like factorization. In particular, it is implemented via the projection

.. math::

    \begin{align*}
        \pi \colon \operatorname{SO}(n) \times \mathbb{R}^n
                &\to \operatorname{PSSD}(n) \\
            (Q, \Lambda) &\mapsto Q\left|\Lambda\right|Q^\intercal
    \end{align*}

where we have identified :math:`\mathbb{R}^r` with a diagonal matrix in :math:`\mathbb{R}^{r \times r}` and :math:`\left|Lambda\right|` denotes the absolute value of the diagonal entries.

.. autoclass:: PSSD
