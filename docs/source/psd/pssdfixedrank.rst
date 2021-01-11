Positive Semidefinite Fixed Rank Matrices
=========================================

.. currentmodule:: geotorch

:math:`\operatorname{PSSDFixedRank}(n,r)` is the manifold of positive semidefinite matrices with rank equal
to :math:`r`, for a given :math:`r \leq n`:

.. math::

    \operatorname{PSSDFixedRank}(n,r) = \{X \in \mathbb{R}^{n\times n}\:\mid\:X \succeq 0,\,\operatorname{rank}(X) = r\}

It is realized via an eigenvalue-like factorization:

.. math::

    \begin{align*}
        \pi \colon \operatorname{St}(n,r) \times \mathbb{R}^r
                &\to \operatorname{PSSDFixedRank}(n,r) \\
            (Q, \Lambda) &\mapsto Qf(\Lambda)Q^\intercal
    \end{align*}

where we have identified the vector :math:`\Lambda` with a diagonal matrix in :math:`\mathbb{R}^{r \times r}`. The function :math:`f\colon \mathbb{R} \to (0, \infty)` is applied element-wise to the diagonal. By default, the `softmax` function is used

.. math::

    \begin{align*}
        \operatorname{softmax} \colon \mathbb{R} &\to (0, \infty) \\
            x &\mapsto \log(1+\exp(x)) + \varepsilon
    \end{align*}

where we use a small :math:`\varepsilon > 0` for numerical stability.

.. note::

    For practical applications, it will be almost always more convenient to use the class :class:`geotorch.PSSDLowRank`, as it is less restrictive, and most of the times it will converge to a max-rank solution anyway.

.. autoclass:: PSSDFixedRank
