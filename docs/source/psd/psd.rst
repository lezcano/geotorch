Positive Definite Matrices
==========================

.. currentmodule:: geotorch

:math:`\operatorname{PSD}(n)` is the manifold of positive definite matrices.

.. math::

    \operatorname{PSD}(n) = \{X \in \mathbb{R}^{n\times n}\:\mid\:X \succ 0\}

It is realized via an eigen-like factorization. In particular, it is implemented via the projection

.. math::

    \begin{align*}
        \pi \colon \operatorname{SO}(n) \times \mathbb{R}^n
                &\to \operatorname{PSD}(n) \\
            (Q, \Lambda) &\mapsto Qf(\Lambda)Q^\intercal
    \end{align*}

where we have identified :math:`\mathbb{R}^r` with a diagonal matrix in :math:`\mathbb{R}^{r \times r}` and :math:`\left|Lambda\right|` denotes the absolute value of the diagonal entries. The function :math:`f\colon \mathbb{R} \to (0, \infty)` is applied element-wise to the diagonal. By default, the `softmax` function is used

.. math::

    \begin{align*}
        \operatorname{softmax} \colon \mathbb{R} &\to (0, \infty) \\
            x &\mapsto \log(1+\exp(x)) + \varepsilon
    \end{align*}

for a small :math:`\varepsilon > 0`.

.. note::

    For practical applications, it is more convenient to use the class :class:`geotorch.PSSD`, unless the positive definiteness condition is essential. This is because :class:`geotorch.PSSD` is less restrictive, and most of the times it will converge to a max-rank solution anyway, although in the optimization process there might be times when the matrix might become almost singular.

.. autoclass:: PSD
