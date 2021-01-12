import torch
from .lowrank import LowRank
from .exceptions import VectorError, InManifoldError, InverseError
from .utils import _extra_repr


def scaled_sigmoid(t):
    return 2.0 * (torch.sigmoid(t) - 0.5)


def inv_scaled_sigmoid(t):
    y = 0.5 * t + 0.5
    return torch.log(y / (1.0 - y))


class AlmostOrthogonal(LowRank):
    fs = {
        "scaled_sigmoid": (scaled_sigmoid, inv_scaled_sigmoid),
        "tanh": (torch.tanh, torch.atanh),
        "sin": (torch.sin, torch.asin),
    }

    def __init__(self, size, lam, f="sin", triv="expm"):
        r"""Manifold of matrices with singular values in the interval
        :math:`(1-\lambda, 1+\lambda)`.

        The possible default maps are the :math:`\sin,\,\tanh` functions and a scaled
        sigmoid. The sigmoid is scaled as
        :math:`\operatorname{scaled\_sigmoid}(x) = 2\sigma(x) - 1`
        where :math:`\sigma` is the usual sigmoid function.
        This is done so that the image of the scaled sigmoid is :math:`(-1, 1)`.

        Args:
            size (torch.size): Size of the tensor to be parametrized
            lam (float): Radius of the interval. A float in the interval :math:`(0, 1]`
            f (str or callable or pair of callables): Optional. Either:

                - One of ``["scaled_sigmoid", "tanh", "sin"]``

                - A callable that maps real numbers to the interval :math:`(-1, 1)`

                - A pair of callables such that the first maps the real numbers to
                  :math:`(-1, 1)` and the second is a (right) inverse of the first

                Default: ``"sin"``
            triv (str or callable): Optional.
                A map that maps skew-symmetric matrices onto the orthogonal matrices
                surjectively. This is used to optimize the :math:`U` and :math:`V` in
                the SVD. It can be one of ``["expm", "cayley"]`` or a custom callable.
                Default: ``"expm"``

        """
        super().__init__(size, AlmostOrthogonal.rank(size), triv=triv)
        if lam < 0.0 or lam > 1.0:
            raise ValueError("The radius has to be between 0 and 1. Got {}".format(lam))
        self.lam = lam
        f, inv = AlmostOrthogonal.parse_f(f)
        self.f = f
        self.inv = inv

    @staticmethod
    def parse_f(f):
        if f in AlmostOrthogonal.fs.keys():
            return AlmostOrthogonal.fs[f]
        elif callable(f):
            return f, None
        elif isinstance(f, tuple) and callable(f[0]) and callable(f[1]):
            return f
        else:
            raise ValueError(
                "Argument f was not recognized and is "
                "not callable or a pair of callables. "
                "Should be one of {}. Found {}".format(
                    list(AlmostOrthogonal.fs.keys()), f
                )
            )

    @classmethod
    def rank(cls, size):
        if len(size) < 2:
            raise VectorError(cls.__name__, size)
        return min(*size[-2:])

    def submersion(self, U, S, V):
        S = 1.0 + self.lam * self.f(S)
        return super().submersion(U, S, V)

    def submersion_inv(self, X, check_in_manifold=True):
        if self.inv is None:
            raise InverseError(self)
        U, S, V = super().submersion_inv(X)
        if check_in_manifold and not self.in_manifold_singular_values(S):
            raise InManifoldError(X, self)
        # Harcoded epsilon... not a good practice
        if self.lam < 1e-6:
            S = S - 1.0
        else:
            S = self.inv((S - 1.0) / self.lam)
        return U, S, V

    def in_manifold_singular_values(self, S, eps=1e-5):
        lam = self.lam
        if self.lam <= eps:
            lam = eps
        return (
            super().in_manifold_singular_values(S, eps)
            and ((S - 1.0).abs() <= lam).all().item()
        )

    def sample(self, distribution="uniform", init_=None, factorized=True):
        r"""
        Returns a randomly sampled orthogonal matrix according to the specified
        ``distribution``. The options are:

            - ``"uniform"``: Samples a tensor distributed according to the Haar measure
              on :math:`\operatorname{SO}(n)`

            - ``"torus"``: Samples a block-diagonal skew-symmetric matrix.
              The blocks are of the form
              :math:`\begin{pmatrix} 0 & b \\ -b & 0\end{pmatrix}` where :math:`b` is
              distributed according to ``init_``. This matrix will be then projected
              onto :math:`\operatorname{SO}(n)` using ``self.triv``

        .. note

            The ``"torus"`` initialization is particularly useful in recurrent kernels
            of RNNs

        The output of this method can be used to initialize a parametrized tensor
        that has been parametrized with this or any other manifold as::

            >>> layer = nn.Linear(20, 20)
            >>> M = AlmostOrthogonal(layer.weight.size(), lam=0.5)
            >>> geotorch.register_parametrization(layer, "weight", M)
            >>> layer.weight = M.sample()

        Args:
            distribution (string): Optional. One of ``["uniform", "torus"]``.
                    Default: ``"uniform"``
            init\_ (callable): Optional. To be used with the ``"torus"`` option.
                    A function that takes a tensor and fills it in place according
                    to some distribution. See
                    `torch.init <https://pytorch.org/docs/stable/nn.init.html>`_.
                    Default: :math:`\operatorname{Uniform}(-\pi, \pi)`
            factorized (bool): Optional. Return an SVD decomposition of the
                    sampled matrix as a tuple :math:`(U, \Sigma, V)`.
                    Using ``factorized=True`` is more efficient when the result is
                    used to initialize a parametrized tensor.
                    Default: ``True``
        """
        with torch.no_grad():
            device = self[0].base.device
            dtype = self[0].base.dtype
            # Sample U and set S = 1, V = Id
            U = self[0].sample(distribution=distribution, init_=init_)
            S = torch.ones(
                *(self.tensorial_size + (self.n,)), device=device, dtype=dtype
            )
            V = torch.eye(self.n, device=device, dtype=dtype)
            if len(self.tensorial_size) > 0:
                V = V.repeat(*(self.tensorial_size + (1, 1)))

            if factorized:
                return U, S, V
            else:
                Vt = V.transpose(-2, -1)
                # Multiply the three of them, S as a diagonal matrix
                return U @ (S.unsqueeze(-1).expand_as(Vt) * Vt)

    def extra_repr(self):
        return _extra_repr(
            n=self.n,
            lam=self.lam,
            tensorial_size=self.tensorial_size,
            f=self.f,
            no_inv=self.inv is None,
        )
