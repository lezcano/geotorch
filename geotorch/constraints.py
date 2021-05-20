import torch
import geotorch.parametrize as P

from .symmetric import Symmetric
from .skew import Skew
from .sphere import Sphere, SphereEmbedded
from .stiefel import Stiefel
from .grassmannian import Grassmannian
from .almostorthogonal import AlmostOrthogonal
from .lowrank import LowRank
from .fixedrank import FixedRank
from .glp import GLp
from .psd import PSD
from .pssd import PSSD
from .pssdlowrank import PSSDLowRank
from .pssdfixedrank import PSSDFixedRank


def _register_manifold(module, tensor_name, cls, *args):
    tensor = getattr(module, tensor_name)
    M = cls(tensor.size(), *args).to(device=tensor.device, dtype=tensor.dtype)
    P.register_parametrization(module, tensor_name, M)

    # Initialize without checking in manifold
    X = M.sample()
    param_list = module.parametrizations[tensor_name]
    with torch.no_grad():
        for m in reversed(param_list):
            X = m.right_inverse(X, check_in_manifold=False)
        param_list.original.copy_(X)

    return module


def symmetric(module, tensor_name="weight", lower=True):
    r"""Adds a symmetric parametrization to the matrix ``module.tensor_name``.

    When accessing ``module.tensor_name``, the module will return the parametrized
    version :math:`X` so that :math:`X^\intercal = X`.

    If the tensor has more than two dimensions, the parametrization will be
    applied to the last two dimensions.

    Examples::

        >>> layer = nn.Linear(30, 30)
        >>> geotorch.symmetric(layer, "weight")
        >>> torch.allclose(layer.weight, layer.weight.T)
        True

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied. Default: ``"weight"``
        lower (bool): Optional. Uses the lower triangular part of the matrix to
            parametrize the matrix. Default: ``True``
    """
    P.register_parametrization(module, tensor_name, Symmetric(lower))
    return module


def skew(module, tensor_name="weight", lower=True):
    r"""Adds a skew-symmetric parametrization to the matrix ``module.tensor_name``.

    When accessing ``module.tensor_name``, the module will return the parametrized
    version :math:`X` so that :math:`X^\intercal = -X`.

    If the tensor has more than two dimensions, the parametrization will be
    applied to the last two dimensions.

    Examples::

        >>> layer = nn.Linear(30, 30)
        >>> geotorch.skew(layer, "weight")
        >>> torch.allclose(layer.weight, -layer.weight.T)
        True

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied. Default: ``"weight"``
        lower (bool): Optional. Uses the lower triangular part of the matrix to
            parametrize the matrix. Default: ``True``
    """
    P.register_parametrization(module, tensor_name, Skew(lower))
    return module


def sphere(module, tensor_name="weight", radius=1.0, embedded=False):
    r"""Adds a spherical parametrization to the vector (or tensor) ``module.tensor_name``.

    When accessing ``module.tensor_name``, the module will return the parametrized
    version :math:`v` so that :math:`\lVert v \rVert = 1`.

    If the tensor has more than one dimension, the parametrization will be
    applied to the last dimension.

    Examples::

        >>> layer = nn.Linear(20, 30)
        >>> geotorch.sphere(layer, "bias")
        >>> torch.norm(layer.bias)
        tensor(1.)
        >>> geotorch.sphere(layer, "weight")  # Make the columns unit norm
        >>> torch.allclose(torch.norm(layer.weight, dim=-1), torch.ones(30))
        True

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied. Default: ``"weight"``
        radius (float): Optional.
            Radius of the sphere. It has to be positive. Default: 1.
        embedded (bool): Optional.
            Chooses between the implementation of the sphere using the exponential
            map (``embedded=False``) and that using the projection from the ambient space (``embedded=True``)
            Default. ``True``
    """
    cls = SphereEmbedded if embedded else Sphere
    return _register_manifold(module, tensor_name, cls, radius)


def orthogonal(module, tensor_name="weight", triv="expm"):
    r"""Adds an orthogonal parametrization to the tensor ``module.tensor_name``.

    When accessing ``module.tensor_name``, the module will return the
    parametrized version :math:`X` so that :math:`X^\intercal X = \operatorname{I}`.

    If the tensor has more than two dimensions, the parametrization will be
    applied to the last two dimensions.

    Examples::

        >>> layer = nn.Linear(20, 30)
        >>> geotorch.orthogonal(layer, "weight")
        >>> torch.norm(layer.weight.T @ layer.weight - torch.eye(20,20))
        tensor(4.8488e-05)

        >>> layer = nn.Conv2d(20, 40, 3, 3)  # Make the kernels orthogonal
        >>> geotorch.orthogonal(layer, "weight")
        >>> torch.norm(layer.weight.transpose(-2, -1) @ layer.weight - torch.eye(3,3).repeat(40,20,1,1))
        tensor(1.2225e-05)

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied. Default: ``"weight"``
        triv (str or callable): Optional.
            A map that maps a skew-symmetric matrix to an orthogonal matrix.
            It can be the exponential of matrices or the cayley transform passing
            ``["expm", "cayley"]`` or a custom callable.  Default: ``"expm"``
    """
    return _register_manifold(module, tensor_name, Stiefel, triv)


def almost_orthogonal(module, tensor_name="weight", lam=0.1, f="sin", triv="expm"):
    r"""Adds an almost orthogonal parametrization to the tensor ``module.tensor_name``.

    When accessing ``module.tensor_name``, the module will return the
    parametrized version :math:`X` which will have its singular values in
    the interval :math:`[1-\texttt{lam}, 1+\texttt{lam}]`

    If the tensor has more than two dimensions, the parametrization will be
    applied to the last two dimensions.

    Examples::

        >>> layer = nn.Linear(20, 30)
        >>> geotorch.almost_orthogonal(layer, "weight", 0.5)
        >>> S = torch.linalg.svd(layer.weight).S
        >>> all(S >= 0.5 and S <= 1.5)
        True

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied. Default: ``"weight"``
        lam (float): Radius of the interval for the singular values. A float in the interval :math:`[0, 1]`. Default: ``0.1``
        f (str or callable or pair of callables): Optional. Either:

            - One of ``["scaled_sigmoid", "tanh", "sin"]``

            - A callable that maps real numbers to the interval :math:`[-1, 1]`

            - A pair of callables such that the first maps the real numbers to
              :math:`[-1, 1]` and the second is a (right) inverse of the first

            Default: ``"sin"``
        triv (str or callable): Optional.
            A map that maps skew-symmetric matrices onto the orthogonal matrices
            surjectively. This is used to optimize the :math:`U` and :math:`V` in the
            SVD. It can be one of ``["expm", "cayley"]`` or a custom
            callable. Default: ``"expm"``
    """
    return _register_manifold(module, tensor_name, AlmostOrthogonal, lam, f, triv)


def grassmannian(module, tensor_name="weight", triv="expm"):
    r"""Adds an parametrization to the tensor ``module.tensor_name`` so that the
    result represents a subspace. If the initial matrix was of size :math:`n \times k`
    the parametrized matrix will represent a subspace of dimension :math:`k` of
    :math:`\mathbb{R}^n`.

    When accessing ``module.tensor_name``, the module will return the parametrized
    version :math:`X` so that :math:`X` represents :math:`k` orthogonal vectors of
    :math:`\mathbb{R}^n` that span the subspace. That is, the resulting matrix will
    be orthogonal, :math:`X^\intercal X = \operatorname{I}`.

    If the tensor has more than two dimensions, the parametrization will be
    applied to the last two dimensions.

    .. note::

        Even though this space resembles that generated by :func:`geotorch.orthogonal`,
        it is actually a subspace of that, as every subspace can be represented by many
        different basis of vectors that span it.

    Examples::

        >>> layer = nn.Linear(20, 30)
        >>> geotorch.grassmannian(layer, "weight")
        >>> torch.norm(layer.weight.t() @ layer.weight - torch.eye(20,20))
        tensor(1.8933e-05)

        >>> layer = nn.Conv2d(20, 40, 3, 3)  # Make the kernels represent subspaces
        >>> geotorch.grassmannian(layer, "weight")
        >>> torch.norm(layer.weight.transpose(-2, -1) @ layer.weight - torch.eye(3,3).repeat(40,20,1,1))
        tensor(8.3796-06)

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied. Default: ``"weight"``
        triv (str or callable): Optional.
            A map that maps a skew-symmetric matrix to an orthogonal matrix.
            It can be the exponential of matrices or the cayley transform passing
            ``["expm", "cayley"]`` or a custom callable.  Default: ``"expm"``
    """
    return _register_manifold(module, tensor_name, Grassmannian, triv)


def low_rank(module, tensor_name, rank, triv="expm"):
    r"""Adds a low rank parametrization to the tensor ``module.tensor_name``.

    When accessing ``module.tensor_name``, the module will return the
    parametrized version :math:`X` which will have rank at most ``rank``.

    If the tensor has more than two dimensions, the parametrization will be
    applied to the last two dimensions.

    Examples::

        >>> layer = nn.Linear(20, 30)
        >>> geotorch.low_rank(layer, "weight", 4)
        >>> list(torch.linalg.svd(layer.weight).S > 1e-7).count(True) <= 4
        True

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied
        rank (int): Rank of the matrix.
            It has to be less than the minimum of the two dimensions of the
            matrix
        triv (str or callable): Optional.
            A map that maps skew-symmetric matrices onto the orthogonal matrices
            surjectively. This is used to optimize the :math:`U` and :math:`V` in the
            SVD. It can be one of ``["expm", "cayley"]`` or a custom
            callable. Default: ``"expm"``
    """
    return _register_manifold(module, tensor_name, LowRank, rank, triv)


def fixed_rank(module, tensor_name, rank, f="softplus", triv="expm"):
    r"""Adds a fixed rank parametrization to the tensor ``module.tensor_name``.

    When accessing ``module.tensor_name``, the module will return the
    parametrized version :math:`X` which will have rank equal to ``rank``.

    If the tensor has more than two dimensions, the parametrization will be
    applied to the last two dimensions.

    Examples::

        >>> layer = nn.Linear(20, 30)
        >>> geotorch.fixed_rank(layer, "weight", 5)
        >>> list(torch.linalg.svd(layer.weight).S > 1e-7).count(True)
        5

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied
        rank (int): Rank of the matrix.
            It has to be less than the minimum of the two dimensions of the
            matrix
        f (str or callable or pair of callables): Optional. Either:

            - ``"softplus"``

            - A callable that maps real numbers to the interval :math:`(0, \infty)`

            - A pair of callables such that the first maps the real numbers to
              :math:`(0, \infty)` and the second is a (right) inverse of the first

            Default: ``"softplus"``
        triv (str or callable): Optional.
            A map that maps skew-symmetric matrices onto the orthogonal matrices
            surjectively. This is used to optimize the :math:`U` and :math:`V` in the
            SVD. It can be one of ``["expm", "cayley"]`` or a custom
            callable. Default: ``"expm"``
    """
    return _register_manifold(module, tensor_name, FixedRank, rank, f, triv)


def invertible(module, tensor_name="weight", f="softplus", triv="expm"):
    r"""Adds an invertibility constraint to the tensor ``module.tensor_name``.

    When accessing ``module.tensor_name``, the module will return the
    parametrized version :math:`X` which will have positive determinant and,
    in particular, it will be invertible.

    If the tensor has more than two dimensions, the parametrization will be
    applied to the last two dimensions.

    Examples::

        >>> layer = nn.Linear(20, 20)
        >>> geotorch.invertible(layer, "weight", 5)
        >>> torch.det(layer.weight) > 0.0
        True

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied. Default: ``"weight"``
        f (str or callable or pair of callables): Optional. Either:

            - ``"softplus"``

            - A callable that maps real numbers to the interval :math:`(0, \infty)`

            - A pair of callables such that the first maps the real numbers to
              :math:`(0, \infty)` and the second is a (right) inverse of the first

            Default: ``"softplus"``
        triv (str or callable): Optional.
            A map that maps skew-symmetric matrices onto the orthogonal matrices
            surjectively. This is used to optimize the :math:`U` and :math:`V` in the
            SVD. It can be one of ``["expm", "cayley"]`` or a custom
            callable. Default: ``"expm"``
    """
    return _register_manifold(module, tensor_name, GLp, f, triv)


def positive_definite(module, tensor_name="weight", f="softplus", triv="expm"):
    r"""Adds a positive definiteness constraint to the tensor ``module.tensor_name``.

    When accessing ``module.tensor_name``, the module will return the
    parametrized version :math:`X` which will be symmetric and with positive
    eigenvalues

    If the tensor has more than two dimensions, the parametrization will be
    applied to the last two dimensions.

    Examples::

        >>> layer = nn.Linear(20, 20)
        >>> geotorch.positive_definite(layer, "weight")
        >>> (torch.linalg.eigvalsh(layer.weight) > 0.0).all()
        tensor(True)

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied. Default: ``"weight"``
        f (str or callable or pair of callables): Optional. Either:

            - ``"softplus"``

            - A callable that maps real numbers to the interval :math:`(0, \infty)`

            - A pair of callables such that the first maps the real numbers to
              :math:`(0, \infty)` and the second is a (right) inverse of the first

            Default: ``"softplus"``
        triv (str or callable): Optional.
            A map that maps skew-symmetric matrices onto the orthogonal
            matrices surjectively. This is used to optimize the :math:`Q` in the eigenvalue
            decomposition. It can be one of ``["expm", "cayley"]`` or a custom
            callable. Default: ``"expm"``
    """
    return _register_manifold(module, tensor_name, PSD, f, triv)


def positive_semidefinite(module, tensor_name="weight", triv="expm"):
    r"""Adds a positive definiteness constraint to the tensor ``module.tensor_name``.

    When accessing ``module.tensor_name``, the module will return the
    parametrized version :math:`X` which will be symmetric and with
    non-negative eigenvalues

    If the tensor has more than two dimensions, the parametrization will be
    applied to the last two dimensions.

    Examples::

        >>> layer = nn.Linear(20, 20)
        >>> geotorch.positive_semidefinite(layer, "weight")
        >>> L = torch.linalg.eigvalsh(layer.weight)
        >>> L[L.abs() < 1e-7] = 0.0  # Round errors
        >>> (L >= 0.0).all()
        tensor(True)

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied. Default: ``"weight"``
        triv (str or callable): Optional.
            A map that maps skew-symmetric matrices onto the orthogonal
            matrices surjectively. This is used to optimize the :math:`Q` in the eigenvalue
            decomposition. It can be one of ``["expm", "cayley"]`` or a custom
            callable. Default: ``"expm"``
    """
    return _register_manifold(module, tensor_name, PSSD, triv)


def positive_semidefinite_low_rank(module, tensor_name, rank, triv="expm"):
    r"""Adds a positive definiteness constraint to the tensor ``module.tensor_name``.

    When accessing ``module.tensor_name``, the module will return the
    parametrized version :math:`X` which will be symmetric and with non-negative
    eigenvalues and at most ``rank`` of them non-zero.

    If the tensor has more than two dimensions, the parametrization will be
    applied to the last two dimensions.

    Examples::

        >>> layer = nn.Linear(20, 20)
        >>> geotorch.positive_semidefinite_low_rank(layer, "weight", 5)
        >>> L = torch.linalg.eigvalsh(layer.weight)
        >>> L[L.abs() < 1e-7] = 0.0  # Round errors
        >>> (L >= 0.0).all()
        tensor(True)
        >>> list(L > 0.0).count(True) <= 5
        True

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied
        rank (int): Rank of the matrix.
            It has to be less than the minimum of the two dimensions of the
            matrix
        triv (str or callable): Optional.
            A map that maps skew-symmetric matrices onto the orthogonal
            matrices surjectively. This is used to optimize the :math:`Q` in the eigenvalue
            decomposition. It can be one of ``["expm", "cayley"]`` or a custom
            callable. Default: ``"expm"``
    """
    return _register_manifold(module, tensor_name, PSSDLowRank, rank, triv)


def positive_semidefinite_fixed_rank(
    module, tensor_name, rank, f="softplus", triv="expm"
):
    r"""Adds a positive definiteness constraint to the tensor ``module.tensor_name``.

    When accessing ``module.tensor_name``, the module will return the
    parametrized version :math:`X` which will be symmetric and with non-negative
    eigenvalues and exactly ``rank`` of them non-zero.

    If the tensor has more than two dimensions, the parametrization will be
    applied to the last two dimensions.

    Examples::

        >>> layer = nn.Linear(20, 20)
        >>> geotorch.positive_semidefinite_fixed_rank(layer, "weight", 5)
        >>> L = torch.linalg.eigvalsh(layer.weight)
        >>> L[L.abs() < 1e-7] = 0.0  # Round errors
        >>> (L >= 0.0).all()
        tensor(True)
        >>> list(L > 0.0).count(True)
        5

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied
        rank (int): Rank of the matrix.
            It has to be less than the minimum of the two dimensions of the
            matrix
        f (str or callable or pair of callables): Optional. Either:

            - ``"softplus"``

            - A callable that maps real numbers to the interval :math:`(0, \infty)`

            - A pair of callables such that the first maps the real numbers to
              :math:`(0, \infty)` and the second is a (right) inverse of the first

            Default: ``"softplus"``
        triv (str or callable): Optional.
            A map that maps skew-symmetric matrices onto the orthogonal
            matrices surjectively. This is used to optimize the :math:`Q` in the
            eigenvalue decomposition. It can be one of ``["expm", "cayley"]`` or
            a custom callable. Default: ``"expm"``
    """
    return _register_manifold(module, tensor_name, PSSDFixedRank, rank, f, triv)
