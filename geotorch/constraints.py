import geotorch.parametrize as P

from .symmetric import Symmetric
from .skew import Skew
from .sphere import Sphere
from .so import SO
from .stiefel import Stiefel, StiefelTall
from .grassmannian import Grassmannian, GrassmannianTall
from .almostorthogonal import AlmostOrthogonal
from .lowrank import LowRank
from .fixedrank import FixedRank
from .glp import GLp
from .psd import PSD
from .pssd import PSSD
from .pssdlowrank import PSSDLowRank
from .pssdfixedrank import PSSDFixedRank


def symmetric(module, tensor_name, lower=True):
    r"""Adds a symmetric parametrization to the matrix ``module[tensor_name]``.

    When accessing ``module[tensor_name]``, the module will return the parametrized
    version :math:`X` so that :math:`X^\intercal = X`.

    If the tensor has more than two dimensions, the parametrization will be
    applied to the last two dimensions.

    Examples::

        >>> layer = nn.Linear(30, 30)
        >>> geotorch.symmetric(layer, "weight")
        >>> torch.norm(layer.weight - layer.weight.t())
        tensor(0.)

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied
        lower (bool): Optional. Uses the lower triangular part of the matrix to
            parametrize the matrix. Default: `True`
    """
    size = getattr(module, tensor_name).size()
    P.register_parametrization(module, tensor_name, Symmetric(size, lower))


def skew(module, tensor_name, lower=True):
    r"""Adds a skew-symmetric parametrization to the matrix ``module[tensor_name]``.

    When accessing ``module[tensor_name]``, the module will return the parametrized
    version :math:`X` so that :math:`X^\intercal = -X`.

    If the tensor has more than two dimensions, the parametrization will be
    applied to the last two dimensions.

    Examples::

        >>> layer = nn.Linear(30, 30)
        >>> geotorch.skew(layer, "weight")
        >>> torch.norm(layer.weight + layer.weight.t())
        tensor(0.)

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied
        lower (bool): Optional. Uses the lower triangular part of the matrix to
            parametrize the matrix. Default: `True`
    """
    size = getattr(module, tensor_name).size()
    P.register_parametrization(module, tensor_name, Skew(size, lower))


def sphere(module, tensor_name, r=1.0):
    r"""Adds a spherical parametrization to the vector (or tensor)
    ``module[tensor_name]``.

    When accessing ``module[tensor_name]``, the module will return the parametrized
    version :math:`v` so that :math:`\lVert v \rVert = 1`.

    If the tensor has more than one dimension, the parametrization will be
    applied to the last dimension.

    Examples::

        >>> layer = nn.Linear(20, 30)
        >>> geotorch.sphere(layer, "bias")
        >>> torch.norm(layer.bias)
        tensor(1.)
        >>> geotorch.sphere(layer, "weight")  # Make the columns unit norm
        >>> torch.norm(torch.norm(layer.weight, dim=-1) - torch.ones(30))
        tensor(6.1656e-07)

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied
        r (float): Optional.
            Radius of the sphere. It has to be positive. Default: 1.
    """
    size = getattr(module, tensor_name).size()
    P.register_parametrization(module, tensor_name, Sphere(size, r))


def orthogonal(module, tensor_name, triv="expm"):
    r"""Adds an orthogonal parametrization to the tensor ``module[tensor_name]``.

    When accessing ``module[tensor_name]``, the module will return the
    parametrized version :math:`X` so that :math:`X^\intercal X = \operatorname{Id}`.

    If the tensor has more than two dimensions, the parametrization will be
    applied to the last two dimensions.

    Examples::

        >>> layer = nn.Linear(20, 30)
        >>> geotorch.orthogonal(layer, "weight")
        >>> torch.norm(layer.weight.t() @ layer.weight - torch.eye(20,20))
        tensor(4.8488e-05)

        >>> layer = nn.Conv2d(20, 40, 3, 3)  # Make the kernels orthogonal
        >>> geotorch.orthogonal(layer, "weight")
        >>> torch.norm(layer.weight.transpose(-2, -1) @ layer.weight - torch.eye(3,3).repeat(40,20,1,1))
        tensor(1.2225e-05)

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied
        triv (str or callable): Optional.
            A map that maps a skew-symmetric matrix to an orthogonal matrix.
            It can be the exponential of matrices or the cayley transform passing
            `["expm", "cayley"]` or a custom callable.  Default: `"expm"`
    """
    size = getattr(module, tensor_name).size()
    if len(size) < 2:
        raise ValueError(
            "Cannot put orthogonal constraints on a vector. "
            "Got a tensor of size {}".format(size)
        )
    n, k = size[-2:]
    n, k = max(n, k), min(n, k)
    if n == k:
        cls = SO
    elif n > 4 * k:
        cls = StiefelTall
    else:
        cls = Stiefel
    P.register_parametrization(module, tensor_name, cls(size, triv))


def almost_orthogonal(module, tensor_name, lam, f="sigmoid", triv="expm"):
    r"""Adds an almost orthogonal parametrization to the tensor ``module[tensor_name]``.

    When accessing ``module[tensor_name]``, the module will return the
    parametrized version :math:`X` which will have its singular values in
    the interval :math:`[1-\texttt{lam}, 1+\texttt{lam}]`

    If the tensor has more than two dimensions, the parametrization will be
    applied to the last two dimensions.

    Examples::

        >>> layer = nn.Linear(20, 30)
        >>> geotorch.almost_orthogonal(layer, "weight", 0.5)
        >>> S = torch.svd(layer.weight).S
        >>> all(S >= 0.5 and S <= 1.5)
        True

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied
        lam (float): Radius. A float in the interval [0, 1]
        f (str or callable): Optional. One of `["sigmoid", "tanh", "sin"]`
            or a callable that maps real numbers onto the interval :math:`[-1, 1]`.
            Default: `"sigmoid"`
        triv (str or callable): Optional.
            A map that maps :math:`\operatorname{Skew}(n)` onto the orthogonal
            matrices surjectively. This is used to optimize the U, V in the
            SVD. It can be one of `["expm", "cayley"]` or a custom
            callable. Default: `"expm"`
    """
    size = getattr(module, tensor_name).size()
    P.register_parametrization(
        module, tensor_name, AlmostOrthogonal(size, lam, f, triv)
    )


def grassmannian(module, tensor_name, triv="expm"):
    r"""Adds an parametrization to the tensor ``module[tensor_name]`` so that the
    result represents a subspace. If the initial matrix was of size :math:`n \times k`
    the parametrized matrix will represent a subspace of dimension :math:`k` of
    :math:`\mathbb{R}^n`.

    When accessing ``module[tensor_name]``, the module will return the parametrized
    version :math:`X` so that :math:`X` represents :math:`k` orthogonal vectors of
    :math:`\mathbb{R}^n` that span the subspace. That is, the resulting matrix will
    be orthogonal, :math:`X^\intercal X = \operatorname{Id}`.

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
            on which the parametrization will be applied
        triv (str or callable): Optional.
            A map that maps a skew-symmetric matrix to an orthogonal matrix.
            It can be the exponential of matrices or the cayley transform passing
            `["expm", "cayley"]` or a custom callable.  Default: `"expm"`
    """
    size = getattr(module, tensor_name).size()
    if len(size) < 2:
        raise ValueError(
            "Cannot put grassmannian constraints on a vector. "
            "Got a tensor of size {}".format(size)
        )
    n, k = size[-2:]
    n, k = max(n, k), min(n, k)
    cls = GrassmannianTall if n > 4 * k else Grassmannian
    P.register_parametrization(module, tensor_name, cls(size, triv))


def low_rank(module, tensor_name, rank, triv="expm"):
    r"""Adds a low rank parametrization to the tensor ``module[tensor_name]``.

    When accessing ``module[tensor_name]``, the module will return the
    parametrized version :math:`X` which will have rank at most ``rank``.

    If the tensor has more than two dimensions, the parametrization will be
    applied to the last two dimensions.

    Examples::

        >>> layer = nn.Linear(20, 30)
        >>> geotorch.low_rank(layer, "weight", 4)
        >>> list(torch.svd(layer.weight).S > 1e-7).count(True) <= 4
        True

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied
        rank (int): Rank of the matrix.
            It has to be less than the minimum of the two dimensions of the
            matrix
        triv (str or callable): Optional.
            A map that maps :math:`\operatorname{Skew}(n)` onto the orthogonal
            matrices surjectively. This is used to optimize the U, V in the
            SVD. It can be one of `["expm", "cayley"]` or a custom
            callable. Default: `"expm"`
    """
    size = getattr(module, tensor_name).size()
    P.register_parametrization(module, tensor_name, LowRank(size, rank, triv))


def fixed_rank(module, tensor_name, rank, f="softplus", triv="expm"):
    r"""Adds a fixed rank parametrization to the tensor ``module[tensor_name]``.

    When accessing ``module[tensor_name]``, the module will return the
    parametrized version :math:`X` which will have rank equal to ``rank``.

    If the tensor has more than two dimensions, the parametrization will be
    applied to the last two dimensions.

    Examples::

        >>> layer = nn.Linear(20, 30)
        >>> geotorch.fixed_rank(layer, "weight", 5)
        >>> list(torch.svd(layer.weight).S > 1e-7).count(True)
        5

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied
        rank (int): Rank of the matrix.
            It has to be less than the minimum of the two dimensions of the
            matrix
        f (str or callable): Optional. The string `"softplus"` or a callable
            that maps real numbers onto the interval :math:`(0, \infty)`. Default: `"softplus"`
        triv (str or callable): Optional.
            A map that maps :math:`\operatorname{Skew}(n)` onto the orthogonal
            matrices surjectively. This is used to optimize the U, V in the
            SVD. It can be one of `["expm", "cayley"]` or a custom
            callable. Default: `"expm"`
    """
    size = getattr(module, tensor_name).size()
    P.register_parametrization(module, tensor_name, FixedRank(size, rank, f, triv))


def invertible(module, tensor_name, f="softplus", triv="expm"):
    r"""Adds an invertibility constraint to the tensor ``module[tensor_name]``.

    When accessing ``module[tensor_name]``, the module will return the
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
            on which the parametrization will be applied
        f (str or callable): Optional. The string `"softplus"` or a callable
            that maps real numbers onto the interval :math:`(0, \infty)`. Default: `"softplus"`
        triv (str or callable): Optional.
            A map that maps :math:`\operatorname{Skew}(n)` onto the orthogonal
            matrices surjectively. This is used to optimize the U, V in the
            SVD. It can be one of `["expm", "cayley"]` or a custom
            callable. Default: `"expm"`
    """
    size = getattr(module, tensor_name).size()
    P.register_parametrization(module, tensor_name, GLp(size, f, triv))


def positive_definite(module, tensor_name, f="softplus", triv="expm"):
    r"""Adds a positive definiteness constraint to the tensor
    ``module[tensor_name]``.

    When accessing ``module[tensor_name]``, the module will return the
    parametrized version :math:`X` which will be symmetric and with positive
    eigenvalues

    If the tensor has more than two dimensions, the parametrization will be
    applied to the last two dimensions.

    Examples::

        >>> layer = nn.Linear(20, 20)
        >>> geotorch.positive_definite(layer, "weight")
        >>> (torch.symeig(layer.weight).eigenvalues > 0.0).all()
        tensor(True)

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied
        f (str or callable): Optional. The string `"softplus"` or a callable
            that maps real numbers onto the interval :math:`(0, \infty)`. Default: `"softplus"`
        triv (str or callable): Optional.
            A map that maps :math:`\operatorname{Skew}(n)` onto the orthogonal
            matrices surjectively. This is used to optimize the Q in the eigenvalue
            decomposition. It can be one of `["expm", "cayley"]` or a custom
            callable. Default: `"expm"`
    """
    size = getattr(module, tensor_name).size()
    P.register_parametrization(module, tensor_name, PSD(size, f, triv))


def positive_semidefinite(module, tensor_name, triv="expm"):
    r"""Adds a positive definiteness constraint to the tensor
    ``module[tensor_name]``.

    When accessing ``module[tensor_name]``, the module will return the
    parametrized version :math:`X` which will be symmetric and with
    non-negative eigenvalues

    If the tensor has more than two dimensions, the parametrization will be
    applied to the last two dimensions.

    Examples::

        >>> layer = nn.Linear(20, 20)
        >>> geotorch.positive_semidefinite(layer, "weight")
        >>> L = torch.symeig(layer.weight).eigenvalues
        >>> L[L.abs() < 1e-7] = 0.0  # Round errors
        >>> (L >= 0.0).all()
        tensor(True)

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied
        triv (str or callable): Optional.
            A map that maps :math:`\operatorname{Skew}(n)` onto the orthogonal
            matrices surjectively. This is used to optimize the Q in the eigenvalue
            decomposition. It can be one of `["expm", "cayley"]` or a custom
            callable. Default: `"expm"`
    """
    size = getattr(module, tensor_name).size()
    P.register_parametrization(module, tensor_name, PSSD(size, triv))


def positive_semidefinite_low_rank(
    module, tensor_name, rank, f="softplus", triv="expm"
):
    r"""Adds a positive definiteness constraint to the tensor
    ``module[tensor_name]``.

    When accessing ``module[tensor_name]``, the module will return the
    parametrized version :math:`X` which will be symmetric and with non-negative
    eigenvalues and at most `rank` of them non-zero.

    If the tensor has more than two dimensions, the parametrization will be
    applied to the last two dimensions.

    Examples::

        >>> layer = nn.Linear(20, 20)
        >>> geotorch.positive_semidefinite_low_rank(layer, "weight", 5)
        >>> L = torch.symeig(layer.weight).eigenvalues
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
            A map that maps :math:`\operatorname{Skew}(n)` onto the orthogonal
            matrices surjectively. This is used to optimize the Q in the eigenvalue
            decomposition. It can be one of `["expm", "cayley"]` or a custom
            callable. Default: `"expm"`
    """
    size = getattr(module, tensor_name).size()
    P.register_parametrization(module, tensor_name, PSSDLowRank(size, rank, triv))


def positive_semidefinite_fixed_rank(
    module, tensor_name, rank, f="softplus", triv="expm"
):
    r"""Adds a positive definiteness constraint to the tensor
    ``module[tensor_name]``.

    When accessing ``module[tensor_name]``, the module will return the
    parametrized version :math:`X` which will be symmetric and with non-negative
    eigenvalues and exactly `rank` of them non-zero.

    If the tensor has more than two dimensions, the parametrization will be
    applied to the last two dimensions.

    Examples::

        >>> layer = nn.Linear(20, 20)
        >>> geotorch.positive_semidefinite_fixed_rank(layer, "weight", 5)
        >>> L = torch.symeig(layer.weight).eigenvalues
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
        f (str or callable): Optional. The string `"softplus"` or a callable
            that maps real numbers onto the interval :math:`(0, \infty)`. Default: `"softplus"`
        triv (str or callable): Optional.
            A map that maps :math:`\operatorname{Skew}(n)` onto the orthogonal
            matrices surjectively. This is used to optimize the Q in the eigenvalue
            decomposition. It can be one of `["expm", "cayley"]` or a custom
            callable. Default: `"expm"`
    """
    size = getattr(module, tensor_name).size()
    P.register_parametrization(module, tensor_name, PSSDFixedRank(size, rank, f, triv))
