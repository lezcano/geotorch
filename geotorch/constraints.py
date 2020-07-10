import geotorch.parametrize as P
from geotorch.sphere import Sphere
from geotorch.stiefel import Stiefel, StiefelTall
from geotorch.grassmannian import Grassmannian, GrassmannianTall
from geotorch.lowrank import LowRank
from geotorch.symmetric import Symmetric
from geotorch.skew import Skew


def symmetric(module, tensor_name, lower=True):
    r""" Adds a symmetric parametrization to the matrix ``module[tensor_name]``

    When accessing ``module[tensor_name]``, the module will return the parametrized
    version :math:`X` so that :math:``X^\intercal = X``.

    If the tensor has more than two dimensions, the symmetric parametrization will be
    added in the last two dimensions.

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
    r""" Adds a skew-symmetric parametrization to the matrix ``module[tensor_name]``

    When accessing ``module[tensor_name]``, the module will return the parametrized
    version :math:`X` so that :math:``X^\intercal = -X``.

    If the tensor has more than two dimensions, the skew parametrization will be
    added in the last two dimensions.

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
    r""" Adds a spherical parametrization to the vector (or tensor)
    ``module[tensor_name]``

    When accessing ``module[tensor_name]``, the module will return the parametrized
    version :math:`v` so that :math:``\lVert v \rVert = 1``.

    If the tensor has more than one dimension, the spherical parametrization will be
    added in the last dimension.

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
    r""" Adds an orthogonal parametrization to the tensor ``module[tensor_name]``

    When accessing ``module[tensor_name]``, the module will return the
    parametrized version :math:`X` so that :math:`X^\mathcal X = \operatorname{Id}`

    If the tensor has more than two dimensions, the orthogonal parametrization will be
    added in the last two dimensions.

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
    cls = StiefelTall if n > 4 * k else Stiefel
    P.register_parametrization(module, tensor_name, cls(size, triv))


def grassmannian(module, tensor_name, triv="expm"):
    r""" Adds an parametrization to the tensor ``module[tensor_name]`` so that the
    result represents a subspace. If the initial matrix was of size :math:`n \times k`
    the parametrized matrix will represent a subspace of dimension :math:`k` of
    :math:`\mathbb{R}^n`.

    When accessing ``module[tensor_name]``, the module will return the parametrized
    version :math:`X` so that :math:`X` represents :math:`k` orthogonal vectors of
    :math:`\mathbb{R}^n` that span the subspace. That is, the resulting matrix will
    be orthogonal, :math:`X^\mathcal X = \operatorname{Id}`.

    If the tensor has more than two dimensions, the parametrization will be
    added in the last two dimensions.

    .. note::

        Even though this space resembles that generated by :func:`~geotorch.orthogonal`,
        it is actually a subspace of that, as every subspace can be represented by many
        different basis of vectors that span it.

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


def lowrank(module, tensor_name, rank):
    r""" Adds a low rank parametrization to the tensor ``module[tensor_name]``

    When accessing ``module[tensor_name]``, the module will return the
    parametrized version :math:`X` will have rank at most ``rank``.

    If the tensor has more than two dimensions, the parametrization will be
    added in the last two dimensions.

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
            on which the parametrization will be applied
        rank (int): Rank of the matrices.
            It has to be less than the minimum of the last two dimensions of the
            tensor
    """
    size = getattr(module, tensor_name).size()
    P.register_parametrization(module, tensor_name, LowRank(size, rank))
