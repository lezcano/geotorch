from abc import abstractmethod
import math
import random
import torch
import scipy.linalg as la

from manifold import BaseManifold
from linalg.expm import expm


def cayley_map(X):
    n = X.size(0)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)
    return torch.solve(Id - X, Id + X)[0]


class SO(BaseManifold):
    trivializations = {
                       "expm": expm,
                       "cayley": cayley_map,
                      }

    def __init__(self, triv="expm"):
        super().__init__()
        if triv not in SO.trivializations.keys() and not callable(triv):
            raise ValueError("Argument triv was not recognized and is "
                             "not callable. Should be one of {}. Found {}"
                             .format(list(SO.trivializations.keys()), triv))
        if callable(triv):
            self.triv = triv
        else:
            self.triv = SO.trivializations[triv]

    @classmethod
    def apply(cls, module, name, mode="auto"):
        method = super().apply(module, name, mode)
        # Initialize the base
        base_name = method._tensor_name + "_base"
        setattr(method._module, base_name, torch.eye(method.param_orig.size(0)))
        return method

    def frame(self, x, base):
        r"""
        Parametrizes the skew-symmetric matrices in terms of the lower triangle of ``x``
        """
        if x.ndimension() != 2 or x.size(0) != x.size(1):
            raise ValueError("Expected a square matrix. Got a tensor of shape {}"
                             .format(list(x.size())))

        x = x.triu(diagonal=1)
        x = x - x.t()
        return x

    def trivialization(self, x, base):
        r"""
        Maps the skew-symmetric matrices into SO(n) surjectively
        """
        return base.mm(self.triv(x))

    @staticmethod
    def torus_init_(tensor, init_=None):
        r"""Samples the 2D input `tensor` as a block-diagonal matrix
        with blocks of the form :math:`\begin{pmatrix} 0 & b \\ -b & 0\end{pmatrix}`
        where :math:`b` is distributed according to `init_`.

        Args:
            tensor (torch.nn.Tensor): a 2-dimensional matrix
            init_: Optional. A function that takes a tensor and fills
                    it in place according to some distribution. Default:
                   :math:`\mathcal{U}(-\pi, \pi)
        """
        if tensor.ndimension() != 2 or tensor.size(0) != tensor.size(1):
            raise ValueError("Expected a square matrix. Got a tensor of shape {}"
                             .format(list(x.size())))

        if init_ is None:
            init_ = lambda t: torch.nn.init.uniform_(t, -math.pi, math.pi)

        # Non-zero elements that we are going to set on the diagonal
        n_diag = tensor.size(0) // 2
        diag = tensor.new(n_diag)
        init_(diag)

        with torch.no_grad():
            # First non-central diagonal
            diag_z = torch.zeros(tensor.size(0)-1)
            diag_z[::2] = diag
            tensor_init = torch.diag(diag_z, diagonal=1, out=tensor)
            tensor = tensor - tensor.t()
        return tensor

    @staticmethod
    def haar_init_(tensor):
        r"""Samples the 2D input `tensor` as skew-symmetric matrix such that
        :math:`\exp(tensor)` is distributed uniformly on the orthogonal matrices.

        Args:
            tensor (torch.nn.Tensor): a 2-dimensional matrix
        """
        torch.nn.init.orthogonal_(tensor)
        with torch.no_grad():
            if tensor.det() < 0.:
                # Go bijectively from O^-(n) to O^+(n) \iso SO(n)
                tensor[0] *= -1.
            tensorn = la.logm(tensor.data.cpu().numpy()).real
            tensorn = .5 * (tensorn - tensorn.T)
            tensor.copy_(torch.tensor(tensorn))
        return tensor


class Stiefel(BaseManifold):
    total_space = SO()
    trivializations = {
                       "expm": expm,
                       "cayley": cayley_map,
                      }

    def __init__(self, triv="expm"):
        super().__init__()
        if triv not in Stiefel.trivializations.keys() and not callable(triv):
            raise ValueError("Argument triv was not recognized and is "
                             "not callable. Should be one of {}. Found {}"
                             .format(list(SO.trivializations.keys()), triv))

        if callable(triv):
            self.triv = triv
        else:
            self.triv = Stiefel.trivializations[triv]

    @property
    def inverted(self):
        n, m = self.param_orig.size()
        return n < m

    @property
    def large(self):
        n, m = self.param_orig.size()
        return max(n, m) <= 2. * min(n, m)

    def frame(self, x, base):
        r"""
        Parametrizes the T_B St(n,k) in terms of the lower triangle of ``x``
        """
        if x.ndimension() != 2:
            raise ValueError("Expected a matrix. Got a {}-dimensional tensor."
                             .format(x.ndimension()))
        if self.inverted:
            x = x.t()

        if self.large:
            # TODO Check limit cases like n = 1, 2, 3 ; m = k/2 - 1, k/2, k/2 + 1
            # Compute just the projection from SO(n) / SO(k)
            n, m = x.size()
            low = x[:, :m//2].tril(-1)
            up =  x[:, :m//2 + m%2].triu(1)
            # Compute the reflection of low
            low = low.flip(-1).flip(-2)
            # S is square upper triangular
            S = torch.cat([up, low], dim=1)
            return Stiefel.total_space.frame(S, base)
        else:
            # We use the projection onto the tangent space of T_base
            # https://www.manoptjl.org/stable/manifolds/stiefel/#Manopt.project
            btx = b.t().mm(x)
            return x - base.mm(.5 * btx.t().mm(btx))

    def pi(self, x):
        n, m = self.param_orig.size()
        return x[:n, :m]

    def update_base(self):
        with torch.no_grad():
            if not self.large:
                super().update_base()
            else:
                base = self.base
                param_orig = self.param_orig
                v = self.frame(param_orig, base)
                x = Stiefel.total_space.trivialization(v, base)
                self.base.data.copy_(x)
                self.param_orig.zero_()
                # Keep the original behaviour.
                # If self.param was not generated we generate it
                if self._tensor_name not in self._module._buffers:
                    self._module._buffers[self._tensor_name] = self.pi(x)

    def trivialization(self, x, base):
        r"""
        Implements optimization on St(n,k) via the exponential map with the metric inherited from R^{n x k}
        """
        if self.large:
            return self.pi(Stiefel.total_space.trivialization(x, base))
        else:
            # We compute the exponential map
            # www.manoptjl.org/stable/manifolds/stiefel/#Base.exp
            # Eq before 2.14
            # https://arxiv.org/pdf/physics/9806030.pdf
            X1 = torch.cat([base, x], dim=1)
            bx = base.t().mm(x)
            xx = x.t().mm(x)
            Id = torch.eye(base.size(1), dtype=base.dtype, device=base.device)
            X2 = torch.cat([torch.cat([bx, -xx], dim=1), torch.cat([Id, bx], dim=1)])
            eX2 = expm(X2)
            embx = expm(-bx)
            zeros = torch.zeros_like(Id)
            X3 = torch.cat([embx, zeros])
            # Order matters
            ret = X1.mm(X2.mm(X3))
            if self.inverted:
                ret = ret.t()
            return ret

    @classmethod
    def apply(cls, module, name, mode="auto"):
        method = super().apply(module, name, mode)
        # Initialize the base
        if not method.inverted:
            n, m = method.param_orig.size()
        else:
            m, n = method.param_orig.size()
        base_name = method._tensor_name + "_base"
        if method.large:
            setattr(method._module, base_name, torch.eye(n=n))
        else:
            setattr(method._module, base_name, torch.eye(n=n, m=m))
        return method

    @staticmethod
    def torus_init_(tensor, init_=None):
        r"""Samples the 2D input `tensor` as a block-diagonal matrix
        with blocks of the form :math:`\begin{pmatrix} 0 & b \\ -b & 0\end{pmatrix}`
        where :math:`b` is distributed according to `init_`.

        Args:
            tensor (torch.nn.Tensor): a 2-dimensional matrix
            init_: Optional. A function that takes a tensor and fills
                    it in place according to some distribution. Default:
                   :math:`\mathcal{U}(-\pi, \pi)
        """
        if tensor.ndimension() != 2 or tensor.size(0) != tensor.size(1):
            raise ValueError("Expected a square matrix. Got a tensor of shape {}"
                             .format(list(x.size())))

        if init_ is None:
            init_ = lambda t: torch.nn.init.uniform_(t, -math.pi, math.pi)

        # Non-zero elements that we are going to set on the diagonal
        n, m = tensor.size()
        inverted = n < m
        with torch.no_grad():
            if inverted:
                tensor = tensor.t()
                n, m = m, n
            n_diag = n // 2
            diag = tensor.new(n_diag)
            init_(diag)
            # First non-central diagonal
            diag_z = torch.zeros(tensor.size(0)-1)
            diag_z[::2] = diag
            # Make diag_z2 longer than diag_z2
            diag_z1, diag_z2 = torch.split(diag_z,
                                           [len(diag_z) // 2,
                                            len(diag_z) // 2 + (len(diag_z) % 2)])
            diag_z2 = diag_z2.flip(0)
            if len(diag_z2) > len(diag_z1):
                diag_z1 = torch.cat([diag_z1, diag_z1.new_zeros(1)])

            tensor.data.zero_()
            tensor[:n_diag+1, :n_diag+1] = torch.diag(diag_z1, diagonal=1) +\
                                           torch.diag(diag_z2, diagonal=-1)
            if inverted:
                tensor = tensor.t()
        return tensor
