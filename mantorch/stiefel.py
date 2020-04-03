import torch
import torch.nn as nn
import torch.nn.utils.parametrization as P

from .manifold import Manifold, Fibration
from .so import SO, uniform_init_, cayley_map
from .linalg.expm import expm


class Stiefel(Fibration):
    r"""
    Implement everything as the fibration SO(n) -> St(n,k)
    """

    def __init__(self, size, triv="expm"):
        super().__init__(dimensions=2, size=size,
                         total_space=SO(size=Stiefel.size_so(size), triv=triv, lower=True))
        self.triv = triv

    @staticmethod
    def size_so(size):
        size_so = list(size)
        size_so[-1] = size_so[-2] = max(size[-1], size[-2])
        return tuple(size_so)

    def embedding(self, A):
        size_z = self.tensorial_size + (self.n, self.n - self.k)
        return torch.cat([A, A.new_zeros(*size_z)], dim=-1)

    def fibration(self, X):
        return X[..., :, :self.k]

    def uniform_init_(self):
        self.total_space.uniform_init_()

    def extra_repr(self):
        return super().extra_repr() + ", triv={}".format(self.triv)


def stable_qr(X):
    # Make the QR decomposition unique provided X is non-singular
    # so that no subgradients are needed
    # This should be done with QR with pivoting...
    Q, R = torch.qr(X)
    d = R.diagonal(dim1=-2, dim2=-1).sign()
    return Q * d.unsqueeze(-2).expand_as(Q),\
           R * d.unsqueeze(-1).expand_as(R)


def non_singular_(X):
    # This should be done with QR with pivoting...
    with torch.no_grad():
        n, k = X.size()[-2:]
        eps = k*1e-8
        # If it's close to zero, we give it a wiggle
        small = X.norm(dim=(-2, -1)) < eps
        if small.any():
            if X.ndimension() == 2:
                X[:k] += eps*torch.eye(k,k)
            else:
                size_e = X.size()[:-2] + (k, k)
                eye = (eps * torch.eye(k)).expand(*size_e)
                small = small.unsqueeze_(-1).unsqueeze_(-1).float().expand(*size_e)
                X[..., :k, :k] += small * eye
        return X


class StiefelTall(Manifold):
    """
    Implements St(n,k), 1 <= k <= n/2
    """
    trivializations = {"expm": expm,
                       "cayley": cayley_map}

    def __init__(self, size, triv="expm"):
        super().__init__(dimensions=2, size=size)
        if triv not in StiefelTall.trivializations.keys() and not callable(triv):
            raise ValueError("Argument triv was not recognized and is "
                             "not callable. Should be one of {}. Found {}"
                             .format(list(StiefelTall.trivializations.keys()), triv))

        if callable(triv):
            self.triv = triv
        else:
            self.triv = StiefelTall.trivializations[triv]

        size_z = size[:-2] + (self.k, self.k)
        self.register_parameter("fibr_aux", nn.Parameter(torch.zeros(*size_z)))
        self.uniform_init_()

    def trivialization(self, X, B):
        # We compute the exponential map as per Edelman
        # This also works for the Cayley
        # Note that this Cayley map is not the same as that of Wen & Yin

        # Equivalent to (in the paper):
        # Id = torch.eye(n)
        # IBBt = Id - B @ B.t()
        # delta = B @ A + IBBt @ X
        # Q, R = torch.qr(IBBt @ delta)
        if torch.is_grad_enabled():
            non_singular_(X)
        Q, R = stable_qr(X - B @ (B.transpose(-2, -1) @ X))
        # Form
        # A \in Skew(k)
        # Atilde = [[A, -R.t()],
        #           [R,  0    ]] \in Skew(2k)
        A = self.fibr_aux.tril(-1)
        z_size = self.tensorial_size + (2*self.k, self.k)
        Atilde = torch.cat([torch.cat([A, R], dim=-2), X.new_zeros(*z_size)], dim=-1)
        Atilde = Atilde - Atilde.transpose(-2, -1)

        BQ = torch.cat([B, Q], dim=-1)
        MN = expm(Atilde)[..., :, :self.k]
        return BQ @ MN

    def uniform_init_(self):
        with torch.no_grad():
            uniform_init_(self.base)
            if self.is_registered():
                self.original().zero_()
            self.fibr_aux.zero_()

    def extra_repr(self):
        return super().extra_repr() + ", triv={}".format(self.triv.__name__)
