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
        return torch.cat([A, A.new_zeros(self.n, self.n-self.k)], dim=1)

    def fibration(self, X):
        return X[:, :self.k]

    def uniform_init_(self, original_tensor):
        self.total_space.uniform_init_(original_tensor)

    def extra_repr(self):
        return super().extra_repr() + ", triv={}".format(self.triv)


def stable_qr(X):
    # Make the QR decomposition unique provided X is non-singular
    # so that no subgradients are needed
    # This should be implemented in main pytorch or smth...
    Q, R = torch.qr(X)
    d = R.diagonal().sign()
    return Q * d.unsqueeze(-2).expand_as(Q),\
           R * d.unsqueeze(-1).expand_as(R)


def non_singular_(X):
    with torch.no_grad():
        k = X.size(-1)
        eps = 1e-7
        # If it's close to zero, we give it a wiggle
        small = torch.norm(X) < k * k * eps
        if small.any():
            if X.ndimension() == 2:
                X[:k] += torch.normal(0., eps, (k, k)).clamp_(-5.*eps, 5.*eps)
            else:
                X[small][:k] += torch.normal(0., eps, X[small][:k].size()).clamp_(-5.*eps, 5.*eps)



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

        self.register_parameter("fibr_aux", nn.Parameter(torch.zeros(self.k, self.k)))
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
        non_singular_(X)
        Q, R = stable_qr(X - B @ (B.transpose(-2, -1) @ X))
        # Form
        # A \in Skew(k)
        # Atilde = [[A, -R.t()],
        #           [R,  0    ]] \in Skew(2k)
        A = self.fibr_aux.tril(-1)
        Atilde = torch.cat([torch.cat([A, R]),torch.zeros(2*self.k, self.k)], dim=1)
        Atilde = Atilde - Atilde.transpose(-2, -1)

        BQ = torch.cat([B, Q], dim=1)
        MN = expm(Atilde)[:, :self.k]
        return BQ @ MN

    def uniform_init_(self):
        with torch.no_grad():
            uniform_init_(self.base)
            for p in self.parameters():
                p.zero_()

    def extra_repr(self):
        inv_map = {v: k for k, v in StiefelTall.trivializations.items()}
        name = inv_map.get(self.triv, "custom")
        return super().extra_repr() + ", triv={}".format(name)
