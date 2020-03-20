import torch

from .stiefel import Stiefel, StiefelTall

def echelon(A):
    """ Puts a tall matrix A in column-echelon form """
    k = A.size(-1)
    A = A.t()
    return torch.solve(A, A[:, :k])[0].t()

def echelon_square(A, k):
    """ Puts the first k columns of a square matrix A in column-echelon form """
    n = A.size(0)
    A = A.t()
    C = torch.eye(n, device=A.device, dtype=A.dtype)
    C[:k, :k] = A[:k, :k]
    return torch.solve(A, C)[0].t()


class Grassmanian(Stiefel):
    r"""
    Implement everything as the fibration St(n,k) -> Gr(n,k)
    G(n,k) \iso St(n,k) / O(k) \iso O(n) / (O(k) x O(n-k))
    """
    def init(self, t):
        super().init(t)
        k = self.k
        with torch.no_grad():
            self.base = echelon_square(self.base)
            # Correct precission errors
            self.base[:k, :k] = torch.eye(k, device=base.device, dtype=base.dtype)

    def trivialization(self, x, base):
        # Zero-out the upper k x k square
        Z = x.new_zeros(self.k, self.k)
        x = torch.cat([Z, x[self.k:]])
        return super().trivializarion(x, base)


class GrassmanianTall(StiefelTall):

    def init(self, t):
        super().init(t)
        del self._parameters["fibr_aux"]
        k = self.k
        self.register_buffer("fibr_aux", t.new_zeros(k, k))
        with torch.no_grad():
            self.base = echelon(self.base)
            # Correct precission errors
            self.base[:k, :k] = torch.eye(k, device=base.device, dtype=base.dtype)


