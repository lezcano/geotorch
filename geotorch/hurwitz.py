import torch
from .psd import PSD
from .skew import Skew
from .product import ProductManifold
from .exceptions import VectorError, NonSquareError
from .utils import _extra_repr
from .exceptions import (
    VectorError,
    NonSquareError,
    InManifoldError,
)

def get_lyap_exp(A):
    return -torch.max(torch.real(torch.linalg.eigvals(A)))

class Hurwitz(ProductManifold):

    def __init__(self, size, alpha: float = 0.0, triv="expm"):
        r"""
        Manifold of matrices with eigenvalues with negative real parts,
        also called Hurwitz matrices. They represend linear stable linear dynamical systems.

        We have the following result : A is Hurwitz with prescribed decay rate \alpha
        if and only if it can be written as
        :math:`A = P^{-1}(-\frac{Q}{2} +S) - \alpha I_n`
        with $P > 0, Q > 0$ and ^S^skew-symmetric
        Args:
            size (torch.size): Size of the tensor to be parametrized
            alpha (float): the upper bound on the matrix's eigenvalues real part
                :math: `\min_{\lambda \in Sp(A)}  \Re(\lambda) \leq -\alpha`
            triv (str or callable): Optional.
                A map that maps skew-symmetric matrices onto the orthogonal matrices
                surjectively. This is used to optimize the :math:`Q` in the eigenvalue
                decomposition. It can be one of ``["expm", "cayley"]`` or a custom
                callable. Default: ``"expm"``
        """

        assert alpha >=0, ValueError(f"alpha must be positive found {alpha}")

        n, tensorial_size = Hurwitz.parse_size(size)
        super().__init__(Hurwitz.manifolds(n, tensorial_size, triv))
        self.n = n
        self.tensorial_size = tensorial_size
        self.alpha = alpha
        self.register_buffer('In', torch.eye(n))

    @classmethod
    def parse_size(cls, size):
        if len(size) < 2:
            raise VectorError(cls.__name__, size)
        n, k = size[-2:]
        tensorial_size = size[:-2]
        if n != k:
            raise NonSquareError(cls.__name__, size)
        return n, tensorial_size
    
    @staticmethod
    def manifolds(n, tensorial_size, triv):
        size = tensorial_size + (n, n)
        return PSD(size, triv=triv), PSD(size, triv=triv), Skew(size)
    

    def submersion(self, Q, P, S):
            return P @ (-0.5 * Q + S) - self.alpha * self.In

    def forward(self, X1, X2, X3):
        Q, P, S = super().forward([X1, X2, X3])
        return self.submersion(Q, P, S)


    def submersion_inv(self, A: torch.Tensor, check_in_manifold=True, rho=1, tol = 1e-5):
        r"""
            Args: 
                A (torch.Tensor): a square and Hurwitz matrix with eigenvalues lower than -alpha
                check_in_manifold (bool): if set to True we raise an error if the matrix A is not in
                the manifold

                rho (float): a scaling parameter for the matrix Q = -rho I

        """
        if check_in_manifold and not self.in_manifold_eigen(A):
            print(f"Enforced alpha = {self.alpha} and given alpha = {get_lyap_exp(A)}")
            raise InManifoldError(get_lyap_exp(A), self.alpha)
        
        with torch.no_grad():
            A_shifted = A + (self.alpha - tol) * self.In 
            A_shifted_T = A_shifted.mT.contiguous()
            print(A_shifted.dtype)
            M = torch.kron(self.In,  A_shifted_T) + torch.kron(A_shifted_T, self.In)
            Q = (rho * self.In).unsqueeze(0).repeat(*self.tensorial_size, 1, 1)
            flat_Q = Q.flatten(-2, -1)
            vec_P = torch.linalg.solve(M, -flat_Q)
            P = vec_P.view(*self.tensorial_size, self.n, self.n)
            residual = A.mT @ P + P @ A + 2 * (self.alpha - tol)* P + Q
            print(f"A eigenvalues : {torch.linalg.eigvals(A)}")
            print(f"Alpha : {self.alpha}")
            print(f"Residual norm: {torch.norm(residual):.2e}")

            assert torch.allclose(A.mT @ P + P @ A + 2* (self.alpha - tol)*P, -Q, atol=1e-5)

            S = P @ (A + (self.alpha-tol) * self.In) + 0.5 * Q 

            print(S)
            print(S.mT)
            P_inv = torch.inverse(P)
            A_rec = P_inv @ (-Q/2 + S) -(self.alpha-tol) * self.In
            print(f"Distance between As at the end of computation of ri {torch.dist(A_rec, A)}")
            print(f"Current alpha in submersion_inv {(self.alpha-tol)}")
            return Q, P_inv, S

        
    def right_inverse(self, A, check_in_manifold=True):
        Q, P, S = self.submersion_inv(A, check_in_manifold)
        X1, X2, X3 = super().right_inverse([Q, P, S])
        return X1, X2, X3
    
    def in_manifold_eigen(self, A, eps=1e-6):
        r"""
        Check that all eigenvalues have real part lower than -alpha
        """
        if A.size()[:-2] != self.tensorial_size:  # check dimensions
            print(A.size(), self.tensorial_size)
            return False
        else:
            eig = torch.linalg.eigvals(A)
            reig = torch.real(eig)
            return (reig <= -self.alpha + eps).all().item()
        

    def sample(self, init_=torch.nn.init.xavier_normal_, gamma=1.0):
        r"""
        Returns a randomly sampled Hurwitz (α-stable) matrix on the manifold as:

            A = P^{-1}(-Q/2 + S) - αI,  with Q = γP

        The output of this method can be used to initialize a parametrized tensor as::

            >>> layer = nn.Linear(20, 20)
            >>> M = Hurwitz(layer.weight.size(), alpha=0.5)
            >>> geotorch.register_parametrization(layer, "weight", M)
            >>> layer.weight = M.sample()

        Args:
            init_ (callable): Initialization method for random matrices.
                            Default: ``torch.nn.init.xavier_normal_``
            gamma (float): Scaling parameter for Q = gamma * P.
                        Default: ``1.0``
        """
        with torch.no_grad():

            X_p = torch.empty(*(self.tensorial_size + (self.n, self.n)))
            init_(X_p)
            P = X_p @ X_p.transpose(-2, -1) + 1e-3 * self.In  

            X_q = torch.empty(*(self.tensorial_size + (self.n, self.n)))
            init_(X_q)
            Q = X_q @ X_q.mT + 1e-3 * self.In

            
            X_s = torch.empty_like(X_p)
            init_(X_s)
            S = X_s - X_s.transpose(-2, -1)

            
            P_inv = torch.inverse(P)
            A = P_inv @ (-0.5 * Q + S) - self.alpha * self.In

            return A


    def extra_repr(self) -> str:
        print(self.alpha)
        return _extra_repr(
            n=self.n,
            alpha=self.alpha,
            tensorial_size=self.tensorial_size,
        )