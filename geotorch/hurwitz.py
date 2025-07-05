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
        also called Hurwitz matrices.

        :math`A` is Hurwitz with prescribed decay rate :math:`\alpha`
        if and only if it can be written as
        ..math::
            A = P^{-1}(-\frac{Q}{2} +S) - \alpha \mathrm{I}_n
        with :math:`P \in \operatorname{PSD}(n), Q \in \operatorname{PSD}(n)` and :math: `S \in \operatorname{Skew}(n)`

        Args:
            size (torch.size): Size of the tensor to be parametrized
            alpha (float): the upper bound on the matrix's eigenvalues real part
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
        self.register_buffer('In', torch.eye(n).expand(*self.tensorial_size, n, n))

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
        size =  tensorial_size + (n, n)
        return PSD(size, triv=triv), PSD(size, triv=triv), Skew(size)
    

    def submersion(self, Q, P, S):
            return P @ (-0.5 * Q + S) - self.alpha * self.In

    def forward(self, X1, X2, X3):
        Q, P, S = super().forward([X1, X2, X3])
        return self.submersion(Q, P, S)


    def submersion_inv(self, A: torch.Tensor, check_in_manifold=True, rho=1, tol = 1e-4):
        r"""
            Args: 
                A (torch.Tensor): a square and Hurwitz matrix with eigenvalues lower than -alpha
                check_in_manifold (bool): if set to True we raise an error if the matrix A is not in
                the manifold

                rho (float): a scaling parameter for the matrix Q = -rho I

                tol (float): a small margin to ensure P is not close to singularity, can be increased if needed

        """
        if check_in_manifold and not self.in_manifold_eigen(A):
            print(f"Enforced alpha = {self.alpha} and given alpha = {get_lyap_exp(A)}")
            raise InManifoldError(get_lyap_exp(A), self.alpha)
        
        with torch.no_grad():
            A_shifted = A + (self.alpha - tol) * self.In 
            A_shifted_T = A_shifted.mT.contiguous()

            # Batched Kronecker product using vmap for cleaner implementation
            def _single_kron_sum(A_single):
                I_single = torch.eye(self.n, device=A.device, dtype=A.dtype)
                return torch.kron(I_single, A_single) + torch.kron(A_single, I_single)
            
            flat_batch_size = torch.prod(torch.tensor(self.tensorial_size)) if self.tensorial_size else 1
            A_flat = A_shifted_T.reshape(flat_batch_size, self.n, self.n)

            M_flat = torch.vmap(_single_kron_sum)(A_flat)
            M = M_flat.reshape(*self.tensorial_size, self.n*self.n, self.n*self.n)

            Q = (rho * self.In)              
            flat_Q = Q.flatten(-2, -1)


            vec_P = torch.linalg.solve(M, -flat_Q.unsqueeze(-1)).squeeze(-1)
            P = vec_P.view(*self.tensorial_size, self.n, self.n)

            residual = torch.norm(A.mT @ P + P @ A + 2 * (self.alpha - tol) * P + Q)
            if residual >=tol:
                raise ValueError(f"Lyapunov equation ill-conditioned solve failed. \n Residual norm: {torch.norm(residual):.2e}")

            S = P @ (A + (self.alpha-tol) * self.In) + 0.5 * Q 

            P_inv = torch.inverse(P)

            return Q, P_inv, S

        
    def right_inverse(self, A, check_in_manifold=True):
        Q, P, S = self.submersion_inv(A, check_in_manifold)
        X1, X2, X3 = super().right_inverse([Q, P, S])
        return X1, X2, X3
    
    def in_manifold_eigen(self, A, eps=1e-6):
        r"""
        Check that all eigenvalues have real part lower than -alpha
        """
        eig = torch.linalg.eigvals(A)
        reig = torch.real(eig)
        return (reig <= -self.alpha + eps).all().item()
    

    def sample(self, init_=torch.nn.init.xavier_normal_):
        r"""
        Returns a randomly sampled Hurwitz (:math:`\alpha`-stable) matrix on the manifold as:

        .. math::
            A = P^{-1}(-Q/2 + S) - \alpha I
        with :math:`P^{-1}_{i,j}, Q_{i,j}, S_{i,j} \sim \texttt{init_}`

        The output of this method can be used to initialize a parametrized tensor as::

            >>> layer = nn.Linear(20, 20)
            >>> M = Hurwitz(layer.weight.size(), alpha=0.5)
            >>> geotorch.register_parametrization(layer, "weight", M)
            >>> layer.weight = M.sample()

        Args:
            init_ (callable): Initialization method for random matrices.
                            Default: ``torch.nn.init.xavier_normal_``
        """
        with torch.no_grad():

            mani_psd = PSD((self.n, self.n))
            mani_skew = Skew((self.n, self.n))

            P = mani_psd.sample(init_)
            Q = mani_psd.sample(init_)
            S = mani_skew.sample(init_)
            
            A = self.submersion(Q, P, S)
            return A


    def extra_repr(self) -> str:
        print(self.alpha)
        return _extra_repr(
            n=self.n,
            alpha=self.alpha
            )