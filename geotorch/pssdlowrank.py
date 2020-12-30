import torch
from .symmetric import SymF


class PSSDLowRank(SymF):
    def __init__(self, size, rank, triv="expm"):
        r"""
        Variety of the symmetric positive semidefinite matrices of rank
        at most :math:`r`.

        Args:
            size (torch.size): Size of the tensor to be parametrized
            rank (int): Rank of the matrices.
                It has to be less or equal to
                :math:`\min(\texttt{size}[-1], \texttt{size}[-2])`
            triv (str or callable): Optional.
                A map that maps skew-symmetric matrices onto the orthogonal matrices
                surjectively. This is used to optimize the :math:`Q` in the eigenvalue
                decomposition. It can be one of ``["expm", "cayley"]`` or a custom
                callable. Default: ``"expm"``
        """
        super().__init__(size, rank, f=(torch.abs, torch.abs), triv=triv)
