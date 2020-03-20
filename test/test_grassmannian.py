from unittest import TestCase

import torch

from mantorch.stiefel import Stiefel, StiefelTall
from mantorch.grassmanian import echelon, echelon_square

class TestOrthogonal(TestCase):

    def test_echelon_square_(self):
        r"""Test that the functions `echelon_` and `echelon_square_`
        do not modify the column subspaces
        """
        # Test some corner-cases
        test = [(8,1), (8,3), (7,6), (8,8)]

        def proj(T):
            """ Project onto the columns space """
            return T @ torch.solve(T.t(), T.t() @ T)[0]

        # Make the test deterministic
        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.random.manual_seed(8888)
            for n, k in test:
                St = StiefelTall()
                l = torch.nn.Linear(n,k)
                l.register_parametrization(St, "weight")
                E = echelon(St.base)
                # Check that it is indeed in Echelon form
                self.assertTrue(torch.norm(E[:k, :k] - torch.eye(k)) < 1e-5)

                P1 = proj(St.base)
                P2 = proj(E)
                # If the two projections are the the same,
                # we have not changed the column space
                self.assertTrue(torch.norm(P1 - P2) < 1e-5)

                St = Stiefel()
                l = torch.nn.Linear(n,k)
                l.register_parametrization(St, "weight")
                E = echelon_square(St.base, k)
                self.assertTrue(torch.norm(E[:k, :k] - torch.eye(k)) < 1e-5)

                P1 = proj(St.base)
                P2 = proj(E)
                self.assertTrue(torch.norm(P1 - P2) < 1e-5)


