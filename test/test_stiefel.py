from unittest import TestCase
from copy import deepcopy
import itertools

import torch
import torch.nn as nn
import torch.nn.utils.parametrization as P

from mantorch.stiefel import Stiefel, StiefelTall
from mantorch.grassmanian import Grassmanian, GrassmanianTall

class TestStiefel(TestCase):

    def assertIsOrthogonal(self, X):
        if X.size(0) < X.size(1):
            X = X.t()
        Id = torch.eye(X.size(1))
        self.assertAlmostEqual(torch.norm(X.t() @ X - Id).item(), 0., places=3)

    def test_stiefel(self):
        self._test_orthogonality(Stiefel, StiefelTall)

    def test_grassmanian(self):
        self._test_orthogonality(Grassmanian, GrassmanianTall)

    def _test_orthogonality(self, cls, cls_tall):
        r"""Test that we may instantiate the parametrizations and
        register them in modules of several sizes. Check that the
        results are orthogonal and equal in the three cases.
        """
        sizes = [(8,1), (8,3), (8,4), (8,8),
                 (7,1), (7,3), (7,4), (7,7),
                 (1,7), (2,7), (5,7)]
        trivs = ["expm"]

        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.random.manual_seed(8888)
            for (n, k), triv in itertools.product(sizes, trivs):
                l1 = nn.Linear(n,k)
                with torch.no_grad():
                    l1.weight.zero_()
                l2 = deepcopy(l1)
                P.register_parametrization(l1, cls(size=l1.weight.size(), triv=triv), "weight")
                P.register_parametrization(l2, cls_tall(size=l2.weight.size(), triv=triv), "weight")

                # Check that the initialization of both layers is orthogonal
                for layer in [l1, l2]:
                    self.assertIsOrthogonal(layer.weight)
                self.assertIsOrthogonal(l1.weight_param.base)
                self.assertIsOrthogonal(l2.weight_param.base)

                # Make the initialization the same
                St = l1.weight_param
                X = l1.weight.t() if St.transpose else l1.weight
                l2.weight_param.base.data = X.data
                self.assertAlmostEqual(torch.norm(l1.weight - l2.weight).item(), 0., places=5)
                self.assertIsOrthogonal(l2.weight_param.base)

                ret = []
                for i, layer in enumerate([l1, l2]):
                    print(layer)
                    # Take one SGD step
                    optim = torch.optim.SGD(layer.parameters(), lr=1.)
                    ones = torch.ones(5, n)
                    loss = layer(ones).sum()
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    with P.cached(layer):
                        X1 = layer.weight
                        self.assertIsOrthogonal(X1)
                        loss = layer(ones).sum()
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    #print(layer.weight_param.orig_param.orig_param.orig_param.orig)
                    #print(layer.weight_param.orig_param.orig_param.orig)
                    #print(layer.weight_param.orig_param.orig)
                    #A = layer.weight_param.orig_param.orig
                    #W = layer.weight_param.orig
                    #print(W)
                    #self.assertIsOrthogonal(W)
                    #print(layer.weight)

                    X2 = layer.weight
                    self.assertIsOrthogonal(X2)
                    ret.append((X1, X2))

                def transpose(l):
                   return map(list, zip(*l))

                self.assertAlmostEqual(torch.norm(ret[0][0] - ret[1][0]).item(), 0., places=3)
                self.assertAlmostEqual(torch.norm(ret[0][1] - ret[1][1]).item(), 0., places=2)

