from unittest import TestCase
from copy import deepcopy
import itertools

import torch
import torch.nn as nn
import torch.nn.utils.parametrization as P

from mantorch.so import SO
from mantorch.stiefel import Stiefel, StiefelTall
from mantorch.grassmanian import Grassmanian, GrassmanianTall


class TestStiefel(TestCase):

    def assertIsOrthogonal(self, X):
        if X.ndimension() == 2:
            self._assertIsOrthogonal(X)
        elif X.ndimension() > 2:
            # Sample a few elements and see that they are orthogonal
            for _ in range(4):
                coords = [torch.randint(low=0, high=s, size=(1,)).item()
                           for s in X.size()[:-2]]
                coords = coords + [...]
                self._assertIsOrthogonal(X[coords])

    def _assertIsOrthogonal(self, X):
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
                 (1,7), (2,7), (1,1), (1,2)]
        trivs = ["expm"]

        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.random.manual_seed(8888)
            for (n, k), triv in itertools.product(sizes, trivs):
                for layer_init in [nn.Linear(n,k), nn.Conv2d(k, 4, 3)]:
                    print("START")
                    test_so = cls != Grassmanian and n == k
                    l1 = layer_init
                    l2 = deepcopy(l1)
                    if test_so:
                        l3 = deepcopy(l1)
                    P.register_parametrization(l1, cls(size=l1.weight.size(), triv=triv), "weight")
                    P.register_parametrization(l2, cls_tall(size=l2.weight.size(), triv=triv), "weight")
                    if test_so:
                        P.register_parametrization(l3, SO(size=l3.weight.size(), triv=triv), "weight")

                    layers = [l1, l2]
                    if test_so:
                        layers.append(l3)

                    for layer in layers:
                        with torch.no_grad():
                            layer.weight_param.uniform_init_()

                    # Check that the initialization of the layers is orthogonal
                    for layer in layers:
                        self.assertIsOrthogonal(layer.weight)
                        self.assertIsOrthogonal(layer.weight_param.base)

                    # Make the initialization the same
                    X = l1.weight.t() if l1.weight_param.transpose else l1.weight
                    for layer in layers[1:]:
                        with torch.no_grad():
                            layer.weight_param.base.copy_(X)
                        self.assertAlmostEqual(torch.norm(l1.weight - layer.weight).item(), 0., places=5)
                        self.assertIsOrthogonal(layer.weight_param.base)

                    if isinstance(l1, nn.Linear):
                        input_ = torch.rand(5, n)
                    elif isinstance(l1, nn.Conv2d):
                        # batch x in_channel x in_length x in_width
                        input_ = torch.rand(6, k, 7, 8)

                    results = []
                    for i, layer in enumerate(layers):
                        print(layer)
                        # Take one SGD step
                        optim = torch.optim.SGD(layer.parameters(), lr=1.)
                        loss = layer(input_).sum()
                        optim.zero_grad()
                        loss.backward()
                        optim.step()

                        with P.cached(layer):
                            X1 = layer.weight
                            self.assertIsOrthogonal(X1)
                            loss = layer(input_).sum()
                        optim.zero_grad()
                        loss.backward()
                        optim.step()

                        X2 = layer.weight
                        self.assertIsOrthogonal(X1)
                        results.append((X1, X2))

                    # Check pairwise equality
                    with torch.no_grad():
                        for l, m in itertools.combinations(range(len(results)), 2):
                            norm0 = torch.norm(results[l][0] - results[m][0]).item()
                            norm1 = torch.norm(results[l][1] - results[m][1]).item()
                            if isinstance(l1, nn.Linear):
                                self.assertAlmostEqual(norm0, 0., places=3)
                                self.assertAlmostEqual(norm1, 0., places=2)
                            elif isinstance(l1, nn.Conv2d):
                                # We are more permissive, as we are computing the norm
                                # of the whole batch of matrices. We are fine with it
                                # being correct on average
                                print(l1.weight.size())
                                self.assertAlmostEqual(norm0/(k*4.), 0., places=3)
                                self.assertAlmostEqual(norm1/(k*4.), 0., places=2)
