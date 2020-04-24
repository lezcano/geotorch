![](https://github.com/lezcano/geotorch/workflows/Build/badge.svg)
[![codecov](https://codecov.io/gh/Lezcano/geotorch/branch/master/graph/badge.svg?token=1AKM2EQ7RT)](https://codecov.io/gh/Lezcano/geotorch)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# GeoTorch
> A library for constrained optimization and manifold optimization for deep learning in Pytorch

## Overview

GeoTorch provides a way to perform constrained optimization and optimization on manifolds in a non-intrusive way. It is compatible with any layer and model implemented in Pytorch and any optimizer with no modifications.

```python
from torch import nn
import geotorch

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(4, 5)
        self.cnn = nn.Conv(16, 32, 3)
        # Make the linear layer into an orthogonal layer
        geotorch.orthogonal(self.linear, "weight")
        # Also works on tensors. Makes every kernel orthogonal
        geotorch.orthogonal(self.cnn, "weight")

    def forward(self, x):
        # Here self.linear.weight is orthogonal
        # Every kernel of the cnn is also orthogonal
        ...

# Use the model as you'd normally do, everything works as in a non-parametrized model
model = Model()

# Use your optimizer of choice. Any optimizer works out of the box on any manifold
optim = torch.optim.Adam(model.parameters(), lr=lr)
```

## Manifolds

GeoTorch currently supports the following manifolds:
- `Sphere`
- `SO`: Manifold of orthogonal square matrices
- `Stiefel`: Manifold of matrices with orthonormal columns
- `Grassmannian`: Manifold of k-subspaces in Rⁿ
- `Low-Rank`: Variety of matrices n x k of rank r or less
- `Skew`: Vector space of skew-symmetric matrices
- `Sym`: Vector space of symmetric matrices
- `Rn`: Unrestricted optimisation

Every manifold of dimension `(m, n)`can be applied to tensors of shape `(*, m, n)`, so we also get efficient parallel implementations of product manifolds such as
- `Oblique Manifold`: Sⁿ × ...ᵐ⁾ × Sⁿ

Furthermore, it implements the following constructions:
- `Manifold`: Manifold that supports Riemannian Gradient Descent and trivializations
- `Fibration`: Fibred space π : E → M, constructed from a `Manifold` E, a submersion π and local sections of dπ. Think the `Stiefel` manifold π : SO(n) → St(n, k) or the `Grassmannian` π : St(n, k) → Gr(n, k)
- `ProductManifold`: M₁ × ... × Mₖ

## Bibliography

Please cite the following work if you found GeoTorch useful. In this paper one may find a simplified mathematical explanation of some basic version of GeoTorch
```
@inproceedings{lezcano2019trivializations,
    title = {Trivializations for gradient-based optimization on manifolds},
    author = {Lezcano-Casado, Mario},
    booktitle={Advances in Neural Information Processing Systems, NeurIPS},
    pages = {9154--9164},
    year = {2019},
}
```
