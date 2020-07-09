![build](https://github.com/Lezcano/geotorch/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/Lezcano/geotorch/branch/master/graph/badge.svg?token=1AKM2EQ7RT)](https://codecov.io/gh/Lezcano/geotorch)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

# GeoTorch
> A library for constrained optimization and manifold optimization for deep learning in Pytorch

## Overview

GeoTorch provides a way to perform constrained optimization and optimization on manifolds in a way that is compatible with all the layers, models, and optimizers  implemented in Pytorch with no modifications.

```python
from torch import nn
import geotorch

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(64, 128)
        self.cnn = nn.Conv2d(16, 32, 3)
        # Make the linear layer into a low rank layer with rank at most 10
        geotorch.lowrank(self.linear, "weight", rank=10)
        # Also works on tensors. Makes every kernel orthogonal
        geotorch.orthogonal(self.cnn, "weight")

    def forward(self, x):
        # self.linear is lowrank and every kernel of the CNN is also orthogonal
        ...

# Use the model as you'd normally do, everything works as in a non-parametrized model
model = Model()

# Use your optimizer of choice. Any optimizer works out of the box on any manifold
optim = torch.optim.Adam(model.parameters(), lr=lr)
```

## Manifolds

GeoTorch currently supports the following manifolds:
- `Rn(n)`: Rⁿ. Unrestricted optimization
- `Sym(n)`: Vector space of symmetric matrices
- `Skew(n)`: Vector space of skew-symmetric matrices
- `Sphere(n)`: Sphere in Rⁿ. It is Sⁿ⁻¹ = { x ∈ Rⁿ | ||x|| = 1 }
- `SO(n)`: Manifold of n×n orthogonal matrices
- `Stiefel(n,k)`: Manifold of n×k matrices with orthonormal columns
- `Grassmannian(n,k)`: Manifold of k-dimensional subspaces in Rⁿ
- `LowRank(n,k,r)`: Variety of n×k matrices of rank r or less

Every manifold of dimension `(n, k)`can be applied to tensors of shape `(*, n, k)`, so we also get efficient parallel implementations of product manifolds such as
- `ObliqueManifold(n,k)`: Matrix with unit length columns, Sⁿ⁻¹ × ...ᵏ⁾ × Sⁿ⁻¹

It also implements the following constructions:
- `Manifold`: Manifold that supports Riemannian Gradient Descent and trivializations
- `Fibration`: Fibred space π : E → M, constructed from a `Manifold` E, a submersion π and local sections of dπ. For example the `Stiefel` manifold π : SO(n) → St(n, k) or the `Grassmannian` π : St(n, k) → Gr(n, k)
- `ProductManifold`: M₁ × ... × Mₖ

The following manifolds will be implemented soon. Open an issue if you really need some of these or you would like other manifolds to be implemented!
- `AlmostOrthogonal(n,k,t)`: Manifold of n×k matrices with singular values in the interval (1-t, 1+t)
- `FixedRank(n,k,r)`: Manifold of n×k matrices of rank r
- `PD(n)`: Cone of n×n symmetric positive definite matrices
- `PSD(n)`: Cone of n×n symmetric positive semi-definite matrices
- `PSDLowRank(n,k)`: Variety of n×n symmetric positive semi-definite matrices of rank k or less
- `PSDFixedRank(n,k)`: Manifold of n×n symmetric positive semi-definite matrices of rank k
- `SymF(n, f)`: Symmetric positive definite matrices with eigenvalues in the image of a map `f`. If the map `f` is an embedding, this is a manifold

## Sharing Weights

If one wants to use a parametrized tensor in different places in their model, or uses one parametrized layer many times, for example in an RNN, it is recommended to wrap the forward pass as follows to avoid each parametrization to be computed many times:
```python
with geotorch.parametrize.cached():
    logits = model(input_)
```

Of course, this `with` statement may be used simply inside the forward function where the parametrized layer is used several times.


## Beyond Optimization: Normalizing Flows

As every manifold in GeoTorch is, at its core, a map from a flat space into a manifold, the tools implemented here also serve as a building block in normalizing flows. Using a factorized manifold such as `LowRank` it is direct to compute the determinant of the transformation it defines, as we have direct access to the signular values of the layer.

## Bibliography

Please cite the following work if you found GeoTorch useful. This paper exposes a simplified mathematical explanation of part of the inner-workings of GeoTorch
```
@inproceedings{lezcano2019trivializations,
    title = {Trivializations for gradient-based optimization on manifolds},
    author = {Lezcano-Casado, Mario},
    booktitle={Advances in Neural Information Processing Systems, NeurIPS},
    pages = {9154--9164},
    year = {2019},
}
```
