![](https://github.com/lezcano/geotorch/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/Lezcano/geotorch/branch/master/graph/badge.svg?token=1AKM2EQ7RT)](https://codecov.io/gh/Lezcano/geotorch)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# GeoTorch
> A library for constrained and manifold optimization for deep learning in Pytorch

## Overview

GeoTorch provides a way to perform constrained optimization and optimization on manifolds in a non-intrusive way. It is compatible with any layer and model implemented in Pytorch and any optimizer with no modifications.

```python
from torch import nn
import geotorch

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(4, 5)
        # Make the linear layer into an orthogonal layer
        geotorch.orthogonal(self.linear, "weight")

    def forward(self, x):
        # Here self.linear.weight is orthogonal
        return self.linear(x)

# Use the model as you'd normally do, everything works as in a non-parametrized model
model = Model()

# Use your optimizer of choice. Any optimizer works out of the box on any manifold
optim = torch.optim.Adam(model.parameters(), lr=lr)
```
