# Mantorch
> A library for constrained and manifold optimization for deep learning in Pytorch

## Overview

Mantorch provides a way to perform constrained optimization and optimization on manifolds in a non-intrusive way. It is compatible with any layer implemented in Pytorch and any optimizer in it with no modifications.

```python
from torch import nn
import mantorch as M

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(4, 5)
		# Make the linear layer into an orthogonal layer
		M.orthogonal(self.linear.weight)

    def forward(self, x):
        # Here self.linear.weight is orthogonal
        return self.linear(x)
```
