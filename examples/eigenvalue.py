"""
In this program we show how to use GeoTorch to compute the maximum eigenvalue
of a symmetric matrix via the Rayleigh quotient, restricting the optimisation
problem to the Sphere
"""
import torch
from torch import nn
import geotorch

N = 1000      # matrix size
LR = 1.0 / N  # step-size.
# Obs. If the distribution of the matrix is changed, this parameter should be tuned


class Model(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.x = nn.Parameter(torch.rand(n))
        geotorch.sphere(self, "x")
        self.parametrizations.x.uniform_init_()

    def forward(self, A):
        x = self.x
        return x.T @ A @ x


# Generate matrix
A = torch.rand(N, N)  # Uniform on [0, 1)
A = 0.5 * (A + A.T)

# Compare against diagonalization
max_eigenvalue = torch.symeig(A).eigenvalues.max()
print("Max eigenvalue: {:10.5f}".format(max_eigenvalue))

# Instantiate model and optimiser
model = Model(N)
optim = torch.optim.SGD(model.parameters(), lr=LR)

eigenvalue = float("inf")
i = 0
while (eigenvalue - max_eigenvalue).abs() > 1e-3:
    eigenvalue = model(A)

    optim.zero_grad()
    (-eigenvalue).backward()
    optim.step()
    print("{:2}. Best guess: {:10.5f}".format(i, eigenvalue.item()))
    i += 1

print("Final error {:.5f}".format((eigenvalue - max_eigenvalue).abs()))
