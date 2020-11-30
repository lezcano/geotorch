import torch
from torch import nn
from .utils import base, _extra_repr


class Rn(nn.Module):
    def __init__(self, size):
        r"""
        Vector space of unconstrained vectors.

        Args:
            size (torch.size): Size of the tensor to be applied to
        """
        super().__init__()
        self.n = size[-1]
        self.tensorial_size = size[:-1]
        self.register_buffer("base", torch.zeros(*size))

    @base
    def forward(self, X):
        # We implement it with a base to be able to use it within a fibered space
        return X + self.base

    def extra_repr(self):
        return _extra_repr(n=self.n, tensorial_size=self.tensorial_size)
