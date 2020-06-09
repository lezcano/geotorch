from .constructions import AbstractManifold


class Skew(AbstractManifold):
    def __init__(self, size, lower=True):
        r"""
        Vector space of skew-symmetric matrices, parametrized in terms of the upper or lower triangular
        part of a matrix.

        Args:
            size (torch.size): Size of the tensor to be applied to
            lower (bool): Optional. Uses the lower triangular part of the matrix to parametrize
                the matrix. Default: `True`
        """
        super().__init__(dimensions=2, size=size)
        if self.n != self.k:
            raise ValueError(
                "The Skew parametrization can just be applied to square matrices. "
                "Got a tensor of size {}".format(
                    self.dim[::-1] if self.transpose else self.dim
                )
            )
        self.lower = lower

    def forward(self, X):
        if self.lower:
            X = X.tril(-1)
        else:
            X = X.triu(1)
        return X - X.transpose(-2, -1)

    def extra_repr(self):
        return "n={}".format(self.n)
