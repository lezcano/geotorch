from .constructions import Manifold


class Rn(Manifold):
    def __init__(self, size):
        r"""
        Vector space of unconstrained vectors.

        Args:
            size (torch.size): Size of the tensor to be applied to
        """
        super().__init__(dimensions=1, size=size)
        self.base.zero_()

    def trivialization(self, X):
        # We implement it with a base to be able to use it within a fibration
        return X + self.base
