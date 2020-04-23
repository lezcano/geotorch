from .manifold import Manifold


class Rn(Manifold):
    def __init__(self, size):
        super().__init__(dimensions=1, size=size)
        self.base.zero_()

    def trivialization(self, X, B):
        return X + B
