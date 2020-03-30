from .manifold import AbstractManifold

class Rn(AbstractManifold):
    def __init__(self, size):
        super().__init__(dimensions=1, size=size)

    def forward(self, X):
        return X
