from .manifold import Frame

class Skew(Frame):

    def __init__(self, lower=True):
        self.lower = lower

    def forward(selt, X):
        if t.ndimension() != 2 or t.size(0) != t.size(1):
            raise ValueError("Expected a square matrix. Got a tensor of shape {}"
                             .format(list(t.size())))
        if self.lower:
            X = X.tril(-1)
        else:
            X = X.triu(-1)
        return X - X.t()

    def extra_repr(self):
        return "n={}".format(self.orig.size(0)) if hasattr(self, "orig") else ""


