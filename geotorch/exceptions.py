class VectorError(ValueError):
    def __init__(self, name, size):
        super().__init__(
            "Cannot instantiate {} on a tensor of less than 2 dimensions. "
            "Got a tensor of size {}".format(name, size)
        )


class NonSquareError(ValueError):
    def __init__(self, name, size):
        super().__init__(
            "The {} parametrization can just be applied to square matrices. "
            "Got a tensor of size {}".format(name, size)
        )


class RankError(ValueError):
    def __init__(self, n, k, rank):
        super().__init__(
            "The rank has to be 1 <= rank <= min({}, {}). Found {}".format(n, k, rank)
        )
