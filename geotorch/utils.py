import math
import torch


def update_base(layer, tensor_name):
    with torch.no_grad():
        setattr(layer, tensor_name, getattr(layer, tensor_name))


def transpose(fun):
    def new_fun(self, X, *args, **kwargs):
        # It might happen that we get at tuple inside ``initialize_``
        # In that case we do nothing
        if isinstance(X, torch.Tensor):
            if self.transposed:
                X = X.transpose(-2, -1)
        X = fun(self, X, *args, **kwargs)
        if self.transposed:
            X = X.transpose(-2, -1)
        return X

    return new_fun


def normalized_matrix_one_norm(X):
    if X.size(-2) != X.size(-1):
        raise ValueError("X should be square. Got {}".format(X.size()))
    # Older torch versions do not implement matrix norm 1
    # We normalise by sqrt(n) as that's the growth of the operator norm with n. See
    # https://terrytao.wordpress.com/2010/01/09/254a-notes-3-the-operator-norm-of-a-random-matrix/
    n = X.size(-1)
    if torch.__version__ >= "1.7.0":
        return torch.linalg.norm(X, dim=(-2, -1), ord=1) / math.sqrt(n)
    else:
        return X.abs().sum(dim=-2).max(dim=-1).values / math.sqrt(n)


def _extra_repr(**kwargs):  # noqa: C901
    if "n" in kwargs:
        ret = "n={}".format(kwargs["n"])
    elif "dim" in kwargs:
        ret = "dim={}".format(kwargs["dim"])
    else:
        ret = ""

    if "k" in kwargs:
        ret += ", k={}".format(kwargs["k"])
    if "rank" in kwargs:
        ret += ", rank={}".format(kwargs["rank"])
    if "radius" in kwargs:
        ret += ", radius={}".format(kwargs["radius"])
    if "lam" in kwargs:
        ret += ", lambda={}".format(kwargs["lam"])
    if "f" in kwargs:
        ret += ", f={}".format(kwargs["f"].__name__)
    if "tensorial_size" in kwargs:
        ts = kwargs["tensorial_size"]
        if len(ts) != 0:
            ret += ", tensorial_size={}".format(tuple(ts))
    if "triv" in kwargs:
        ret += ", triv={}".format(kwargs["triv"].__name__)
    if "no_inv" in kwargs:
        if kwargs["no_inv"]:
            ret += ", no inverse"
    if "transposed" in kwargs:
        if kwargs["transposed"]:
            ret += ", transposed"
    return ret
