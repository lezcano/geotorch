import torch
from contextlib import contextmanager

_update_base = 0


@contextmanager
def new_base():
    global _update_base
    _update_base += 1
    try:
        yield
    finally:
        _update_base -= 1


def update_base(layer, tensor_name):
    with torch.no_grad():
        with new_base():
            # execute forward pass
            getattr(layer, tensor_name)
        layer.parametrizations[tensor_name].original.zero_()


def base(forward):
    def new_forward(self, *args, **kwargs):
        X = forward(self, *args, **kwargs)
        if _update_base:
            with torch.no_grad():
                # Store the base in canonical form
                # We just transpose it if it is a matrix manifold and it's transposed
                if X.ndimension() == len(self.tensorial_size) + 2 and X.size(
                    -2
                ) < X.size(-1):
                    X = X.transpose(-2, -1)
                self.base.data.copy_(X)
        return X

    return new_forward


def transpose(forward):
    def new_forward(self, X, *args, **kwargs):
        transpose = X.size(-2) < X.size(-1)
        if transpose:
            X = X.transpose(-2, -1)
        X = forward(self, X, *args, **kwargs)
        if transpose:
            X = X.transpose(-2, -1)
        return X

    return new_forward


def _extra_repr(**kwargs):
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
    if "r" in kwargs:
        ret += ", radius={}".format(kwargs["r"])
    if "tensorial_size" in kwargs:
        ts = kwargs["tensorial_size"]
        if len(ts) != 0:
            ret += ", tensorial_size={}".format(ts)
    if "triv" in kwargs:
        ret += ", triv={}".format(kwargs["triv"].__name__)
    if "no_inv" in kwargs:
        if kwargs["no_inv"]:
            ret += ", no inverse"
    return ret
