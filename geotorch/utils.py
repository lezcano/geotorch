import torch


def update_base(layer, tensor_name):
    with torch.no_grad():
        setattr(layer, tensor_name, getattr(layer, tensor_name).data)


def transpose(fun):
    def new_fun(self, X, *args, **kwargs):
        # It might happen that we get at tuple inside ``right_inverse``
        # In that case we do nothing
        if isinstance(X, torch.Tensor):
            if self.transposed:
                X = X.transpose(-2, -1)
        X = fun(self, X, *args, **kwargs)
        if self.transposed:
            X = X.transpose(-2, -1)
        return X

    return new_fun


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
