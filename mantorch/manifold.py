from abc import abstractmethod
import torch
import torch.nn.utils.parametrization as P
import torch.nn as nn

class Manifold(P.Parametrization):

    def __init__(self):
        super().__init__()
        self.register_buffer("base", None)

    def trivialization(self, x, base):
        r"""
        Parametrizes the manifold in terms of a tangent space
        Args:
            x (torch.nn.Tensor): A tensor
            base (torch.nn.Tensor): A point in the manifold
        Returns:
            tensor (torch.nn.Tensor): A tensor on the manifold
        Note:
            This function should be surjective, otherwise not all the manifold
            will be explored
        """

        raise NotImplementedError()


    def forward(self, t):
        if not hasattr(self, "orig"):
            raise RuntimeError("Parametrization {} has to be applied to a tensor with "
                    "`register_parametrization` before being used.".format(type(self).__name__))

        return self.trivialization(t, self.base)

    def size(self, k=None):
        if k is None:
            return self.original().size()
        else:
            return self.original().size(k)

    def update_base(self):
        if not hasattr(self, "orig"):
            raise RuntimeError("Parametrization {} has to be applied to a tensor with "
                    "`register_parametrization` before being used.".format(type(self).__name__))
        with torch.no_grad():
            self.base.data.copy_(self.param.data)
            self.orig.zero_()


class Frame(P.Parametrization):
    pass


class Fibration(P.Parametrization):
    pass



