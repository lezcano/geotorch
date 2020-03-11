from abc import abstractmethod
import torch
import torch.nn as nn

class Manifold(nn.Parametrization):

    def __init__(self):
        super().__init__()
        self.register_buffer("base", None)

    def init(self, t):
        raise NotImplementedError()

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
        if self.base is None:
            raise RuntimeError("Parametrization {} has to be applied to a tensor with "
                    "`module.register_parametrization`".format(type(self).__name__))

        return self.trivialization(t, self.base)

    @property
    def size(self):
        x = self.original
        n = x.size(-2)
        m = x.size(-1)
        return n, m

    def update_base(self):
        if "orig" not in self._parameters:
            raise RuntimeError("Parametrization {} has to be applied to a tensor with "
                    "`module.register_parametrization`".format(type(self).__name__))
        with torch.no_grad():
            self.base.data.copy_(self.param.data)
                self.orig.zero_()
