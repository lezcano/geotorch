from abc import abstractmethod
import torch
import torch.nn as nn

class Manifold(nn.Parametrization):

    def __init__(self):
        super().__init__()
        self.register_buffer("base", None)

    def init(self, t):
        raise NotImplementedError()

    def frame(self, x, base):
        r"""
        Parametrizes the tangent space in terms of x
        Args:
            tensor (torch.nn.Tensor): A tensor of size n x m.
            tensor (torch.nn.Tensor): A point on the parametrized manifold (i.e. a matrix)
        Returns:
            tensor (tensor.nn.Tensor): A tensor of size in the tangent space to which
                                       we are pulling back the optimization problem
        """
        raise NotImplementedError()

    def trivialization(self, x, base):
        r"""
        Parametrizes the manifold in terms of the tangent space from the frame
        Args:
            x (torch.nn.Tensor): A tensor returned by self.frame(x)
            base (torch.nn.Tensor): The point at which the trivialization is performed
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

        # Element on a tangent space
        base = self.base
        x = self.frame(t, base)

        # Element on the manifold
        x = self.trivialization(x, base)
        return x

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
