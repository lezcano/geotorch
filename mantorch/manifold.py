from abc import abstractmethod
import torch
import torch.nn.utils.parametrize as P

class BaseManifold(P.BaseParametrizationMethod):

    @classmethod
    def apply(cls, module, name, mode="auto"):
        method = super().apply(module, name, mode)
        method._module.register_buffer(method._tensor_name + "_base", None)
        return method

    @abstractmethod
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

    @abstractmethod
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

    @property
    def param(self):
        return getattr(self._module, self._tensor_name)

    @property
    def param_orig(self):
        orig_name = self._tensor_name + "_orig"
        return getattr(self._module, orig_name)

    @property
    def base(self):
        base_name = self._tensor_name + "_base"
        return getattr(self._module, base_name)

    def update_base(self):
        with torch.no_grad():
            self.base.data.copy_(self.param.data)
            self.param_orig.zero_()

    def compute_param(self, t):
        # Element on a tangent space
        base = self.base
        x = self.frame(t, base)

        # Element on the manifold
        x = self.trivialization(x, base)
        return x

    def remove(self):
        super().remove()
        self._delete_base()

    def undo(self):
        super().undo()
        self._delete_base()

    def _delete_base(self):
        if self.mode == "forward":
            params = []
        elif self.mode == "auto":
            params = get_auto_parametrizations(self._module, self._tensor_name)
        # If it's the last one we remove it
        if len(params) == 0:
            del self._module._buffers[self._tensor_name + "_base"]
