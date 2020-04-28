from torch.nn.modules.container import ModuleDict
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from contextlib import contextmanager

_cache_enabled = 0
_cache = {}


@contextmanager
def cached():
    r"""Context-manager that enables the caching system within
    :class:`torch.nn.Parametrization`

    This is usful when one uses certain parametrized parameter more than
    once. An example of this is the loop in an RNN model
    """
    global _cache
    global _cache_enabled
    _cache_enabled += 1
    try:
        yield
    finally:
        _cache_enabled -= 1
        if not _cache_enabled:
            _cache = dict.fromkeys(_cache, None)


class Parametrization(Module):
    r"""A kind of Module that parametrizes a Parameter or a buffer in terms
    of its forward function

    After registering a Parametrization on a tensor ``t`` within a module with :meth:`
    #register_parametrization`, ``module.t`` will be turn into a property, and will
    return ``parametrization(t)``, where ``t`` is the previous tensor

    After registering a Parametrization, the parametrization will be accessible under
    ``module.parametrizations.t``. The unparametrized tensor may be accessed through
    ``parametrization.original_tensor()``

    Before a parametrization is registered, it may be chained to others to compose them
    via the ``chain`` method.

    Parametrizations may be registered on parameters, buffers, and already parametrized
    tensors

        # Defining several parametrizations on the same tensor
        class MyParametrization(nn.Parametrization):
            def forward(self, X):
                return 2. * X
        module = nn.Linear(5,3)
        p1 = MyParametrization()
        p2 = MyParametrization()
        # Register p1 on the Parameter "weight"
        register_parametrization(module, "weight", p1)
        # Register p2 on the parametrized Parameter "weight"
        register_parametrization(module, "weight", p2)
        assert(module.weight == 4. * module.parametrizations.weight.original_tensor())))

    We can also first compose them and then register them. The following code is
    equivalent to the previous one

        module = nn.Linear(5, 3)
        # Chain the two parametrizations and register them
        p2.chain(p1)
        register_parametrization(module, "weight", p2)
        # Now p2.evaluate() computes p1 and then p2
        assert(module.weight == 4. * module.parametrizations.weight.original_tensor())))


    Parametrized parameters and buffers have an in-built caching system via the context
    manager :class:``cached``
    """

    def evaluate(self):
        r"""Evaluates the parametrization"""
        if self.is_registered():
            return self(self.original)
        else:
            raise RuntimeError(
                "A Parametrization cannot be evaluated before " "registering it"
            )

    def original_tensor(self):
        r"""Returns the tensor the parametrization was registered on"""
        try:
            return self.last_parametrization().original
        except AttributeError:
            raise ValueError("The parametrization is not registered to a tensor.")

    def last_parametrization(self):
        r"""Returns the last parametrization of the chain.

        In particular, if the parametrization is not chained, it returns itself
        """
        last = self
        while last.is_chained():
            last = last.parametrizations.original
        return last

    def chain(self, parametrization):
        r""" It modifies the current parametrization, parametrizing the original
        tensor in terms of the new parametrization

        The original parametrization is modified after chain is applied to it

        Args:
            parametrization (nn.Parametrization): the parametrization to be chained
        Returns:
            Module: The original module with the parametrization module appended to it
        """
        if not isinstance(parametrization, Parametrization):
            raise ValueError(
                "Expecting a Parametrization. Found '{}'".format(type(parametrization))
            )

        if self.is_registered():
            raise ValueError(
                "Cannot chain a parametrization on a parametrization that "
                "was already registered parametrization."
            )

        _set_parametrization(self.last_parametrization(), "original", parametrization)
        return self

    def is_chained(self):
        r"""Returns True if it is chained to other parametrizations
        """
        return is_parametrized(self, "original")

    def is_registered(self):
        r"""Returns True if it is registered on a module
        """
        try:
            self.original_tensor()
            return True
        except ValueError:
            return False


def set_caching(module, tensor_name):
    r""" Sets up the caching mechanism for a given parametrization. This fuction is
    automatically invoked when using `register_parametrization` on a parameter or
    a buffer

    After applying this function, the values of the parametrization will be cached
    when the contextmanager `torch.nn.cached()` is active

    Args:
        module (nn.Module): module on which to remove the caching mechanism
        tensor_name (string): name of the parameter, buffer, or parametrization
        from which the parametrization will be applied
    """
    if not is_parametrized(module, tensor_name):
        raise ValueError(
            "The tensor {} in module {} is not parametrized".format(
                tensor_name, type(module)
            )
        )

    global _cache
    key = (id(module), tensor_name)
    if key not in _cache:
        _cache[key] = None


def remove_caching(module, tensor_name):
    r""" Removes a caching mechanism for a given parametrization. This is automatically
    done when using `remove_parametrization` on a parameter or a buffer

    After applying this function, the values of the parametrization will not be cached even
    in the presence of the contextmanager `torch.nn.cached()`

    Args:
        module (nn.Module): module on which to remove the caching mechanism
        tensor_name (string): name of the parameter, buffer, or parametrization
        from which the parametrization will be applied
    """
    if not is_parametrized(module, tensor_name):
        raise ValueError(
            "The tensor {} in module {} is not parametrized".format(
                tensor_name, type(module)
            )
        )

    global _cache
    key = (id(module), tensor_name)
    if key in _cache:
        _cache.pop(key)


def has_caching(module, tensor_name):
    r"""Returns True if module[name] is parametrized and has
    a caching mechanism

    Args:
        module (nn.Module): module to query
        tensor_name (string): attribute in the module to query
    """
    global _cache
    key = (id(module), tensor_name)
    return is_parametrized(module, tensor_name) and key in _cache


def _set_parametrization(module, tensor_name, parametrization):
    r""" Sets up the parametrization mechanism used by parametrizations.
    This works by substituting the class of the module by a class
    that extends it and makes `tensor_name` into a property. It also
    registers the parametrization under a ModuleDict called `parametrizations`.
    """
    # Define the getter
    def get_parametrized(module):
        global _cache_enabled
        global _cache

        key = (id(module), tensor_name)
        # If the the _cache is not enabled or the caching was not enabled for this
        # tensor, this function just evaluates the parametrization
        if _cache_enabled and key in _cache:
            if _cache[key] is None:
                _cache[key] = module.parametrizations[tensor_name].evaluate()
            return _cache[key]
        else:
            return module.parametrizations[tensor_name].evaluate()

    if not is_parametrized(module):
        if hasattr(module, "parametrizations"):
            raise ValueError(
                "Attribute 'parametrizations' found. Cannot parametrize "
                "a module that has an attribute named 'parametrizations'"
            )

        # If it has not been parametrized, we create a new class so that
        # we can inject properties in it
        cls_name = "Parametrized" + module.__class__.__name__

        param_cls = type(
            cls_name,
            (module.__class__,),
            {
                tensor_name: property(get_parametrized),
                "__qualname__": cls_name + str(id(module)),
            },
        )

        # Declare the class globally to be able to pickle it
        globals()[param_cls.__qualname__] = param_cls
        module.__class__ = param_cls
        module.parametrizations = ModuleDict()
    else:
        # If it has been parametrized, there is no need create a new one
        setattr(module.__class__, tensor_name, property(get_parametrized))

    # Register the parametrization
    module.parametrizations[tensor_name] = parametrization


def register_parametrization(module, tensor_name, parametrization):
    r"""Adds a parametrization to ``module[tensor_name]``

    When accessing ``module[tensor_name]``, the module will return the
    parametrized version ``parametrization(module[tensor_name])``

    Parametrizations may be composed by registering several parametrizations
    on the same attribute.

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
        on which the parametrization will be applied
        parametrization (nn.Parametrization): the parametrization to be applied
    """
    if parametrization.is_registered():
        raise ValueError(
            "The parametrization {} is already registered to other "
            "tensor. A parametrization may not be registered to more "
            "than one tensor."
        )

    if is_parametrized(module, tensor_name):
        # Putting a parametrization on a paramterization
        prev_parametrization = module.parametrizations[tensor_name]
        # Chain the parametrizations
        parametrization.chain(prev_parametrization)
    elif tensor_name in module._buffers or tensor_name in module._parameters:
        # Buffer or Parameter
        original = getattr(module, tensor_name, None)
        # Delete the previous parameter or buffer
        delattr(module, tensor_name)
        # Set the parametrization
        _set_parametrization(module, tensor_name, parametrization)
        # Set the cache on this tensor
        set_caching(module, tensor_name)
        # Register the tensor on the last parametrization
        last = parametrization.last_parametrization()
        if isinstance(original, Parameter):
            last.register_parameter("original", original)
        else:
            last.register_buffer("original", original)
    else:
        raise ValueError(
            "Module '{}' does not have a parameter, a buffer, nor a "
            "parametrized element with name '{}'".format(module, tensor_name)
        )
    # Register the parametrization
    module.parametrizations[tensor_name] = parametrization


def is_parametrized(module, tensor_name=None):
    r"""Returns True if module has an active parametrization
    If the argument ``name`` is specified, it returns True if
    module[name] returns a parametrized tensor

    Args:
        module (nn.Module): module to query
        name (string, optional): attribute in the module to query
        Default: None
    """
    parametrizations = getattr(module, "parametrizations", None)
    if parametrizations is None or not isinstance(parametrizations, ModuleDict):
        return False
    if tensor_name is None:
        # Check that there is at least one
        # This should always be true if we have module.parametrizations
        return len(module.parametrizations) > 0
    else:
        return tensor_name in module.parametrizations


def remove_parametrization(module, tensor_name, leave_parametrized=True):
    r"""Removes parametrizations active on the parameter ``tensor_name``.
    If ``leave_parametrized == True``, ``module[tensor_name]`` will be set to
    its current output: the parametrized tensor.
    If ``leave_parametrized == False``, ``module[tensor_name]`` will be set to
    its unparametrized value, that is,
    ``module.parametrizations.[tensor_name].original_tensor()``

    .. warning ::

        If the parametrization changes the size of the tensor and the parametrization
        is on a parameter being optimized, since this function will register a new
        parameter, the parameters on the optimizer have to be manually updated via
        ``optim.params = model.parameters()`` after calling this method.

    Args:
        module (nn.Module): module from which remove the parametrization
        tensor_name (str): name of the parametrization to be removed
        leave_parametrized (bool, optional): leave the attribute ``tensor_name``
        parametrized or not. Default: False
    """

    if not is_parametrized(module, tensor_name):
        raise ValueError(
            "Module {} does not have a parametrization on {}".format(
                module, tensor_name
            )
        )

    # TODO
    # We implement the removal recursively
    parametrization = module.parametrizations[tensor_name]
    original = parametrization.original_tensor()
    # Parametrization on a parameter or a buffer
    is_parameter = isinstance(original, Parameter)
    if leave_parametrized:
        t = getattr(module, tensor_name)
        if t.size() != original.size():
            if is_parameter:
                original = Parameter(t)
            else:
                original = t
        else:
            original.data = t

    # Remove the caching mechanism if it has one
    remove_caching(module, tensor_name)
    # Delete the property that manages the parametrization
    delattr(module.__class__, tensor_name)
    # Delete the parametrization
    delattr(module.parametrizations, tensor_name)

    if is_parameter:
        module.register_parameter(tensor_name, original)
    else:
        module.register_buffer(tensor_name, original)

    # Roll back the fancy parametrized class if no other
    # buffer or parameter is currently parametrized
    if not is_parametrized(module):
        # Delete the associated class
        del globals()[module.__class__.__qualname__]
        # Restore class
        parents = module.__class__.__bases__
        # If everything's working as expected, this should never throw
        if len(parents) != 1:
            raise TypeError(
                "Found a Parametrized module with more than "
                "one parent class. This is currently not supported."
            )
        module.__class__ = parents[0]
        delattr(module, "parametrizations")
