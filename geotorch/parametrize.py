import torch
from torch.nn.modules.container import ModuleList, ModuleDict
from torch.nn.parameter import Parameter
from contextlib import contextmanager


_cache_enabled = 0
_cache = {}


def _key(module, tensor_name):
    return id(module), tensor_name


@contextmanager
def cached():
    r"""Context-manager that enables the caching system within
    parametrizations registered with :func:`register_parametrization`

    This is useful when one uses certain parametrized parameter more than
    once. An example of this is the loop in an RNN model or when sharing weights
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


class ParametrizationList(ModuleList):
    r"""A sequential container that holds and manages a Parameter of a buffer.
    Parametrized parameters and buffers have an in-built caching system via the context
    manager :func:`cached`
    """

    def __init__(self, modules, original):
        super().__init__(modules)
        if isinstance(original, Parameter):
            self.register_parameter("original", original)
        else:
            self.register_buffer("original", original)

    def evaluate(self):
        r"""Evaluates the parametrization"""
        return self(self.original)

    def set_value_(self, value):
        with torch.no_grad():
            for module in reversed(self):
                value = module.initialize_(value)
            self.original.copy_(value)

    def forward(self, input):
        for module in self:
            input = module(input)
        return input


def set_caching(module, tensor_name):
    r"""Sets up the caching mechanism for a given parametrization. This function is
    automatically invoked when using :func:`register_parametrization`

    After applying this function, the values of the parametrization will be cached
    when the contextmanager :func:`torch.nn.cached()` is active

    This function is the inverse of :func:`remove_caching`

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

    key = _key(module, tensor_name)
    if key not in _cache:
        _cache[key] = None


def remove_caching(module, tensor_name):
    r"""Removes a caching mechanism for a given parametrization

    After applying this function, the values of the parametrization will not be cached even
    in the presence of the contextmanager :func:`torch.nn.cached()`

    This function is the inverse of :func:`set_caching`

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

    key = _key(module, tensor_name)
    if key in _cache:
        _cache.pop(key)


def has_caching(module, tensor_name):
    r"""Returns True if module[name] is parametrized and has
    a caching mechanism

    Args:
        module (nn.Module): module to query
        tensor_name (string): attribute in the module to query
    """
    key = _key(module, tensor_name)
    return is_parametrized(module, tensor_name) and key in _cache


def _inject_parametrization_list(module):
    if not hasattr(module, "parametrizations"):
        # If there's no attribute, we add one
        module.parametrizations = ModuleDict()
    else:
        # The module has a `module.parametrizations` of a different type. We notify of this
        raise ValueError(
            "Attribute 'parametrizations' found of type different to ModuleDict."
            "Cannot parametrize a module that has an attribute named 'parametrizations'"
        )


def _inject_new_class(module):
    r"""Sets up the parametrization mechanism used by parametrizations.
    This works by substituting the class of the module by a class
    that extends it to be able to inject a property

    Args:
        module (nn.Module): module on which to inject the property
    """

    # We create a new class so that we can inject properties in it
    cls_name = "Parametrized" + module.__class__.__name__

    param_cls = type(
        cls_name,
        (module.__class__,),
        {
            "__qualname__": cls_name + str(id(module)),
        },
    )

    # Declare the class globally to be able to pickle it
    # TODO Is there a better way to do this?
    # Perhaps via __reduce__? See second answer in:
    # https://stackoverflow.com/questions/4647566/pickle-a-dynamically-parameterized-sub-class
    globals()[param_cls.__qualname__] = param_cls
    module.__class__ = param_cls


def _inject_property(module, tensor_name):
    # Define the getter
    def get_parametrized(module):
        global _cache_enabled
        global _cache

        key = _key(module, tensor_name)
        # If the _cache is not enabled or the caching was not enabled for this
        # tensor, this function just evaluates the parametrization
        if _cache_enabled and key in _cache:
            if _cache[key] is None:
                _cache[key] = module.parametrizations[tensor_name].evaluate()
            return _cache[key]
        else:
            return module.parametrizations[tensor_name].evaluate()

    # Define the setter
    def set_value(module, value):
        module.parametrizations[tensor_name].set_value_(value)

    setattr(module.__class__, tensor_name, property(get_parametrized, set_value))


def register_parametrization(module, tensor_name, parametrization):
    r"""Adds a parametrization to ``module[tensor_name]``

    If the module was not parametrized, this function will add an attribute
    ``parametrizations`` to the module, and the list of parametrizations on
    the parameter ``tensor_name`` will be accessible under
    ``module.parametrizations[tensor_name]``.

    The parameter or buffer will be accessible under
    ``module.parametrizations[tensor_name].original``

    When accessing ``module[tensor_name]``, the module will return the
    parametrized version ``parametrization(module[tensor_name])``

    Parametrizations may be composed by registering several parametrizations
    on the same attribute. The new parametrizations will be added appended to
    ``module.parametrizations[tensor_name]`` which acts as a :class:`nn.Sequential`
    module

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string): name of the parameter, buffer, or parametrization
        on which the parametrization will be applied
        parametrization (nn.Parametrization): the parametrization to be applied
    """
    if is_parametrized(module, tensor_name):
        # Just add the new parametrization to the parametrization list
        module.parametrizations[tensor_name].append(parametrization)
    elif tensor_name in module._buffers or tensor_name in module._parameters:
        # Set the parametrization mechanism
        # Fetch the original buffer or parameter
        original = getattr(module, tensor_name, None)
        # Delete the previous parameter or buffer
        delattr(module, tensor_name)
        # If this is the first parametrization of the module, we set it up
        if not is_parametrized(module):
            # Inject the a ModuleDict into module.parametrizations if it does not exist yet
            _inject_parametrization_list(module)
            # Change the class
            _inject_new_class(module)
        # Add a property into the class
        _inject_property(module, tensor_name)
        # Add a ParametrizationList
        module.parametrizations[tensor_name] = ParametrizationList(
            [parametrization], original
        )
        # Set the cache on this tensor
        set_caching(module, tensor_name)
    else:
        raise ValueError(
            "Module '{}' does not have a parameter, a buffer, nor a "
            "parametrized element with name '{}'".format(module, tensor_name)
        )


def is_parametrized(module, tensor_name=None):
    r"""Returns True if module has an active parametrization
    If the argument ``name`` is specified, it returns True if
    `module[name]` returns a parametrized tensor

    Args:
        module (nn.Module): module to query
        name (string, optional): attribute in the module to query
        Default: None
    """
    parametrizations = getattr(module, "parametrizations", None)
    if parametrizations is None or not isinstance(parametrizations, ModuleDict):
        return False
    if tensor_name is None:
        # Check that there is at least one parametrized buffer or Parameter
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

    # Fetch the original tensor
    original = module.parametrizations[tensor_name].original
    is_parameter = isinstance(original, Parameter)
    if leave_parametrized:
        t = getattr(module, tensor_name)
        # TODO What is a robust way to assure that original.data = t will succeed?
        # Do we have to check the stride, floating type...? Maybe do a try/catch?
        if t.size() == original.size():
            original.data = t
        else:
            if is_parameter:
                original = Parameter(t)
            else:
                original = t

    # Remove the caching mechanism if it has one
    remove_caching(module, tensor_name)
    # Delete the property that manages the parametrization
    delattr(module.__class__, tensor_name)
    # Delete the ParametrizatinList
    del module.parametrizations[tensor_name]

    # Restore the parameter / buffer into the main class
    if is_parameter:
        module.register_parameter(tensor_name, original)
    else:
        module.register_buffer(tensor_name, original)

    # Roll back the parametrized class if no other buffer or parameter
    # is currently parametrized in this class
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
