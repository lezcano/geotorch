from torch.nn.modules.container import ModuleDict
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import functools
import inspect


class Parametrization(Module):
    r"""A kind of Module that parametrizes a Parameter or a buffer in terms
    of its forward function

    After registering a Parametrization on a tensor ``t`` within a module with :meth:`
    #register_parametrization`, ``module.t`` will be turn into a property, and will
    return ``parametrization(t)``, where ``t`` is the previous tensor. The previous
    tensor will be renamed to ``_t`` and will be accessible through ``module._t``

    After registering a Parametrization, the parametrization will be accessible under
    ``module.parametrizations.t``

    Before a parametrization is registered, it may be chained to others to compose them
    via the ``chain`` method. After being registered on a tensor, a chain of parametrizations
    may be evaluated via ``parametrization.evaluate()``.

    Parametrizations may be registered on parameters, buffers, and already parametrized
    tensors

        # Defining several parametrizations on the same tensor
        module = nn.Linear(5,3)
        p1 = MyParametrization()
        p2 = MyParametrization()
        # Register p1 on the Parameter "weight"
        register_parametrization(module, "weight", p1)
        # Register p2 on the parametrized Parameter "weight"
        register_parametrization(module, "weight", p2)
        assert(module.weight == p2(p1(module._weight)))

    We can also first compose them and then register them. The following code is
    equivalent to the previous one

        module = nn.Linear(5, 3)
        # Now p2.evaluate computes p1 and then p2
        p2.chain(p1)
        # Register the chained parametrization
        register_parametrization(module, "weight", p2)
        assert(module.weight == p2(p1(module._weight)))


    Parametrized parameters and buffers have an in-built caching system via the context
    manager :class:``cached`` and the decorator :class: ``cached_method``
    """

    def evaluate(self, *input):
        r""" It evaluates the current chain of parametrizations on the inputs given

        If no inputs are given, and the parametrization has been registered, it evaluates
        the current chain of parametrizations on the tensor or tensors on which it was registered
        """
        if self.is_chained():
            # Recursive step
            out = self.parametrizations.originals.evaluate(*input)
            # If a tuple, upack it
            # This allows to define and chain Parametrizations that
            # take several arguments and return several arguments
            if isinstance(out, tuple):
                return self(*out)
            else:
                return self(out)
        else:
            if len(input) == 0:
                if self.is_registered():
                    input = self.originals
                else:
                    raise ValueError(
                        "Cannot evaluate a Parametrization that has not "
                        "been registered without providing a tensor."
                    )
            return self(*input)

    def last_parametrization(self):
        r"""Returns the last parametrization of the chain.

        In particular, if the parametrization is not chained, it returns itself
        """
        last = self
        while last.is_chained():
            last = last.parametrizations.originals
        return last

    def chain(self, parametrization):
        r""" It modifies the current parametrization, chaining a new one to it.
        We have that ``p2.evaluate(p1.evaluate(t)) == p2.chain(p1).evaluate(t)``
        The original parametrization, ``p2`` in this case,  is modified after
        chain is applied to it

        Args:
            parametrization (nn.Parametrization): the parametrization to be chained
        Returns:
            Module: The original module with the parametrization module appended to it
        """
        if not isinstance(parametrization, Parametrization):
            raise ValueError(
                "The object '{}' is not a Parametrization".format(parametrization)
            )

        is_registered = self.is_registered()
        if is_registered and parametrization.is_registered():
            raise ValueError(
                "Both parametrizations are registered to a tensor. "
                "The chaining is ambiguous."
            )

        last = self.last_parametrization()
        if is_registered:
            # We save the original tensor to move it to the new last parametrization
            originals = last.originals
            delattr(last, "originals")
        set_parametrization(last, "originals", parametrization, set_cache=False)
        if is_registered:
            new_last = parametrization.last_parametrization()
            new_last.originals = originals
        return self

    def is_chained(self):
        r"""Returns True if it is chained to other parametrizations
        """
        return is_parametrized(self, "originals")

    def is_registered(self):
        r"""Returns True if it is registered on a module
        """
        return hasattr(self.last_parametrization(), "originals")


def set_parametrization(module, name, parametrization, set_cache=True):
    r""" This is a low-level function. You’ll generally want to use
    :meth:` #register_parametrization` instead.

    Sets up the parametrization mechanism used by parametrizations.
    It sets the module so that `module.name` returns the same as
    `parametrization.evaluate()`. It also saves the parametrization
    under `module.parametrizations[name]`.
    If ``set_cache == True``, it also sets up the caching mechanism

    Args:
        module (nn.Module): module on which to set the parametrization
        name (string): name through which the parametrization will be accessed
        parametrization (nn.Parametrization): the parametrization to be set
        set_cache (bool, optional): sets up a caching mechanism. Default: True
    """

    # Set the cache as an unnamed tensor so that it does not come up in the state_dict
    if set_cache:
        cache_name = name + "_cache"
        # In case there was an attribute with name "name+'_cache'"
        if hasattr(module, cache_name):
            raise ValueError(
                "Attribute '{}' found having the unparametrized tensor '{}'. "
                "Cannot parametrize '{}' if there exists an attribute with"
                "name '{}' which is not part of the parametrization".format(
                    cache_name, name, name, cache_name
                )
            )
        setattr(module, cache_name, None)

    # Define the getter and set the caching mechanism
    # If the caching mechanism is off, this function just evaluates the parametrization
    def get_parametrized(module):
        cache = getattr(module, name + "_cache", None)
        if cache is not None:
            return cache
        else:
            return module.parametrizations[name].evaluate()

    if not is_parametrized(module):
        if hasattr(module, "parametrizations"):
            if set_cache:
                # Before throwing, we clean the things that we might have set
                delattr(module, cache_name, None)
            raise ValueError(
                "Attribute 'parametrizations' found. Cannot parametrize "
                "a module that already has an attribute named "
                "'parametrizations'"
            )

        # If it has not been parametrized, we create a new class so that
        # we can inject properties in it
        cls_name = "Parametrized" + module.__class__.__name__

        param_cls = type(
            cls_name,
            (module.__class__,),
            {
                name: property(get_parametrized),
                "__qualname__": cls_name + str(id(module)),
            },
        )

        # Declare the class globally to be able to pickle it
        globals()[param_cls.__qualname__] = param_cls
        module.__class__ = param_cls
        module.parametrizations = ModuleDict()
    else:
        # If it has been parametrized, there is no need create a new one
        setattr(module.__class__, name, property(get_parametrized))

    # Register the parametrization
    module.parametrizations[name] = parametrization


def register_parametrization(module, tensor_name, parametrization, name=None):
    r"""Adds a parametrization to ``module[tensor_name]``

    When accessing ``module[name]``, the module will return the parametrized
    version ``parametrization(module[tensor_name])`` If the argument ``name``
    is not specified, it defaults to ``tensor_name``. The argument ``name``
    allows for putting several parametrizations on the same tensor

    Parametrizations may be composed by registering several parametrizations
    on the same attribute. When doing so, these will be chained

    A parametrization may be registered on several parameters or buffers at
    the same time by giving a list of tensor_names, as long as these have not
    been parametrized. If one does so, it is necessary to specify the ``name``
    argument

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (string or list of strings): parameter or parameters on which
        the parametrization will be applied.
        parametrization (nn.Parametrization): the parametrization to be applied
        name (string, optional): name through which the parametrization will be
        accessed. Default: Same as tensor_name
    """
    is_str = isinstance(tensor_name, str)
    # Check the tensor_name input
    if not is_str and not (
        isinstance(tensor_name, list) and all(isinstance(t, str) for t in tensor_name)
    ):
        raise ValueError(
            "'tensor_name' has to be a string or list of strings. "
            "Got '{}'".format(tensor_name.__class__)
        )

    # We treat the same passing "weight" and ["weight"]
    if is_str:
        tensor_name = [tensor_name]

    # We differentiate this case, as in this case we may chain
    # the parametrization with a previous parametrization
    single_tensor = len(tensor_name) == 1

    if not single_tensor:
        # The user has to specify a name...
        if name is None:
            raise ValueError(
                "If a list of tensors is specified, the argument 'name' "
                "has to be specified as well"
            )
        # ... that is currently not in the module as something
        # (unless it's a buffer or a parameter, those we'll rename to "module['_'+name]")
        elif hasattr(module, name) and not (
            name in module._buffers or name in module._parameters
        ):
            raise ValueError(
                "Module has attribute '{}'. Cannot create a parametrization "
                "of multiple tensors.".format(name)
            )

        # Entries may not be repeated
        if len(set(tensor_name)) != len(tensor_name):
            raise ValueError(
                "The list 'tensor_name' shall not have repeated entries. Got "
                "{}".format(tensor_name)
            )

    # The parametrization that we were passed may not have been registered
    if parametrization.is_registered():
        raise ValueError(
            "The parametrization '{}' is already registered on another "
            "tensor. A parametrization may not be registered on more "
            "than one tensor."
        )

    if name is None:
        name = tensor_name[0]

    if single_tensor and is_parametrized(module, name):
        # Chain the parametrizations
        prev_parametrization = module.parametrizations[name]
        parametrization.chain(prev_parametrization)
    else:
        # Rename
        tensor_names = tensor_name
        # We collect in this loop a list of all the tensors
        originals = []
        for tensor_name in tensor_names:
            if tensor_name in module._buffers or tensor_name in module._parameters:
                originals.append(getattr(module, tensor_name))
            elif is_parametrized(module, tensor_name):
                # We do not allow to put a multiple parametrization on already-parametrized
                # tensors, as it is not possible to load it correctly
                raise ValueError(
                    "We do not allow to put a multiple parametrization "
                    "on an already-parametrized tensor '{}'".format(tensor_name)
                )
            else:
                # Note: Here we haven't started modifying `module` yet, so it is fine to
                # throw. No need to do any cleaning before doing so
                raise ValueError(
                    "Module '{}' does not have a parameter, a buffer, "
                    "nor a parametrized element with name '{}'".format(
                        module, tensor_name
                    )
                )

        if name in module._buffers or name in module._parameters:
            # We move the buffer or parameter module[name]
            # to module["_"+name] to make room for the property
            under_name = "_{}".format(name)
            if hasattr(module, under_name):
                raise ValueError(
                    "The module '{}' already has an attribute "
                    "with name '{}'. Could not move the tensor {}".format(
                        module, under_name, name
                    )
                )
            original = getattr(module, name)
            delattr(module, name)
            if isinstance(original, Parameter):
                module.register_parameter(under_name, original)
            else:
                module.register_buffer(under_name, original)
        # Set the parametrization
        set_parametrization(module, name, parametrization, set_cache=True)
        # Register the original tensors in the last parametrization
        last = parametrization.last_parametrization()
        last.originals = originals

    # Register the parametrization
    module.parametrizations[name] = parametrization


def is_parametrized(module, name=None):
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
    if name is None:
        # Check that there is at least one
        # This should always be true if we have module.parametrizations
        return len(module.parametrizations) > 0
    else:
        return name in module.parametrizations


def has_caching(module, name):
    r"""Returns True if module[name] is parametrized and has
    a caching mechanism

    Args:
        module (nn.Module): module to query
        name (string): attribute in the module to query
    """
    return is_parametrized(module, name) and hasattr(module, name + "_cache")


def remove_parametrization(module, name, leave_parametrized=True):
    r"""Removes active parametrization with name ``name``.
    If ``leave_parametrized == True``, ``module[name]`` will be set to
    its current output: the parametrized tensor.
    If ``leave_parametrized == False``, ``module[name]`` will be set to
    its unparametrized value, that is, ``module.["_" + name]`` if
    ``module["_" + name]`` is a parameter or a buffer and
    ``module.parametrizations[name]`` parametrizes it, otherwise it will
    just remove the parametrization.

    .. warning ::

        When setting ``leave_parametrized == True``, if the parametrization
        changes the size of the tensor and the parametrization is on a parameter
        being optimized, since this function will allocate a new parameter, the
        parameters on the optimizer have to be manually updated via
        ``optim.params = model.parameters()`` after calling this method.

    Args:
        module (nn.Module): module from which remove the parametrization
        name (str): name of the parametrization to be removed
        leave_parametrized (bool, optional): leave the attribute ``name``
        parametrized or not. Default: False
    """

    if not is_parametrized(module, name):
        raise ValueError(
            "Module '{}' does not have a parametrization on '{}'".format(module, name)
        )

    parametrization = module.parametrizations[name]
    originals = parametrization.last_parametrization().originals
    if len(originals) == 1:
        original = originals[0]
    else:
        original = None
    under_name = "_{}".format(name)
    # Same condition as that used in register_parametrization to move it
    same_name = under_name in module._parameters or under_name in module._buffers

    is_parameter = isinstance(original, Parameter)

    if leave_parametrized:
        t = getattr(module, name)
        # We reuse the previous tensor if we can
        if same_name and t.size() == original.size():
            original.data = t
        else:
            if is_parameter:
                original = Parameter(t)
            else:
                original = t

    # Delete the property that manages the parametrization
    delattr(module.__class__, name)
    # Delete the parametrization
    delattr(module.parametrizations, name)
    if hasattr(module, name + "_cache"):
        delattr(module, name + "_cache")

    # If same_name, in particular we have len(originals) == 1
    if leave_parametrized or same_name:
        if is_parameter:
            module.register_parameter(name, original)
        else:
            module.register_buffer(name, original)
        delattr(module, under_name)

    # Roll back the fancy parametrized class if no other
    # buffer or parameter is currently parametrized
    if not is_parametrized(module):
        # Note: We do not delete the class from globals() in case we need to
        # load it later using pickling

        # Restore class
        parents = module.__class__.__bases__
        # This could even be an assert
        if len(parents) != 1:
            raise TypeError(
                "Found a Parametrized module with more than "
                "one parent class. This is currently not supported."
            )
        module.__class__ = parents[0]
        delattr(module, "parametrizations")


class cached:
    r"""Context-manager that enables the caching system within
    :class:`torch.nn.Parametrization`

    This is usful when one uses certain parametrized parameter more than
    once. An example of this is the loop in an RNN model
    """

    def __init__(self, model):
        if model is not None and not isinstance(model, Module):
            raise TypeError(
                "Model should be an nn.Module. Found '{}'".format(model.__class__)
            )
        self.model = model

    def __enter__(self):
        def _enable_caching(module):
            if not hasattr(module, "parametrizations"):
                return
            for name, parametrization in module.parametrizations.items():
                if has_caching(module, name):
                    setattr(module, name + "_cache", parametrization.evaluate())

        self.model.apply(_enable_caching)
        return self.model

    def __exit__(self, *args):
        def _disable_caching(module):
            if not hasattr(module, "parametrizations"):
                return
            for name, _ in module.parametrizations.items():
                if has_caching(module, name):
                    setattr(module, name + "_cache", None)

        self.model.apply(_disable_caching)


class cached_method(cached):
    r"""Decorator that enables the caching system within
    :class:`torch.nn.Parametrization`

    It is used to decorate a method of a model where we want to enable caching,
    for example, the forward method of a Module with parametrized submodules
    """

    def __init__(self):
        super(cached_method, self).__init__(None)

    def __call__(self, func):
        if inspect.isgeneratorfunction(func):
            return self._wrap_generator(func)

        @functools.wraps(func)
        def decorate_context(mod, *args, **kwargs):
            self.model = mod
            with self:
                return func(mod, *args, **kwargs)

        return decorate_context

    def _wrap_generator(self, func):
        """Wrap each generator invocation with the context manager"""

        @functools.wraps(func)
        def generator_context(mod, *args, **kwargs):
            gen = func(mod, *args, **kwargs)
            self.model = mod
            while True:
                try:
                    with self:
                        x = next(gen)
                    yield x
                except StopIteration:
                    break

        return generator_context
