# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Settings mixing class
"""

import inspect
from collections import OrderedDict
from functools import wraps


class StoreInitArgs:
    """Class mixing for storing class and subclass instance init args.

    This mixin adds a ``__new__`` method that stores the values of args
    and kwargs passed to the class instances ``__init__`` method.
    These are stored as ordered dicts under attributes ``__init_args__`
    and ``__init_kwargs__`` respectively.

    .. note::

        This mixin is intended as a mixing for base classes so that when
        creating subclasses, those subclasses can inherit the logic for
        saving and returning settings.

        Note that there is small performance overhead to initializing classes
        with this mixin so it should not be used for adding settings to all
        classes without consideration.
    """

    def __new__(cls, *args, **kwargs):
        # This method automatically stores all arg and kwargs from subclass
        # init methods
        spec = inspect.getfullargspec(cls.__init__)
        ord_args = OrderedDict()
        ord_kwargs = OrderedDict()

        # Parse spec args
        defaults = spec.defaults or []
        if defaults:
            size = len(spec.args) - len(spec.defaults)
            init_args = spec.args[1:size]
            init_kwargs = spec.args[size:]
        else:
            init_args = spec.args[1:]
            init_kwargs = []

        # Initialize defaults to preserve correct arg order
        num_args = len(init_args)
        ord_args.update(zip(init_args, num_args * [None]))
        ord_kwargs.update(zip(init_kwargs, defaults))

        if init_args and args:
            # Add named args
            ord_args.update(zip(init_args, args))
        if init_kwargs and args:
            # Update non-default values
            ord_kwargs.update(zip(init_kwargs, args[num_args:]))

        # Parse variadic args
        if spec.varargs:
            num_varargs = len(args) - num_args
            ord_args.update(
                ((f"{spec.varargs}[{i}]", args[num_args + i]) for i in range(num_varargs))
            )

        # Add defaults for kwonly args
        for kwarg in spec.kwonlyargs:
            if kwarg not in ord_kwargs:
                ord_kwargs[kwarg] = spec.kwonlydefaults.get(kwarg, None)

        # Parse kwargs
        for arg, argval in kwargs.items():
            if arg in init_args:
                ord_args[arg] = argval
            else:
                ord_kwargs[arg] = argval

        # pylint: disable = attribute-defined-outside-init
        instance = super().__new__(cls)
        instance.__init_args__ = ord_args
        instance.__init_kwargs__ = ord_kwargs
        return instance

    def __init_subclass__(cls, **kwargs):
        # This method fixes class documentations for subclass
        # that inherit the base class new method
        super().__init_subclass__(**kwargs)

        # Copy the doc string and annotation from the subclasses
        # init method to its new method to override base class
        # __new__ documentation
        @wraps(cls.__init__, assigned=("__annotations__",))
        def __new__(sub_cls, *args, **kwargs):
            return super(cls, sub_cls).__new__(sub_cls, *args, **kwargs)

        # Monkey patch the subclass new method with the method with
        # fixed documentation annotations
        cls.__new__ = __new__
