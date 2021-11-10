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
from typing import Dict, Any


class Settings:
    """Class mixing for storing instance init settings.

    This mixin adds a ``__new__`` method that stores the values of args
    and kwargs passed to the class instances ``__init__`` method and a
    ``settings`` property that returns an ordered dict of these values.

    .. note::

        This mixin is intended as a mixing for base classes so that when
        creating subclasses, those subclasses can inherit the logic for
        saving and returning settings.

        Note that there is small performance overhead to initializing classes
        with this mixin so it should not be used for adding settings to all
        classes without consideration. For classes that already store values
        required to recover the ``__init__`` args they should instead
        implement an appropriate :meth:`settings` property directly.
    """

    def __new__(cls, *args, **kwargs):
        # This method automatically stores all arg and kwargs from subclass
        # init methods for use in converting an experiment to config

        # Get all non-self init args and kwarg names for subclass
        spec = inspect.getfullargspec(cls.__init__)
        init_arg_names = spec.args[1:]
        num_init_kwargs = len(spec.defaults) if spec.defaults else 0
        num_init_args = len(init_arg_names) - num_init_kwargs

        # Convert passed values for args and kwargs into an ordered dict
        # This will sort args passed as kwargs and kwargs passed as
        # positional args in the function call
        num_call_args = len(args)
        ord_args = OrderedDict()
        ord_kwargs = OrderedDict()
        for i, argname in enumerate(init_arg_names):
            if i < num_init_args:
                update = ord_args
            else:
                update = ord_kwargs
            if i < num_call_args:
                update[argname] = args[i]
            elif argname in kwargs:
                update[argname] = kwargs[argname]

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

    @property
    def settings(self) -> Dict[str, Any]:
        """Return the settings used to initialize this instance."""
        settings = {}
        # Note that this relies on dicts entries being implicitly ordered
        # to store init args as kwargs.
        args = getattr(self, "__init_args__", {})
        for key, val in args.items():
            settings[key] = val
        kwargs = getattr(self, "__init_kwargs__", {})
        for key, val in kwargs.items():
            settings[key] = val
        return settings
