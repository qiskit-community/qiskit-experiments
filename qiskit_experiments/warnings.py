# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A collection of typical warnings."""

import functools
import warnings

from typing import Callable, Optional, Type, Dict, Union


def deprecated_function(
    last_version: Optional[str] = None,
    msg: Optional[str] = None,
    stacklevel: int = 2,
) -> Callable:
    """A function or method decorator to show deprecation warning.

    Args:
        last_version: The Qiskit Experiment version that this function is removed.
        msg: Extra message, for example, to indicate alternative approach.
        stacklevel: Stacklevel of this warning. See Python Warnings documentation for details.

    Examples:

        .. code-block::

            @deprecated_logic(last_version="0.3", msg="Use new_function instead.")
            def old_function(*args, **kwargs):
                pass

            def new_function(*args, **kwargs):
                pass

    Returns:
        Deprecated function or method.
    """

    def deprecated_wrapper(func: Callable):
        @functools.wraps(func)
        def _wrap(*args, **kwargs):
            namespace = func.__qualname__.split(".")
            if len(namespace) == 1:
                message = f"The function '{func.__name__}' has been deprecated and "
            else:
                cls_name, meth_name = namespace
                message = f"The method '{meth_name}' of '{cls_name}' class has been deprecated and "
            if last_version:
                message += f"will be removed after Qiskit Experiments {last_version}. "
            else:
                message += "will be removed in future release. "
            if msg:
                message += msg
            warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)
            return func(*args, **kwargs)

        return _wrap

    return deprecated_wrapper


def deprecated_class(
    last_version: Optional[str] = None,
    new_cls: Optional[Type] = None,
    msg: Optional[str] = None,
    stacklevel: int = 2,
) -> Callable:
    """A class decorator to show deprecation warning and
    patch __new__ method of the class to instantiate the new class.

    Args:
        last_version: The Qiskit Experiment version that this class is removed.
        new_cls: Alternative class type.
        msg: Extra message, for example, to indicate alternative approach.
        stacklevel: Stacklevel of this warning. See Python Warnings documentation for details.

    Examples:

        .. code-block::

            @deprecated_class(last_version="0.3", new_cls=NewCls)
            class OldClass:
                pass

            class NewClass:
                pass

    Returns:
        Deprecated class.
    """

    def patch_new(cls) -> Type:
        @functools.wraps(cls.__init__, assigned=("__annotations__",))
        def new(deprecated_cls, *args, **kwargs):
            message = f"Class '{deprecated_cls.__name__}' has been deprecated"
            if new_cls:
                message += f" and replaced with '{new_cls.__name__}'. "
            else:
                message += ". "
            if last_version:
                message += f"This class will be removed after Qiskit Experiments {last_version}. "
            else:
                message += "This class will be removed in future release. "
            if msg:
                message += msg
            warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)
            instance = object.__new__(new_cls or deprecated_cls)
            instance.__init__(*args, **kwargs)
            return instance

        cls.__new__ = new
        return cls

    return patch_new


def deprecated_options(
    options_map: Dict[str, Union[str, None]],
    last_version: Optional[str] = None,
    msg: Optional[str] = None,
    stacklevel: int = 2,
) -> Callable:
    """A set options method decorator to show deprecation warning for deprecated options.

    Args:
        options_map: A dictionary of deprecated options.
            For options being deprecated without replacement the value should be None,
            for options that are being renamed, the value should be the new option name.
        last_version: The Qiskit Experiment version that this class is removed.
        msg: Extra message, for example, to indicate alternative approach.
        stacklevel: Stacklevel of this warning. See Python Warnings documentation for details.

    Examples:

        .. code-block::

            class SomeExperiment(BaseExperiment):
                @deprecated_options(
                    options_map={"key1": "new_key1", "key2": None},
                    last_version="0.3",
                )
                def set_experiment_options(self, **fields):
                    super().set_experiment_options()

    Returns:
        Deprecated set options method.
    """

    def update_signature(set_options_method):
        @functools.wraps(set_options_method)
        def _wrap(self, **fields):
            deprecated = options_map.keys() & fields.keys()
            if any(deprecated):
                all_options = ", ".join(f"'{dep}'" for dep in deprecated)
                message = f"Options {all_options} have been deprecated and "
                if last_version:
                    message += f"will be removed after Qiskit Experiments {last_version}. "
                else:
                    message += "will be removed in future release. "
                clsname = set_options_method.__qualname__.split(".")[0]
                message += (
                    f"If this is a loaded '{clsname}' class instance, "
                    "please save the experiment again for further retrieval with future software. "
                )
                if msg:
                    message += msg
                warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)
                for dep in deprecated:
                    new_opt = options_map[dep]
                    if new_opt is not None:
                        warnings.warn(
                            f"Option '{dep}' is now replaced with '{new_opt}'.",
                            UserWarning,
                            stacklevel=stacklevel,
                        )
                        fields[new_opt] = fields[dep]
                    del fields[dep]
            set_options_method(self, **fields)

        return _wrap

    return update_signature


def deprecated_init_args(
    arguments_map: Dict[str, Union[str, None]],
    last_version: Optional[str] = None,
    msg: Optional[str] = None,
    stacklevel: int = 2,
) -> Callable:
    """A class decorator to show deprecation warnings for old constructor arguments and
    patch __init__ method of the class to remove deprecated arguments.

    This also overrides instance ``__init_kwargs__`` so that re-saved instance can
    be instantiated without warnings.

    Args:
        arguments_map: A dictionary of deprecated arguments.
            For arguments being deprecated without replacement the value should be None,
            for arguments that are being renamed, the value should be the new option name.
        last_version: The Qiskit Experiment version that this class is removed.
        msg: Extra message, for example, to indicate alternative approach.
        stacklevel: Stacklevel of this warning. See Python Warnings documentation for details.

    Examples:

        .. code-block::

            @deprecated_init_args(
                arguments_map={"opt1": "new_opt1", "opt2": None},
                last_version="0.3",
            )
            class SomeExperiment(BaseExperiment):
                def __init__(self, qubit, new_opt1, backend):
                    super().__init__([qubit], backend=backend)

    Returns:
        Experiment class with deprecated constructor arguments.
    """

    def patch_init(cls) -> Type:
        cls_init = getattr(cls, "__init__")

        @functools.wraps(cls.__init__, assigned=("__annotations__",))
        def init(self, *args, **kwargs):
            deprecated_args = arguments_map.keys() & kwargs.keys()
            if any(deprecated_args):
                all_options = ", ".join(f"'{dep}'" for dep in deprecated_args)
                message = f"Options {all_options} have been deprecated and "
                if last_version:
                    message += f"will be removed after Qiskit Experiments {last_version}. "
                else:
                    message += "will be removed in future release. "
                message += (
                    f"If this is a loaded '{cls.__name__}' class instance, "
                    "please save the experiment again for further retrieval with future software. "
                )
                if msg:
                    message += msg
                warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)
                for dep in deprecated_args:
                    new_arg = arguments_map[dep]
                    if new_arg is not None:
                        warnings.warn(
                            f"Option '{dep}' is now replaced with '{new_arg}'.",
                            UserWarning,
                            stacklevel=stacklevel,
                        )
                        kwargs[new_arg] = kwargs[dep]
                        self.__init_kwargs__[new_arg] = kwargs[dep]
                    del kwargs[dep]
                    del self.__init_kwargs__[dep]
            cls_init(self, *args, **kwargs)

        cls.__init__ = init
        return cls

    return patch_init
