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

from typing import Callable, Optional, Type, Dict


def deprecated_function(
    version_removed: Optional[str] = None,
    use_instead: Optional[str] = None,
) -> Callable:
    """A function decorator to show deprecation warning.

    Args:
        version_removed: The Qiskit Experiment version that this function is removed.
        use_instead: Alternative function.

    Examples:

        .. code-block::

            @deprecated_function(version_removed="0.3", use_instead="use new_function")
            def old_function(*args, **kwargs):
                pass

    Returns:
        Deprecated function.
    """
    def deprecated_wrapper(func: Callable):
        @functools.wraps(func)
        def _wrap(*args, **kwargs):
            message = f"The function '{func.__name__}' has been deprecated and "
            if version_removed:
                message += f"will be removed in Qiskit Experiments {version_removed}. "
            else:
                message += "will be removed in future release. "
            if use_instead:
                message += f"Please '{use_instead}' instead. "
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return _wrap
    return deprecated_wrapper


def deprecated_method(
    version_removed: Optional[str] = None,
    use_instead: Optional[str] = None,
) -> Callable:
    """A method decorator to show deprecation warning.

    Args:
        version_removed: The Qiskit Experiment version that this method is removed.
        use_instead: Alternative method.

    Examples:

        .. code-block::

            class SomeClass:
                @deprecated_method(version_removed="0.3", use_instead="use new_method")
                def old_method(self, *args, **kwargs):
                    pass

                def new_method(self, *args, **kwargs):
                    pass

    Returns:
        Deprecated method.
    """
    def deprecated_wrapper(method: Callable):
        @functools.wraps(method)
        def _wrap(self, *args, **kwargs):
            clsname = method.__qualname__.split(".")[0]
            message = f"Calling the method '{method.__name__}' of '{clsname}' has been deprecated and "
            if version_removed:
                message += f"will be removed in Qiskit Experiments {version_removed}. "
            else:
                message += "will be removed in future release. "
            if use_instead:
                message += f"Please '{use_instead}' instead. "
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return method(self, *args, **kwargs)
        return _wrap
    return deprecated_wrapper


def deprecated_clsmethod(
    version_removed: Optional[str] = None,
    use_instead: Optional[str] = None,
) -> Callable:
    """A class method decorator to show deprecation warning.

    Args:
        version_removed: The Qiskit Experiment version that this class method is removed.
        use_instead: Alternative method.

    Examples:

        .. code-block::

            class SomeClass:
                @classmethod
                @deprecated_method(version_removed="0.3", use_instead="use new_method")
                def old_method(self, *args, **kwargs):
                    pass

                @classmethod
                def new_method(self, *args, **kwargs):
                    pass

    Returns:
        Deprecated method.
    """
    def deprecated_wrapper(method: Callable):
        @functools.wraps(method)
        def _wrap(cls, *args, **kwargs):
            clsname = cls.__name__
            message = f"Calling the class method '{method.__name__}' of '{clsname}' has been deprecated and "
            if version_removed:
                message += f"will be removed in Qiskit Experiments {version_removed}. "
            else:
                message += "will be removed in future release. "
            if use_instead:
                message += f"Please '{use_instead}' instead. "
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return method(cls, *args, **kwargs)
        return _wrap
    return deprecated_wrapper


def deprecated_class(
    version_removed: Optional[str] = None,
    new_cls: Optional[Type] = None,
) -> Callable:
    """A class decorator to show deprecation warning and
    patch __new__ method of the class to instantiate the new class.

    Args:
        version_removed: The Qiskit Experiment version that this class is removed.
        new_cls: Alternative class type.

    Examples:

        .. code-block::

            @deprecated_class(version_removed="0.3", new_cls=NewCls)
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
            if version_removed:
                message += f"This class will be removed in Qiskit Experiments {version_removed}. "
            else:
                message += "This class will be removed in future release. "
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            instance = object.__new__(new_cls or deprecated_cls)
            instance.__init__(*args, **kwargs)
            return instance
        cls.__new__ = new
        return cls

    return patch_new


def deprecated_options(
    options_map: Dict[str, str],
    version_removed: Optional[str] = None,
) -> Callable:
    """A set options method decorator to show deprecation warning for deprecated options.

    Args:
        options_map: A dictionary of old option to new option. If new option value is ``None``
            it simply removes the option from set options method.
        version_removed: The Qiskit Experiment version that this class is removed.

    Examples:

        .. code-block::

            class SomeExperiment(BaseExperiment):
                @deprecated_options(
                    options_map={"key1": "new_key1", "key2": None},
                    version_removed="0.3",
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
                if version_removed:
                    message += f"will be removed in Qiskit Experiments {version_removed}. "
                else:
                    message += "will be removed in future release. "
                clsname = set_options_method.__qualname__.split(".")[0]
                message += f"If this is a loaded '{clsname}' class instance, "
                message += "please save the experiment again for further retrieval with future software."
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                for dep in deprecated:
                    new_opt = options_map[dep]
                    if new_opt is not None:
                        warnings.warn(
                            f"Option '{dep}' is now replaced with '{new_opt}'.",
                            UserWarning,
                            stacklevel=2,
                        )
                        fields[new_opt] = fields[dep]
                    del fields[dep]
            set_options_method(self, **fields)
        return _wrap

    return update_signature


def deprecated_constructor_signature(
    arguments_map: Dict[str, str],
    version_removed: Optional[str] = None,
) -> Callable:
    """A class decorator to show deprecation warnings for old constructor arguments and
    patch __init__ method of the class to remove deprecated arguments.

    This also overrides instance ``__init_kwargs__`` so that re-saved instance can
    be instantiated without warnings.

    Args:
        arguments_map: A dictionary of old argument to new argument.
            If new argument value is ``None`` it simply removes the argument from the **kwargs.
        version_removed: The Qiskit Experiment version that this class is removed.

    Examples:

        .. code-block::

            @deprecated_constructor_signature(
                arguments_map={"opt1": "new_opt1", "opt2": None},
                version_removed="0.3",
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
                if version_removed:
                    message += f"will be removed in Qiskit Experiments {version_removed}. "
                else:
                    message += "will be removed in future release. "
                message += f"If this is a loaded '{cls.__name__}' class instance, "
                message += "please save the experiment again for further retrieval with future software."
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                for dep in deprecated_args:
                    new_arg = arguments_map[dep]
                    if new_arg is not None:
                        warnings.warn(
                            f"Option '{dep}' is now replaced with '{new_arg}'.",
                            UserWarning,
                            stacklevel=2,
                        )
                        kwargs[new_arg] = kwargs[dep]
                        self.__init_kwargs__[new_arg] = kwargs[dep]
                    del kwargs[dep]
                    del self.__init_kwargs__[dep]
            cls_init(self, *args, **kwargs)
        cls.__init__ = init
        return cls

    return patch_init
