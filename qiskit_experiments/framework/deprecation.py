# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Deprecation utilities"""

from __future__ import annotations

import functools
import os
import sys
import warnings
from collections.abc import Callable
from typing import Type


def warn_from_qe(message: str, category: Warning, stacklevel: int = 2):
    """Issue a warning, blaming the code calling Qiskit Experiments if possible

    For Python 3.12+, this function uses ``skip_file_prefixes`` to blame the
    function outside of Qiskit Experiments that is calling a Qiskit Experiments
    function leading to the warning being issued. For older Python versions,
    the ``stacklevel`` is used.

    Args:
        message: Warning message to emit
        category: Warning category to use
        stacklevel: The stacklevel to use for the warning, relative to the
            caller of this function. This option is only used for Python 3.11
            and earlier. For Python 3.12, ``skip_file_prefixes`` is used and
            the ``stacklevel`` is fixed to blame the first caller outside of
            ``qiskit_experiments``.
    """
    # qiskit_experiments import is delayed to avoid cyclic imports
    import qiskit_experiments

    if sys.version_info[:2] >= (3, 12):
        # stacklevel is weirder than usual with skip_file_prefixes. It still
        # counts the frame of this function as stacklevel 0 even though it
        # would be filtered by the skip_file_prefixes filter, but then it
        # doesn't count anything else matched by skip_file_prefixes, so we need
        # 2 here to match the first outside caller. We do not want higher
        # because that will call higher levels of outside callers, and our goal
        # is to let the stacklevel argument try to get the right level for
        # Python 3.11 and older where it is counting internal stack levels.
        kwargs = {
            "skip_file_prefixes": (os.path.dirname(qiskit_experiments.__file__),),
            "stacklevel": 2,
        }
    else:
        # Add 1 to make up for this level
        kwargs = {"stacklevel": stacklevel + 1}
    warnings.warn(message, category, **kwargs)


def deprecate_func(
    *,
    since: str,
    additional_msg: str | None = None,
    pending: bool = False,
    package_name: str = "Qiskit Experiments",
    removal_timeline: str = "no earlier than 3 months after the release date",
    is_property: bool = False,
    stacklevel: int = 2,
):
    """Decorator to indicate a function has been deprecated.

    It should be placed beneath other decorators like `@staticmethod` and property decorators.

    When deprecating a class, set this decorator on its `__init__` function.

    Args:
        since: The version the deprecation started at. If the deprecation is pending, set
            the version to when that started; but later, when switching from pending to
            deprecated, update ``since`` to the new version.
        additional_msg: Put here any additional information, such as what to use instead.
            For example, "Instead, use the function ``new_func`` from the module
            ``<my_module>.<my_submodule>``, which is similar but uses GPU acceleration."
        pending: Set to ``True`` if the deprecation is still pending.
        package_name: The package name shown in the deprecation message (e.g. the PyPI package name).
        removal_timeline: How soon can this deprecation be removed? Expects a value
            like "no sooner than 6 months after the latest release" or "in release 9.99".
        is_property: If the deprecated function is a `@property`, set this to True so that the
            generated message correctly describes it as such. (This isn't necessary for
            property setters, as their docstring is ignored by Python.)
        stacklevel: Stack level passed to :func:`warnings.warn`.
    Returns:
        Callable: The decorated callable.
    """

    def decorator(func):
        qualname = func.__qualname__  # For methods, `qualname` includes the class name.
        mod_name = func.__module__

        # Detect what function type this is.
        if is_property:
            # `inspect.isdatadescriptor()` doesn't work because you must apply our decorator
            # before `@property`, so it looks like the function is a normal method.
            deprecated_entity = f"The property ``{mod_name}.{qualname}``"
        # To determine if's a method, we use the heuristic of looking for a `.` in the qualname.
        # This is because top-level functions will only have the function name. This is not
        # perfect, e.g. it incorrectly classifies nested/inner functions, but we don't expect
        # those to be deprecated.
        #
        # We can't use `inspect.ismethod()` because that only works when calling it on an instance
        # of the class, rather than the class type itself, i.e. `ismethod(C().foo)` vs
        # `ismethod(C.foo)`.
        elif "." in qualname:
            if func.__name__ == "__init__":
                cls_name = qualname[: -len(".__init__")]
                deprecated_entity = f"The class ``{mod_name}.{cls_name}``"
            else:
                deprecated_entity = f"The method ``{mod_name}.{qualname}()``"
        else:
            deprecated_entity = f"The function ``{mod_name}.{qualname}()``"

        msg, category = _write_deprecation_msg(
            deprecated_entity=deprecated_entity,
            package_name=package_name,
            since=since,
            pending=pending,
            additional_msg=additional_msg,
            removal_timeline=removal_timeline,
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warn_from_qe(msg, category=category, stacklevel=stacklevel)
            return func(*args, **kwargs)

        add_deprecation_to_docstring(wrapper, msg, since=since, pending=pending)
        return wrapper

    return decorator


def _write_deprecation_msg(
    *,
    deprecated_entity: str,
    package_name: str,
    since: str,
    pending: bool,
    additional_msg: str,
    removal_timeline: str,
) -> tuple[str, Type[DeprecationWarning] | Type[PendingDeprecationWarning]]:
    if pending:
        category: Type[DeprecationWarning] | Type[
            PendingDeprecationWarning
        ] = PendingDeprecationWarning
        deprecation_status = "pending deprecation"
        removal_desc = f"marked deprecated in a future release, and then removed {removal_timeline}"
    else:
        category = DeprecationWarning
        deprecation_status = "deprecated"
        removal_desc = f"removed {removal_timeline}"

    msg = (
        f"{deprecated_entity} is {deprecation_status} as of {package_name} {since}. "
        f"It will be {removal_desc}."
    )
    if additional_msg:
        msg += f" {additional_msg}"
    return msg, category


# We insert deprecations in-between the description and Napoleon's meta sections. The below is from
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#docstring-sections. We use
# lowercase because Napoleon is case-insensitive.
_NAPOLEON_META_LINES = frozenset(
    {
        "args:",
        "arguments:",
        "attention:",
        "attributes:",
        "caution:",
        "danger:",
        "error:",
        "example:",
        "examples:",
        "hint:",
        "important:",
        "keyword args:",
        "keyword arguments:",
        "note:",
        "notes:",
        "other parameters:",
        "parameters:",
        "return:",
        "returns:",
        "raises:",
        "references:",
        "see also:",
        "tip:",
        "todo:",
        "warning:",
        "warnings:",
        "warn:",
        "warns:",
        "yield:",
        "yields:",
    }
)


def add_deprecation_to_docstring(
    func: Callable, msg: str, *, since: str | None, pending: bool
) -> None:
    """Dynamically insert the deprecation message into ``func``'s docstring.

    Args:
        func: The function to modify.
        msg: The full deprecation message.
        since: The version the deprecation started at.
        pending: Is the deprecation still pending?
    """
    if "\n" in msg:
        raise ValueError(
            "Deprecation messages cannot contain new lines (`\\n`), but the deprecation for "
            f'{func.__qualname__} had them. Usually this happens when using `"""` multiline '
            f"strings; instead, use string concatenation.\n\n"
            "This is a simplification to facilitate deprecation messages being added to the "
            "documentation. If you have a compelling reason to need "
            "new lines, feel free to improve this function or open a request at "
            "https://github.com/Qiskit/qiskit/issues."
        )

    if since is None:
        version_str = "unknown"
    else:
        version_str = f"{since}_pending" if pending else since

    indent = ""
    meta_index = None
    if func.__doc__:
        original_lines = func.__doc__.splitlines()
        content_encountered = False
        for i, line in enumerate(original_lines):
            stripped = line.strip()

            # Determine the indent based on the first line with content. But, we don't consider the
            # first line, which corresponds to the format """Docstring.""", as it does not properly
            # capture the indentation of lines beneath it.
            if not content_encountered and i != 0 and stripped:
                num_leading_spaces = len(line) - len(line.lstrip())
                indent = " " * num_leading_spaces
                content_encountered = True

            if stripped.lower() in _NAPOLEON_META_LINES:
                meta_index = i
                if not content_encountered:
                    raise ValueError(
                        "add_deprecation_to_docstring cannot currently handle when a Napoleon "
                        "metadata line like 'Args' is the very first line of docstring, "
                        f'e.g. `"""Args:`. So, it cannot process {func.__qualname__}. Instead, '
                        f'move the metadata line to the second line, e.g.:\n\n"""\nArgs:'
                    )
                # We can stop checking since we only care about the first meta line, and
                # we've validated content_encountered is True to determine the indent.
                break
    else:
        original_lines = []

    # We defensively include new lines in the beginning and end. This is sometimes necessary,
    # depending on the original docstring. It is not a big deal to have extra, other than `help()`
    # being a little longer.
    new_lines = [
        indent,
        f"{indent}.. deprecated:: {version_str}",
        f"{indent}  {msg}",
        indent,
    ]

    if meta_index:
        original_lines[meta_index - 1 : meta_index - 1] = new_lines
    else:
        original_lines.extend(new_lines)
    func.__doc__ = "\n".join(original_lines)
