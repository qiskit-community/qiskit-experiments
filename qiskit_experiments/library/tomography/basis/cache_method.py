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
"""
Method decorator for caching regular methods in class instances.
"""

from typing import Dict, Callable, Optional
import functools


def _method_cache_name(instance: any) -> str:
    """Attribute name for storing cache in an instance"""
    return "_CACHE_" + type(instance).__name__


def _get_method_cache(instance: any) -> Dict:
    """Return instance cache for cached methods"""
    cache_name = _method_cache_name(instance)
    try:
        return getattr(instance, cache_name)
    except AttributeError:
        setattr(instance, cache_name, {})
        return getattr(instance, cache_name)


def cache_method(maxsize: Optional[int] = None) -> Callable:
    """Decorator for caching class instance methods.

    Args:
        maxsize: The maximum size of this method's LRU cache.

    Returns:
        The decorator for caching methods.
    """

    def cache_method_decorator(method: Callable) -> Callable:
        """Decorator for caching method.

        Args:
            method: A method to cache.

        Returns:
            The wrapped cached method.
        """

        @functools.wraps(method)
        def _cached_method(self, *args, **kwargs):
            cache = _get_method_cache(self)
            key = method.__name__
            try:
                meth = cache[key]
            except KeyError:
                meth = cache[key] = functools.lru_cache(maxsize)(functools.partial(method, self))

            return meth(*args, **kwargs)

        return _cached_method

    return cache_method_decorator
