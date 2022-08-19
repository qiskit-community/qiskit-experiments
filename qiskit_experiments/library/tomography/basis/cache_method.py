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

from typing import Union, Dict, Callable
import functools


def cache_method(cache: Union[Dict, str] = "_cache") -> Callable:
    """Decorator for caching class instance methods.

    Args:
        cache: The cache or cache attribute name to use. If a dict it will
               be used directly, if a str a cache dict will be created under
               that attribute name if one is not already present.

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

        def _cache_key(*args, **kwargs):
            return args + tuple(list(kwargs.items()))

        if isinstance(cache, str):

            def _get_cache(self):
                if not hasattr(self, cache):
                    setattr(self, cache, {})
                return getattr(self, cache)

        else:

            def _get_cache(_):
                return cache

        @functools.wraps(method)
        def _cached_method(self, *args, **kwargs):
            _cache = _get_cache(self)

            name = method.__name__
            if name not in _cache:
                _cache[name] = {}
            meth_cache = _cache[name]

            key = _cache_key(*args, **kwargs)
            if key in meth_cache:
                return meth_cache[key]
            result = method(self, *args, **kwargs)
            meth_cache[key] = result
            return result

        return _cached_method

    return cache_method_decorator
