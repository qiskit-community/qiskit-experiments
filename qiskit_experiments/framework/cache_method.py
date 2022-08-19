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


def cache_method(
    cache: Union[Dict, str] = "_cache", cache_args: bool = True, require_hashable: bool = True
) -> Callable:
    """Decorator for caching class instance methods.

    Args:
        cache: The cache or cache attribute name to use. If a dict it will
               be used directly, if a str a cache dict will be created under
               that attribute name if one is not already present.
        cache_args: If True include method arg and kwarg values when
                    caching the method. If False only a single return will
                    be cached for the method regardless of any args.
        require_hashable: If True require all cached args and kwargs are
                          hashable. If False un-hashable values are allowed
                          but will be excluded from the cache key.

    Returns:
        The decorator for caching methods.
    """
    cache_fn = _cache_function(cache)
    cache_key_fn = _cache_key_function(cache_args, require_hashable)

    def cache_method_decorator(method: Callable) -> Callable:
        """Decorator for caching method.

        Args:
            method: A method to cache.

        Returns:
            The wrapped cached method.
        """

        @functools.wraps(method)
        def _cached_method(self, *args, **kwargs):
            meth_cache = cache_fn(self, method)
            key = cache_key_fn(*args, **kwargs)
            if key in meth_cache:
                return meth_cache[key]
            result = method(self, *args, **kwargs)
            meth_cache[key] = result
            return result

        return _cached_method

    return cache_method_decorator


def _cache_key_function(cache_args: bool, require_hashable: bool) -> Callable:
    """Return function for generating cache keys.

    Args:
        cache_args: If True include method arg and kwarg values when
                    caching the method. If False only a single return will
                    be cached for the method regardless of any args.
        require_hashable: If True require all cached args and kwargs are
                          hashable. If False un-hashable values are allowed
                          but will be excluded from the cache key.

    Returns:
        The functions for generating cache keys.
    """
    if not cache_args:

        def _cache_key(*args, **kwargs):
            # pylint: disable = unused-argument
            return tuple()

    elif require_hashable:

        def _cache_key(*args, **kwargs):
            return args + tuple(list(kwargs.items()))

    else:

        def _cache_key(*args, **kwargs):
            cache_key = tuple()
            for arg in args:
                try:
                    hash(arg)
                except TypeError:
                    continue
                cache_key += (arg,)
            for key, value in kwargs.items():
                try:
                    hash(value)
                except TypeError:
                    continue
                cache_key += ((key, value),)
            return cache_key

    return _cache_key


def _cache_function(cache: Union[Dict, str]) -> Callable:
    """Return function for initializing and accessing cache dict.

    Args:
        cache: The cache or cache attribute name to use. If a dict it will
               be used directly, if a str a cache dict will be created under
               that attribute name if one is not already present.

    Returns:
        The function for accessing the cache dict.
    """
    if isinstance(cache, str):

        def _cache_fn(instance, method):
            if not hasattr(instance, cache):
                setattr(instance, cache, {})
            instance_cache = getattr(instance, cache)
            name = method.__name__
            if name not in instance_cache:
                instance_cache[name] = {}
            return instance_cache[name]

    else:

        def _cache_fn(instance, method):
            # pylint: disable = unused-argument
            name = method.__name__
            if name not in cache:
                cache[name] = {}
            return cache[name]

    return _cache_fn
