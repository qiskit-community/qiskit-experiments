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


def cache_method(cache: Union[Dict, str] = "_cache", cache_args: bool = True) -> Callable:
    """Decorator for caching regular methods in classes.

    .. note::

        When specifying a cache an existing dictionary value will be
        used as is. A string value will be used to check for an existing
        dict under that attribute name in the class instance.
        If the attribute is not present a new cache dict will be created
        and stored in that class instance.

    Args:
        cache: A dictionary or attribute name string to use as cache.
        cache_args: If True include method arg and kwarg values when
                    matching cached values. These values must be hashable.

    Returns:
        The decorator for caching methods.
    """
    cache_fn = _cache_function(cache)
    cache_key_fn = _cache_key_function(cache_args)

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


def _cache_key_function(cache_args: bool) -> Callable:
    """Return function for generating cache keys.

    Args:
        cache_args: If True include method arg and kwarg values when
                    caching the method. If False all calls to the instances
                    method will return the same cached value regardless of
                    any arg or kwarg values.

    Returns:
        The functions for generating cache keys.
    """
    if not cache_args:

        def _cache_key(*args, **kwargs):
            # pylint: disable = unused-argument
            return tuple()

    else:

        def _cache_key(*args, **kwargs):
            return args + tuple(list(kwargs.items()))

    return _cache_key


def _cache_function(cache: Union[Dict, str]) -> Callable:
    """Return function for initializing and accessing cache dict.

    Args:
        cache: The dictionary or cache attribute name to use. If a dict it
               will be used directly, if a str a cache dict will be created
               under that attribute name if one is not already present.

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
