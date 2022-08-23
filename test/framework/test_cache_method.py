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

"""Tests cache_method decorator."""

from test.base import QiskitExperimentsTestCase
from qiskit_experiments.framework.cache_method import cache_method


class TestCacheMethod(QiskitExperimentsTestCase):
    """Test for cache_method decorator"""

    def test_cache_args(self):
        """Test cache_args=True"""

        class CachedClass:
            """Class with cached method"""

            def __init__(self):
                self.method_calls = 0

            @cache_method(cache_args=True)
            def method(self, *args, **kwargs):
                """Test method for caching"""
                self.method_calls += 1
                return args, kwargs

        obj = CachedClass()
        size = 10
        cached_vals = [obj.method(i, i) for i in range(size)]
        for i, val in enumerate(cached_vals):
            self.assertEqual(obj.method(i, i), val, msg="method didn't return cached value")
        self.assertEqual(obj.method_calls, size, msg="Cached method was not evaluated once per arg")

    def test_cache_args_kwargs(self):
        """Test cache_args=True with args and kwargs"""

        class CachedClass:
            """Class with cached method"""

            def __init__(self):
                self.method_calls = 0

            @cache_method(cache_args=True)
            def method(self, *args, **kwargs):
                """Test method for caching"""
                self.method_calls += 1
                return args, kwargs

        obj = CachedClass()
        args = (1, 2, 3)
        names = ["a", "b", "c", "d"]
        cached_vals = [obj.method(*args, name=name) for name in names]
        for name, val in zip(names, cached_vals):
            self.assertEqual(
                obj.method(*args, name=name), val, msg="method didn't return cached value"
            )
        self.assertEqual(
            obj.method_calls, len(names), msg="Cached method was not evaluated once per arg"
        )

    def test_cache_args_false(self):
        """Test cache_args=False"""

        class CachedClass:
            """Class with cached method"""

            def __init__(self):
                self.method_calls = 0

            @cache_method(cache_args=False)
            def method(self, *args, **kwargs):
                """Test method for caching"""
                self.method_calls += 1
                return args, kwargs

        obj = CachedClass()
        ret = obj.method(1999)
        for i in range(10):
            self.assertEqual(obj.method(i), ret, msg="method didn't return cached value")
        self.assertEqual(obj.method_calls, 1, msg="Cached method was not evaluated once")

    def test_non_hashable_raises(self):
        """Test non hashable args raise"""

        class CachedClass:
            """Class with cached method"""

            @cache_method()
            def method(self, *args, **kwargs):
                """Test method for caching"""
                return args, kwargs

        obj = CachedClass()
        self.assertRaises(TypeError, obj.method, [1, 2, 3])
        self.assertRaises(TypeError, obj.method, kwarg=[1, 2, 3])

    def test_cache_name(self):
        """Test decorator with a custom cache name"""

        class CachedClass:
            """Class with cached method"""

            @cache_method(cache="memory")
            def method(self, *args, **kwargs):
                """Test method for caching"""
                return args, kwargs

        obj = CachedClass()
        obj.method(1, 2, 3)
        self.assertTrue(hasattr(obj, "memory"))
        self.assertIn("method", getattr(obj, "memory", {}))

    def test_cache_dict(self):
        """Test decorate with custom cache value"""

        external_cache = {}

        class CachedClass:
            """Class with cached method"""

            @cache_method(cache=external_cache)
            def method(self, *args, **kwargs):
                """Test method for caching"""
                return args, kwargs

        obj = CachedClass()
        obj.method(1, 2, 3)
        self.assertIn("method", external_cache)
