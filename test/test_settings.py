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

"""Tests for base experiment framework."""

from qiskit.test import QiskitTestCase
from qiskit_experiments.framework.settings import Settings


class ExampleStoreArgsVariadic(Settings):
    """Test class with args and kwargs property"""

    def __init__(self, a, *args, b, c="default_c", d="default_d", **kwargs):
        pass

    @property
    def args(self):
        """Return stored init args"""
        return tuple(getattr(self, "__init_args__", {}).values())

    @property
    def kwargs(self):
        """Return stored init kwargs"""
        return dict(getattr(self, "__init_kwargs__", {}))


class ExampleSettingsVariadic(Settings):
    """Test class with args and kwargs property"""

    def __init__(self, a, b, c="default_c", d="default_d", **kwargs):
        pass

    @property
    def args(self):
        """Return stored init args"""
        return tuple(getattr(self, "__init_args__", {}).values())

    @property
    def kwargs(self):
        """Return stored init kwargs"""
        return dict(getattr(self, "__init_kwargs__", {}))


class ExampleSettings(Settings):
    """Test class with args and kwargs property"""

    def __init__(self, a, b, c="default_c", d="default_d"):
        pass

    @property
    def args(self):
        """Return stored init args"""
        return tuple(getattr(self, "__init_args__", {}).values())

    @property
    def kwargs(self):
        """Return stored init kwargs"""
        return dict(getattr(self, "__init_kwargs__", {}))


class TestSettings(QiskitTestCase):
    """Test Settings mixin"""

    # pylint: disable = missing-function-docstring

    def test_standard(self):
        obj = ExampleSettings(1, 2, c="custom_c")
        self.assertEqual(obj.args, (1, 2))
        self.assertEqual(obj.kwargs, {"c": "custom_c", "d": "default_d"})
        self.assertEqual(obj.settings, {"a": 1, "b": 2, "c": "custom_c", "d": "default_d"})

    def test_standard_pos_kwargs(self):
        obj = ExampleSettings(1, 2, "custom_c")
        self.assertEqual(obj.args, (1, 2))
        self.assertEqual(obj.kwargs, {"c": "custom_c", "d": "default_d"})
        self.assertEqual(obj.settings, {"a": 1, "b": 2, "c": "custom_c", "d": "default_d"})

    def test_standard_named_args(self):
        obj = ExampleSettings(b=2, a=1, c="custom_c")
        self.assertEqual(obj.args, (1, 2))
        self.assertEqual(obj.kwargs, {"c": "custom_c", "d": "default_d"})
        self.assertEqual(obj.settings, {"a": 1, "b": 2, "c": "custom_c", "d": "default_d"})

    def test_variadic(self):
        obj = ExampleSettingsVariadic(1, 2, c="custom_c")
        self.assertEqual(obj.args, (1, 2))
        self.assertEqual(obj.kwargs, {"c": "custom_c", "d": "default_d"})
        self.assertEqual(obj.settings, {"a": 1, "b": 2, "c": "custom_c", "d": "default_d"})

    def test_variadic_pos_kwargs(self):
        obj = ExampleSettingsVariadic(1, 2, "custom_c")
        self.assertEqual(obj.args, (1, 2))
        self.assertEqual(obj.kwargs, {"c": "custom_c", "d": "default_d"})
        self.assertEqual(obj.settings, {"a": 1, "b": 2, "c": "custom_c", "d": "default_d"})

    def test_variadic_named_args(self):
        obj = ExampleSettingsVariadic(b=2, a=1, c="custom_c")
        self.assertEqual(obj.args, (1, 2))
        self.assertEqual(obj.kwargs, {"c": "custom_c", "d": "default_d"})
        self.assertEqual(obj.settings, {"a": 1, "b": 2, "c": "custom_c", "d": "default_d"})

    def test_variadic_kwargs(self):
        obj = ExampleSettingsVariadic(1, 2, d="custom_d", f="kwarg_f", g="kwarg_g")
        self.assertEqual(obj.args, (1, 2))
        self.assertEqual(
            obj.kwargs, {"c": "default_c", "d": "custom_d", "f": "kwarg_f", "g": "kwarg_g"}
        )
        self.assertEqual(
            obj.settings,
            {"a": 1, "b": 2, "c": "default_c", "d": "custom_d", "f": "kwarg_f", "g": "kwarg_g"},
        )

    def test_variadic_kwargs_pos_kwargs(self):
        obj = ExampleSettingsVariadic(1, 2, "custom_c", f="kwarg_f", g="kwarg_g")
        self.assertEqual(obj.args, (1, 2))
        self.assertEqual(
            obj.kwargs, {"c": "custom_c", "d": "default_d", "f": "kwarg_f", "g": "kwarg_g"}
        )
        self.assertEqual(
            obj.settings,
            {"a": 1, "b": 2, "c": "custom_c", "d": "default_d", "f": "kwarg_f", "g": "kwarg_g"},
        )

    def test_variadic_kwargs_named_args(self):
        obj = ExampleSettingsVariadic(b=2, a=1, d="custom_d", f="kwarg_f", g="kwarg_g")
        self.assertEqual(obj.args, (1, 2))
        self.assertEqual(
            obj.kwargs, {"c": "default_c", "d": "custom_d", "f": "kwarg_f", "g": "kwarg_g"}
        )
        self.assertEqual(
            obj.settings,
            {"a": 1, "b": 2, "c": "default_c", "d": "custom_d", "f": "kwarg_f", "g": "kwarg_g"},
        )

    def test_variadic_args(self):
        obj = ExampleStoreArgsVariadic(1, 2, b="custom_b", c="custom_c", f="kwarg_f", g="kwarg_g")
        self.assertEqual(obj.args, (1, 2))
        self.assertEqual(
            obj.kwargs,
            {"b": "custom_b", "c": "custom_c", "d": "default_d", "f": "kwarg_f", "g": "kwarg_g"},
        )
