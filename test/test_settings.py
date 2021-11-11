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


class BaseSettingsTest(Settings):
    """Test class with args and kwargs property"""

    @property
    def args(self):
        """Return sotred init args"""
        return tuple(getattr(self, "__init_args__", {}).values())

    @property
    def kwargs(self):
        """Return stored init kwargs"""
        return dict(getattr(self, "__init_kwargs__", {}))


class SettingsStandard(BaseSettingsTest):
    """Settings test class with standard args and kwargs"""

    def __init__(self, a, b, c=1, d=2):
        pass


class SettingsVariadicKwargs(BaseSettingsTest):
    """Settings test class with variadic kwargs"""

    def __init__(self, a, b, c=1, d=2, **kwargs):
        pass


class SettingsVariadicArgs(BaseSettingsTest):
    """Settings test class with variadic args and kwargs"""

    def __init__(self, a, b, *args, c=1, d=2, **kwargs):
        pass


class TestSettings(QiskitTestCase):
    """Test Settings mixin"""

    def test_standard(self):
        """Test mixing for standard init class"""
        obj = SettingsStandard(1, 2, c=10)
        self.assertEqual(obj.args, (1, 2))
        self.assertEqual(obj.kwargs, {"c": 10})
        self.assertEqual(obj.settings, {"a": 1, "b": 2, "c": 10})

    def test_standard_pos_kwargs(self):
        """Test mixing for standard init class with kwargs passed positionally"""
        obj = SettingsStandard(1, 2, 10)
        self.assertEqual(obj.args, (1, 2))
        self.assertEqual(obj.kwargs, {"c": 10})
        self.assertEqual(obj.settings, {"a": 1, "b": 2, "c": 10})

    def test_standard_named_args(self):
        """Test mixing for standard init class with kwargs passed positionally"""
        obj = SettingsStandard(b=2, a=1, c=10)
        self.assertEqual(obj.args, (1, 2))
        self.assertEqual(obj.kwargs, {"c": 10})
        self.assertEqual(obj.settings, {"a": 1, "b": 2, "c": 10})

    def test_variadic_kwargs(self):
        """Test mixing for standard init class"""
        obj = SettingsVariadicKwargs(1, 2, c=10, f=20, g=30)
        self.assertEqual(obj.args, (1, 2))
        self.assertEqual(obj.kwargs, {"c": 10, "f": 20, "g": 30})
        self.assertEqual(obj.settings, {"a": 1, "b": 2, "c": 10, "f": 20, "g": 30})

    def test_variadic_kwargs_pos_kwargs(self):
        """Test mixing for standard init class"""
        obj = SettingsVariadicKwargs(1, 2, 10, f=20, g=30)
        self.assertEqual(obj.args, (1, 2))
        self.assertEqual(obj.kwargs, {"c": 10, "f": 20, "g": 30})
        self.assertEqual(obj.settings, {"a": 1, "b": 2, "c": 10, "f": 20, "g": 30})

    def test_variadic_kwargs_named_args(self):
        """Test mixing for standard init class"""
        obj = SettingsVariadicKwargs(b=2, a=1, c=10, f=20, g=30)
        self.assertEqual(obj.args, (1, 2))
        self.assertEqual(obj.kwargs, {"c": 10, "f": 20, "g": 30})
        self.assertEqual(obj.settings, {"a": 1, "b": 2, "c": 10, "f": 20, "g": 30})

    def test_variadic_args(self):
        """Test mixing for standard init class"""
        obj = SettingsVariadicArgs(1, 2, 3, 4, c=10, f=20, g=30)
        self.assertEqual(obj.args, (1, 2, 3, 4))
        self.assertEqual(obj.kwargs, {"c": 10, "f": 20, "g": 30})
        self.assertEqual(
            obj.settings, {"a": 1, "b": 2, "args0": 3, "args1": 4, "c": 10, "f": 20, "g": 30}
        )
