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

"""Fake backend class for tests."""

from qiskit.exceptions import QiskitError
from qiskit.providers.fake_provider import FakeArmonkV2, FakeAthensV2, FakeBelemV2
from qiskit.providers.models import PulseDefaults
from qiskit.providers.fake_provider.utils.json_decoder import decode_pulse_defaults


class PulseDefaultsMixin:
    """Mixin class to add defaults() to a fake backend

    In particular, this class works with
    ``qiskit.providers.fake_provider.fake_backend.FakeBackendV2`` classes that
    have a ``defs_filename`` attribute with the path to a pulse defaults json
    file.
    """
    _defaults = None

    def defaults(self):
        """Returns a snapshot of device defaults"""
        if not self._defaults:
            self._set_defaults_from_json()
        return self._defaults

    def _set_defaults_from_json(self):
        if not self.props_filename:
            raise QiskitError("No properties file has been defined")
        defs = self._load_json(self.defs_filename)
        decode_pulse_defaults(defs)
        self._defaults = PulseDefaults.from_dict(defs)


class FakeArmonkV2Pulse(FakeArmonkV2, PulseDefaultsMixin):
    """FakeArmonkV2 with pulse defaults"""


class FakeAthensV2Pulse(FakeAthensV2, PulseDefaultsMixin):
    """FakeAthensV2 with pulse defaults"""


class FakeBelemV2Pulse(FakeBelemV2, PulseDefaultsMixin):
    """FakeBelemV2 with pulse defaults"""
