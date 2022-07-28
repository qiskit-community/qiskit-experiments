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
"""
Backend data access helper class

Since `BackendV1` and `BackendV2` do not share the same interface, this
class unifies data access for various data fields.
"""
from qiskit.providers.models import PulseBackendConfiguration
from qiskit.providers import BackendV1, BackendV2
from qiskit.providers.fake_provider import fake_backend, FakeBackendV2, FakeBackend


class BackendData:
    """Class for providing joint interface for accessing backend data"""

    def __init__(self, backend):
        """Inits the backend and verifies version"""
        self._backend = backend
        self._v1 = isinstance(backend, BackendV1)
        self._v2 = isinstance(backend, BackendV2)
        if self._v2:
            self._parse_additional_data()

    def _parse_additional_data(self):
        # data specific parsing not done yet in qiskit-terra
        if self._backend._conf_dict["open_pulse"]:
            if "u_channel_lo" not in self._backend._conf_dict:
                self._backend._conf_dict["u_channel_lo"] = []  # to avoid terra bug
            self._pulse_conf = PulseBackendConfiguration.from_dict(self._backend._conf_dict)

    @property
    def name(self):
        """Returns the backend name"""
        if self._v1:
            return self._backend.name()
        elif self._v2:
            return self._backend.name
        return str(self._backend)

    def control_channel(self, qubits):
        """Returns the backend control channels"""
        if self._v1:
            return self._backend.configuration().control(qubits)
        elif self._v2:
            try:
                return self._backend.control_channel(qubits)
            except (AttributeError, NotImplementedError):
                return self._pulse_conf.control_channels[qubits]
        return None

    @property
    def granularity(self):
        """Returns the backend's time constraint granularity"""
        try:
            if self._v1:
                return self._backend.configuration().timing_constraints.get("granularity", 1)
            elif self._v2:
                return self._backend.target.granularity
        except AttributeError:
            return 1
        return 1

    @property
    def min_length(self):
        """Returns the backend's time constraint minimum duration"""
        try:
            if self._v1:
                return self._backend.configuration().timing_constraints.get("min_length", 0)
            elif self._v2:
                return self._backend.target.min_length
        except AttributeError:
            return 0
        return 0

    @property
    def pulse_alignment(self):
        """Returns the backend's time constraint pulse alignment"""
        try:
            if self._v1:
                return self._backend.configuration().timing_constraints.get("pulse_alignment", 1)
            elif self._v2:
                return self._backend.target.pulse_alignment
        except AttributeError:
            return 1
        return 1

    @property
    def aquire_alignment(self):
        """Returns the backend's time constraint acquire alignment"""
        try:
            if self._v1:
                return self._backend.configuration().timing_constraints.get("aquire_alignment", 1)
            elif self._v2:
                return self._backend.target.aquire_alignment
        except AttributeError:
            return 1
        return 1

    @property
    def dt(self):
        """Returns the backend's input time resolution"""
        if self._v1:
            try:
                return self._backend.configuration().dt
            except AttributeError:
                return None
        elif self._v2:
            return self._backend.dt
        return None

    @property
    def max_circuits(self):
        """Returns the backend's max experiments value"""
        if self._v1:
            return getattr(self._backend.configuration(), "max_experiments", None)
        elif self._v2:
            return self._backend.max_circuits
        return None

    @property
    def coupling_map(self):
        """Returns the backend's coupling map"""
        if self._v1:
            return getattr(self._backend.configuration(), "coupling_map", [])
        elif self._v2:
            return list(self._backend.coupling_map.get_edges())
        return []

    @property
    def control_channels(self):
        """Returns the backend's control channels"""
        if self._v1:
            return getattr(self._backend.configuration(), "control_channels", None)
        elif self._v2:
            try:
                return self._backend.control_channels
            except AttributeError:
                return self._pulse_conf.control_channels
        return None

    @property
    def version(self):
        """Returns the backend's version"""
        if self._v1:
            return getattr(self._backend, "version", None)
        elif self._v2:
            return self._backend.version
        return None

    @property
    def provider(self):
        """Returns the backend's provider"""
        if self._v1:
            return getattr(self._backend, "provider", None)
        elif self._v2:
            return self._backend.provider
        return None

    @property
    def drive_freqs(self):
        """Returns the backend's qubit frequency estimation"""
        if self._v1:
            return getattr(self._backend.defaults(), "qubit_freq_est", [])
        elif self._v2:
            return [property.frequency for property in self._backend.target.qubit_properties]
        return []

    @property
    def meas_freqs(self):
        """Returns the backend's measurement frequency estimation.
        Note: currently BackendV2 does not have access to this data"""
        if self._v1:
            return getattr(self._backend.defaults(), "meas_freq_est", [])
        elif self._v2:
            # meas_freq_est is currently not part of the BackendV2
            return []
        return []

    @property
    def num_qubits(self):
        """Returns the backend's number of qubits"""
        if self._v1:
            return self._backend.configuration().num_qubits
        elif self._v2:
            # meas_freq_est is currently not part of the BackendV2
            return self._backend.num_qubits
        return None

    @property
    def is_simulator(self):
        """Returns True given an indication the backend is a simulator
        Note: for `BackendV2` we sometimes cannot be sure"""
        if self._v1:
            if self._backend.configuration().simulator or isinstance(self._backend, FakeBackend):
                return True
        if self._v2:
            if isinstance(self._backend, (FakeBackendV2, fake_backend.FakeBackendV2)):
                return True

        return False
