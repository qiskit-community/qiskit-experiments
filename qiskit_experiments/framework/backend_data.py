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
from qiskit.providers import BackendV1, BackendV2
from qiskit.providers.fake_provider import fake_backend, FakeBackendV2, FakeBackend

class BackendData:
    """Class for providing joint interface for accessing backend data"""

    @staticmethod
    def name(backend):
        """Returns the backend name"""
        if isinstance(backend, BackendV1):
            return backend.name()
        elif isinstance(backend, BackendV2):
            return backend.name
        return ""

    @staticmethod
    def control_channel(backend, qubits):
        """Returns the backend control channels"""
        if isinstance(backend, BackendV1):
            return backend.configuration().control(qubits)
        elif isinstance(backend, BackendV2):
            return backend.control_channel(qubits)
        return None

    @staticmethod
    def granularity(backend):
        """Returns the backend's time constraint granularity"""
        if isinstance(backend, BackendV1):
            return backend.configuration().timing_constraints.get("granularity", None)
        elif isinstance(backend, BackendV2):
            return backend.target.granularity
        return None

    @staticmethod
    def set_granularity(backend, value):
        """Sets the backend's time constraint granularity"""
        if isinstance(backend, BackendV1):
            setattr(
                backend.configuration(),
                "timing_constraints",
                {"granularity": value},
            )
        elif isinstance(backend, BackendV2):
            backend.target.granularity = 16

    @staticmethod
    def dt(backend):
        """Returns the backend's input time resolution"""
        if isinstance(backend, BackendV1):
            try:
                return backend.configuration().dt
            except AttributeError:
                return None
        elif isinstance(backend, BackendV2):
            return backend.dt
        return None

    @staticmethod
    def max_experiments(backend):
        """Returns the backend's max experiments value"""
        if isinstance(backend, BackendV1):
            return getattr(backend.configuration(), "max_experiments", None)
        elif isinstance(backend, BackendV2):
            return backend.max_circuits
        return None

    @staticmethod
    def coupling_map(backend):
        """Returns the backend's coupling map"""
        if isinstance(backend, BackendV1):
            return getattr(backend.configuration(), "coupling_map", [])
        elif isinstance(backend, BackendV2):
            return list(backend.coupling_map.get_edges())
        return []

    @staticmethod
    def control_channels(backend):
        """Returns the backend's control channels"""
        if isinstance(backend, BackendV1):
            return getattr(backend.configuration(), "control_channels", None)
        elif isinstance(backend, BackendV2):
            return backend.control_channels
        return None

    @staticmethod
    def version(backend):
        """Returns the backend's version"""
        if isinstance(backend, BackendV1):
            return getattr(backend, "version", None)
        elif isinstance(backend, BackendV2):
            return backend.version
        return None

    @staticmethod
    def provider(backend):
        """Returns the backend's provider"""
        if isinstance(backend, BackendV1):
            return getattr(backend, "provider", None)
        elif isinstance(backend, BackendV2):
            return backend.provider
        return None

    @staticmethod
    def qubit_freq_est(backend):
        """Returns the backend's qubit frequency estimation"""
        if isinstance(backend, BackendV1):
            return getattr(backend.defaults(), "qubit_freq_est", [])
        elif isinstance(backend, BackendV2):
            return [property.frequency for property in backend.target.qubit_properties]
        return []

    @staticmethod
    def meas_freq_est(backend):
        """Returns the backend's measurement frequency estimation.
        Note: currently BackendV2 does not have access to this data"""
        if isinstance(backend, BackendV1):
            return getattr(backend.defaults(), "meas_freq_est", [])
        elif isinstance(backend, BackendV2):
            # meas_freq_est is currently not part of the BackendV2
            return []
        return []

    @staticmethod
    def num_qubits(backend):
        """Returns the backend's number of qubits"""
        if isinstance(backend, BackendV1):
            return backend.configuration().num_qubits
        elif isinstance(backend, BackendV2):
            # meas_freq_est is currently not part of the BackendV2
            return backend.num_qubits
        return None

    @staticmethod
    def is_simulator(backend):
        """Returns True given an indication the backend is a simulator
        Note: for `BackendV2` we sometimes cannot be sure"""
        if isinstance(backend, BackendV1):
            if backend.configuration().simulator or isinstance(
            backend, FakeBackend):
                return True
        if isinstance(backend, BackendV2):
            if isinstance(backend, FakeBackendV2) or isinstance(backend, fake_backend.FakeBackendV2):
                return True
        return False
