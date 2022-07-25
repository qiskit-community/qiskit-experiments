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

class BackendData():
    @staticmethod
    def name(backend):
        if isinstance(backend, BackendV1):
            return backend.name()
        elif isinstance(backend, BackendV2):
            return backend.name
        return ""

    @staticmethod
    def control_channel(backend, qubits):
        if isinstance(backend, BackendV1):
            return backend.configuration().control(qubits)
        elif isinstance(backend, BackendV2):
            return backend.control_channel(qubits)
        return None

    @staticmethod
    def granularity(backend):
        if isinstance(backend, BackendV1):
            return backend.configuration().timing_constraints.get("granularity", None)
        elif isinstance(backend, BackendV2):
            return backend.target.granularity
        return None

    @staticmethod
    def set_granularity(backend, value):
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
        if isinstance(backend, BackendV1):
            return backend.configuration().dt
        elif isinstance(backend, BackendV2):
            return backend.dt
        return None

    @staticmethod
    def max_experiments(backend):
        if isinstance(backend, BackendV1):
            return getattr(backend.configuration(), "max_experiments", None)
        elif isinstance(backend, BackendV2):
            return backend.max_circuits
        return None

    @staticmethod
    def coupling_map(backend):
        if isinstance(backend, BackendV1):
            return getattr(backend.configuration(), "coupling_map", [])
        elif isinstance(backend, BackendV2):
            return backend.coupling_map.get_edges()
        return []

    @staticmethod
    def control_channels(backend):
        if isinstance(backend, BackendV1):
            return getattr(backend.configuration(), "control_channels", None)
        elif isinstance(backend, BackendV2):
            return backend.control_channels
        return None

    @staticmethod
    def version(backend):
        if isinstance(backend, BackendV1):
            return getattr(backend, "version", None)
        elif isinstance(backend, BackendV2):
            return backend.version
        return None

    @staticmethod
    def qubit_freq_est(backend):
        if isinstance(backend, BackendV1):
            return getattr(backend.defaults(), "qubit_freq_est", [])
        elif isinstance(backend, BackendV2):
            return  [property.frequency for property in
                      backend.target.qubit_properties]
        return []

    @staticmethod
    def meas_freq_est(backend):
        if isinstance(backend, BackendV1):
            return getattr(backend.defaults(), "meas_freq_est", [])
        elif isinstance(backend, BackendV2):
            # meas_freq_est is currently not part of the BackendV2
            return []
        return []

