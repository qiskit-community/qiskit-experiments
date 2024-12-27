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
import warnings
from qiskit.providers.models import PulseBackendConfiguration  # pylint: disable=no-name-in-module
from qiskit.providers import BackendV1, BackendV2


class BackendData:
    """Class for providing joint interface for accessing backend data"""

    def __init__(self, backend):
        """Inits the backend and verifies version"""

        self._backend = backend
        self._v1 = isinstance(backend, BackendV1)
        self._v2 = isinstance(backend, BackendV2)

        if self._v2:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*qiskit.qobj.pulse_qobj.*", category=DeprecationWarning
                )
                self._parse_additional_data()

    def _parse_additional_data(self):
        # data specific parsing not done yet in upstream qiskit
        if hasattr(self._backend, "_conf_dict") and self._backend._conf_dict["open_pulse"]:
            if "u_channel_lo" not in self._backend._conf_dict:
                self._backend._conf_dict["u_channel_lo"] = []  # to avoid qiskit bug
            self._pulse_conf = PulseBackendConfiguration.from_dict(self._backend._conf_dict)

    @property
    def name(self):
        """Returns the backend name"""
        if self._v1:
            return self._backend.name()
        elif self._v2:
            return self._backend.name
        return str(self._backend)

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
            coupling_map = self._backend.coupling_map
            if coupling_map is None:
                return coupling_map
            return list(coupling_map.get_edges())
        return []

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
        try:
            if self._v1:
                return self._backend.provider()
            elif self._v2:
                return self._backend.provider
        except AttributeError:
            return None
        return None

    @property
    def num_qubits(self):
        """Returns the backend's number of qubits"""
        if self._v1:
            return self._backend.configuration().num_qubits
        elif self._v2:
            # meas_freq_est is currently not part of the BackendV2
            return self._backend.num_qubits
        return None

    def qubit_t1(self, qubit: int) -> float:
        """Return the T1 value for a qubit from the backend properties

        Args:
            qubit: the qubit index to return T1 for

        Returns:
            The T1 value
        """
        if self._v1:
            return self._backend.properties().qubit_property(qubit)["T1"][0]
        if self._v2:
            return self._backend.qubit_properties(qubit).t1
        return float("nan")
