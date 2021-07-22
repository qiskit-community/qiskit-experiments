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

"""Store and manage the results of calibration experiments in the context of a backend."""

from datetime import datetime
from enum import Enum
from typing import List
import copy

from qiskit.providers.backend import BackendV1 as Backend
from qiskit.circuit import Parameter
from qiskit_experiments.calibration_management.calibrations import Calibrations, ParameterKey
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management.basis_gate_library import BasisGateLibrary


class FrequencyElement(Enum):
    """An extendable enum for components that have a frequency."""

    QUBIT = "Qubit"
    READOUT = "Readout"


class BackendCalibrations(Calibrations):
    """
    A Calibrations class to enable a seamless interplay with backend objects.
    This class enables users to export their calibrations into a backend object.
    Additionally, it creates frequency parameters for qubits and readout resonators.
    The parameters are named `qubit_lo_freq` and `meas_lo_freq` to be consistent
    with the naming in backend.defaults(). These two parameters are not attached to
    any schedule.
    """

    __qubit_freq_parameter__ = "qubit_lo_freq"
    __readout_freq_parameter__ = "meas_lo_freq"

    def __init__(
        self,
        backend: Backend,
        library: BasisGateLibrary = None,
    ):
        """Setup an instance to manage the calibrations of a backend.

        BackendCalibrations can be initialized from a basis gate library, i.e. a subclass of
        :class:`BasisGateLibrary`. As example consider the following code:

        .. code-block:: python

            cals = BackendCalibrations(
                    backend,
                    library=FixedFrequencyTransmon(
                        basis_gates=["x", "sx"],
                        default_values={duration: 320}
                    )
                )

        Args:
            backend: A backend instance from which to extract the qubit and readout frequencies
                (which will be added as first guesses for the corresponding parameters) as well
                as the coupling map.
            library: A library class that will be instantiated with the library options to then
                get template schedules to register as well as default parameter values.
        """
        if hasattr(backend.configuration(), "control_channels"):
            control_channels = backend.configuration().control_channels
        else:
            control_channels = None

        super().__init__(control_channels)

        # Use the same naming convention as in backend.defaults()
        self.qubit_freq = Parameter(self.__qubit_freq_parameter__)
        self.meas_freq = Parameter(self.__readout_freq_parameter__)
        self._register_parameter(self.qubit_freq, ())
        self._register_parameter(self.meas_freq, ())

        self._qubits = set(range(backend.configuration().n_qubits))
        self._backend = backend

        for qubit, freq in enumerate(backend.defaults().qubit_freq_est):
            self.add_parameter_value(freq, self.qubit_freq, qubit)

        for meas, freq in enumerate(backend.defaults().meas_freq_est):
            self.add_parameter_value(freq, self.meas_freq, meas)

        if library is not None:

            # Add the basis gates
            for gate in library.basis_gates:
                self.add_schedule(library[gate])

            # Add the default values
            for param_conf in library.default_values():
                schedule_name = param_conf[-1]
                if schedule_name in library.basis_gates:
                    self.add_parameter_value(*param_conf)

    def _get_frequencies(
        self,
        element: FrequencyElement,
        group: str = "default",
        cutoff_date: datetime = None,
    ) -> List[float]:
        """Internal helper method."""

        if element == FrequencyElement.READOUT:
            param = self.meas_freq.name
        elif element == FrequencyElement.QUBIT:
            param = self.qubit_freq.name
        else:
            raise CalibrationError(f"Frequency element {element} is not supported.")

        freqs = []
        for qubit in self._qubits:
            schedule = None  # A qubit frequency is not attached to a schedule.
            if ParameterKey(param, (qubit,), schedule) in self._params:
                freq = self.get_parameter_value(param, (qubit,), schedule, True, group, cutoff_date)
            else:
                if element == FrequencyElement.READOUT:
                    freq = self._backend.defaults().meas_freq_est[qubit]
                elif element == FrequencyElement.QUBIT:
                    freq = self._backend.defaults().qubit_freq_est[qubit]
                else:
                    raise CalibrationError(f"Frequency element {element} is not supported.")

            freqs.append(freq)

        return freqs

    def get_qubit_frequencies(
        self,
        group: str = "default",
        cutoff_date: datetime = None,
    ) -> List[float]:
        """
        Get the most recent qubit frequencies. They can be passed to the run-time
        options of :class:`BaseExperiment`. If no calibrated frequency value of a
        qubit is found then the default value from the backend defaults is used.
        Only valid parameter values are returned.

        Args:
            group: The calibration group from which to draw the
                parameters. If not specified, this defaults to the 'default' group.
            cutoff_date: Retrieve the most recent parameter up until the cutoff date. Parameters
                generated after the cutoff date will be ignored. If the cutoff_date is None then
                all parameters are considered. This allows users to discard more recent values
                that may be erroneous.

        Returns:
            A List of qubit frequencies for all qubits of the backend.
        """
        return self._get_frequencies(FrequencyElement.QUBIT, group, cutoff_date)

    def get_meas_frequencies(
        self,
        group: str = "default",
        cutoff_date: datetime = None,
    ) -> List[float]:
        """
        Get the most recent measurement frequencies. They can be passed to the run-time
        options of :class:`BaseExperiment`. If no calibrated frequency value of a
        measurement is found then the default value from the backend defaults is used.
        Only valid parameter values are returned.

        Args:
            group: The calibration group from which to draw the
                parameters. If not specified, this defaults to the 'default' group.
            cutoff_date: Retrieve the most recent parameter up until the cutoff date. Parameters
                generated after the cutoff date will be ignored. If the cutoff_date is None then
                all parameters are considered. This allows users to discard more recent values
                that may be erroneous.

        Returns:
            A List of measurement frequencies for all qubits of the backend.
        """
        return self._get_frequencies(FrequencyElement.READOUT, group, cutoff_date)

    def export_backend(self) -> Backend:
        """
        Exports the calibrations to a backend object that can be used.

        Returns:
            calibrated backend: A backend with the calibrations in it.
        """
        backend = copy.deepcopy(self._backend)

        backend.defaults().qubit_freq_est = self.get_qubit_frequencies()
        backend.defaults().meas_freq_est = self.get_meas_frequencies()

        # TODO: build the instruction schedule map using the stored calibrations

        return backend
