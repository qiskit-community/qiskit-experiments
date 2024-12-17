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

"""Calibration version of spectroscopy experiments."""

from typing import Iterable, Optional, Sequence

from qiskit.circuit import QuantumCircuit
from qiskit.providers.backend import Backend

from qiskit_experiments.library.characterization.qubit_spectroscopy import QubitSpectroscopy
from qiskit_experiments.library.characterization.ef_spectroscopy import EFSpectroscopy
from qiskit_experiments.calibration_management.update_library import Frequency
from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.calibration_management.base_calibration_experiment import (
    BaseCalibrationExperiment,
)


class RoughFrequencyCal(BaseCalibrationExperiment, QubitSpectroscopy):
    """A calibration experiment that runs :class:`.QubitSpectroscopy` to calibrate the qubit
    transition frequency.

    # section: example
        .. jupyter-execute::
            :hide-code:

            import warnings
            warnings.filterwarnings("ignore", ".*Could not determine job completion time.*", UserWarning)

            # backend
            from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend
            backend = SingleTransmonTestBackend(5.2e9,-.25e9, 1e9, 0.8e9, 1e4, noise=True, seed=100)

        .. jupyter-execute::

            import numpy as np
            from qiskit_experiments.calibration_management.calibrations import Calibrations
            from qiskit_experiments.calibration_management.basis_gate_library \
            import FixedFrequencyTransmon
            from qiskit_experiments.library.calibration.rough_frequency import RoughFrequencyCal

            qubit=0
            library = FixedFrequencyTransmon()
            cals = Calibrations.from_backend(backend=backend, libraries=[library])

            freq_est = backend.defaults().qubit_freq_est[qubit]
            frequencies = np.linspace(freq_est-15e6, freq_est+15e6, 51)
            exp_cal = RoughFrequencyCal((qubit,), cals, frequencies, backend=backend)
            exp_cal.set_experiment_options(amp=0.005)

            cal_data = exp_cal.run().block_for_results()
            display(cal_data.figure(0))
            cal_data.analysis_results(dataframe=True)
    """

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        frequencies: Iterable[float],
        backend: Optional[Backend] = None,
        auto_update: bool = True,
        absolute: bool = True,
        cal_parameter_name: Optional[str] = "drive_freq",
    ):
        """See :class:`.QubitSpectroscopy` for detailed documentation.

        Args:
            physical_qubits: List with the qubit on which to run spectroscopy.
            calibrations: If calibrations is given then running the experiment may update the values
                of the frequencies stored in calibrations.
            frequencies: The frequencies to scan in the experiment, in Hz.
            backend: Optional, the backend to run the experiment on.
            auto_update: If set to True, which is the default, then the experiment will
                automatically update the frequency in the calibrations.
            absolute: Boolean to specify if the frequencies are absolute or relative to the
                qubit frequency in the backend.
            cal_parameter_name: The name of the parameter to update in the calibrations.
                This defaults to `drive_freq`.

        Raises:
            QiskitError: If there are less than three frequency shifts.

        """
        super().__init__(
            calibrations,
            physical_qubits,
            frequencies,
            backend=backend,
            absolute=absolute,
            updater=Frequency,
            auto_update=auto_update,
            cal_parameter_name=cal_parameter_name,
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        """QubitSpectroscopy already has the schedules attached in the program circuits."""
        pass


class RoughEFFrequencyCal(BaseCalibrationExperiment, EFSpectroscopy):
    r"""A calibration experiment that runs :class:`.QubitSpectroscopy` for the
    :math:`|1\rangle` <-> :math:`|2\rangle` transition.

    """

    __updater__ = Frequency

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        frequencies: Iterable[float],
        backend: Optional[Backend] = None,
        auto_update: bool = True,
        absolute: bool = True,
    ):
        """See :class:`.QubitSpectroscopy` for detailed documentation.

        Args:
            physical_qubits: List containing the qubit on which to run spectroscopy.
            calibrations: If calibrations is given then running the experiment may update the values
                of the frequencies stored in calibrations.
            frequencies: The frequencies to scan in the experiment, in Hz.
            backend: Optional, the backend to run the experiment on.
            auto_update: If set to True, which is the default, then the experiment will
                automatically update the frequency in the calibrations.
            absolute: Boolean to specify if the frequencies are absolute or relative to the
                qubit frequency in the backend.

        Raises:
            QiskitError: If there are less than three frequency shifts.

        """
        super().__init__(
            calibrations,
            physical_qubits,
            frequencies,
            backend,
            absolute,
            cal_parameter_name="f12",
            updater=Frequency,
            auto_update=auto_update,
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        """Adds the calibrations to the transpiled circuits."""
        schedule = self._cals.get_schedule("x", self.physical_qubits)
        circuit.add_calibration("x", self.physical_qubits, schedule)
