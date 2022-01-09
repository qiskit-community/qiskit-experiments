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

from typing import Iterable, Optional
from qiskit.providers.backend import Backend

from qiskit_experiments.library.characterization.qubit_spectroscopy import QubitSpectroscopy
from qiskit_experiments.library.characterization.ef_spectroscopy import EFSpectroscopy
from qiskit_experiments.calibration_management.update_library import Frequency
from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.calibration_management.base_calibration_experiment import (
    BaseCalibrationExperiment,
)


class RoughFrequencyCal(BaseCalibrationExperiment, QubitSpectroscopy):
    """A calibration experiment that runs QubitSpectroscopy.

    # section: see_also
        qiskit_experiments.library.characterization.qubit_spectroscopy.QubitSpectroscopy
    """

    def __init__(
        self,
        qubit: int,
        calibrations: Calibrations,
        frequencies: Iterable[float],
        backend: Optional[Backend] = None,
        auto_update: bool = True,
        absolute: bool = True,
    ):
        """See :class:`QubitSpectroscopy` for detailed documentation.

        Args:
            qubit: The qubit on which to run spectroscopy.
            calibrations: If calibrations is given then running the experiment may update the values
                of the frequencies stored in calibrations.
            frequencies: The frequencies to scan in the experiment, in Hz.
            backend: Optional, the backend to run the experiment on.
            auto_update: If set to True, which is the default, then the experiment will
                automatically update the frequency in the calibrations.
            absolute: Boolean to specify if the frequencies are absolute or relative to the
                qubit frequency in the backend.

        Raises:
            QiskitError: if there are less than three frequency shifts.

        """
        super().__init__(
            calibrations,
            qubit,
            frequencies,
            backend=backend,
            absolute=absolute,
            updater=Frequency,
            auto_update=auto_update,
        )


class RoughEFFrequencyCal(BaseCalibrationExperiment, EFSpectroscopy):
    """A calibration experiment that runs QubitSpectroscopy.

    # section: see_also
        qiskit_experiments.library.characterization.ef_spectroscopy.EFSpectroscopy
    """

    __updater__ = Frequency

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        qubit: int,
        calibrations: Calibrations,
        frequencies: Iterable[float],
        auto_update: bool = True,
        absolute: bool = True,
    ):
        """See :class:`QubitSpectroscopy` for detailed documentation.

        Args:
            qubit: The qubit on which to run spectroscopy.
            calibrations: If calibrations is given then running the experiment may update the values
                of the frequencies stored in calibrations.
            frequencies: The frequencies to scan in the experiment, in Hz.
            auto_update: If set to True, which is the default, then the experiment will
                automatically update the frequency in the calibrations.
            absolute: Boolean to specify if the frequencies are absolute or relative to the
                qubit frequency in the backend.

        Raises:
            QiskitError: if there are less than three frequency shifts.

        """
        super().__init__(
            calibrations,
            qubit,
            frequencies,
            absolute,
            cal_parameter_name="f12",
            updater=Frequency,
            auto_update=auto_update,
        )
