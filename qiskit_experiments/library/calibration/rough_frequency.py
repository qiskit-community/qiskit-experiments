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

"""Spectroscopy calibration experiment class."""

from typing import List, Optional, Union
import numpy as np

from qiskit_experiments.library.characterization.qubit_spectroscopy import QubitSpectroscopy
from qiskit_experiments.library.characterization.ef_spectroscopy import EFSpectroscopy
from qiskit_experiments.calibration_management.update_library import Frequency
from qiskit_experiments.calibration_management.backend_calibrations import BackendCalibrations
from qiskit_experiments.calibration_management.base_calibration_experiment import (
    BaseCalibrationExperiment,
)


class RoughFrequency(BaseCalibrationExperiment, QubitSpectroscopy):
    """A calibration experiment that runs QubitSpectroscopy."""

    __updater__ = Frequency

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        qubit: int,
        frequencies: Union[List[float], np.array],
        calibrations: Optional[BackendCalibrations] = None,
        unit: Optional[str] = "Hz",
        auto_update: Optional[bool] = True,
        absolute: bool = True,
    ):
        """See :class:`QubitSpectroscopy` for detailed documentation.

        Args:
            qubit: The qubit on which to run spectroscopy.
            frequencies: The frequencies to scan in the experiment.
            cals: If calibrations is given then running the experiment may update the values
                of the frequencies stored in calibrations.
            unit: The unit in which the user specifies the frequencies. Can be one of 'Hz', 'kHz',
                'MHz', 'GHz'. Internally, all frequencies will be converted to 'Hz'.
            absolute: Boolean to specify if the frequencies are absolute or relative to the
                qubit frequency in the backend.

        Raises:
            QiskitError: if there are less than three frequency shifts or if the unit is not known.

        """
        QubitSpectroscopy.__init__(self, qubit, frequencies, unit, absolute)
        self._cals = calibrations
        self._sched_name = None
        self._param_name = None
        self._auto_update = auto_update


class RoughEFFrequency(BaseCalibrationExperiment, EFSpectroscopy):
    """A calibration experiment that runs QubitSpectroscopy."""

    __updater__ = Frequency

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        qubit: int,
        frequencies: Union[List[float], np.array],
        cals: Optional[BackendCalibrations] = None,
        unit: Optional[str] = "Hz",
        absolute: bool = True,
    ):
        """See :class:`QubitSpectroscopy` for detailed documentation.

        Args:
            qubit: The qubit on which to run spectroscopy.
            frequencies: The frequencies to scan in the experiment.
            cals: If calibrations is given then running the experiment may update the values
                of the frequencies stored in calibrations.
            unit: The unit in which the user specifies the frequencies. Can be one of 'Hz', 'kHz',
                'MHz', 'GHz'. Internally, all frequencies will be converted to 'Hz'.
            absolute: Boolean to specify if the frequencies are absolute or relative to the
                qubit frequency in the backend.

        Raises:
            QiskitError: if there are less than three frequency shifts or if the unit is not known.

        """
        EFSpectroscopy.__init__(self, qubit, frequencies, unit, absolute)
        self._cals = cals
        self._param_name = "f12"
