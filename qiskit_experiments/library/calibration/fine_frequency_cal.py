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

"""Fine frequency calibration experiment."""

from typing import List, Optional
import numpy as np

from qiskit.providers.backend import Backend

from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.calibration_management import (
    BaseCalibrationExperiment,
    BackendCalibrations,
)
from qiskit_experiments.library.characterization.fine_frequency import FineFrequency


class FineFrequencyCal(BaseCalibrationExperiment, FineFrequency):
    """A calibration version of the fine frequency experiment."""

    def __init__(
        self,
        qubit: int,
        calibrations: BackendCalibrations,
        backend: Optional[Backend] = None,
        repetitions: List[int] = None,
        auto_update: bool = True,
    ):
        r"""see class :class:`FineDrag` for details.

        Note that this class implicitly assumes that the target angle of the gate
        is :math:`\pi` as seen from the default experiment options.

        Args:
            qubit: The qubit for which to run the fine drag calibration.
            calibrations: The calibrations instance with the schedules.
            backend: Optional, the backend to run the experiment on.
            auto_update: Whether or not to automatically update the calibrations. By
                default this variable is set to True.
        """
        super().__init__(
            calibrations,
            qubit,
            repetitions,
            schedule_name=None,
            backend=backend,
            cal_parameter_name="qubit_lo_freq",
            auto_update=auto_update,
        )

        self.set_transpile_options(
            inst_map=calibrations.default_inst_map,
            basis_gates=["sx", "rz"],
        )

    def update_calibrations(self, experiment_data: ExperimentData):
        r"""Update the qubit frequency based on the measured angle deviation.

        The frequency of the qubit is updated according to

        ..math::

            f \to f - \frac{{\rm d}\theta}{2\pi\tau{\rm d}t}

        Here, :math:`{\rm d}\theta` is the measure angle error from the fit. The duration of
        the single qubit-gate is :math:`\tau` in samples and :math:`{\rm d}t` is the duration
        of a single arbitrary waveform generator sample.
        """
        # TODO
