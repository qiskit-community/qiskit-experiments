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

"""Half angle calibration."""

from typing import Dict, Optional
import numpy as np

from qiskit.providers.backend import Backend

from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.calibration_management import (
    BaseCalibrationExperiment,
    Calibrations,
)
from qiskit_experiments.library.characterization import HalfAngle
from qiskit_experiments.calibration_management.update_library import BaseUpdater


class HalfAngleCal(BaseCalibrationExperiment, HalfAngle):
    """Calibration version of the half-angle experiment.

    # section: see_also
        qiskit_experiments.library.characterization.half_angle.HalfAngle
    """

    def __init__(
        self,
        qubit,
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        schedule_name: str = "sx",
        cal_parameter_name: Optional[str] = "amp",
        auto_update: bool = True,
    ):
        """see class :class:`HalfAngle` for details.

        Args:
            qubit: The qubit for which to run the half-angle calibration.
            calibrations: The calibrations instance with the schedules.
            backend: Optional, the backend to run the experiment on.
            schedule_name: The name of the schedule to calibrate which defaults to sx.
            cal_parameter_name: The name of the parameter in the schedule to update. This will
                default to amp since the complex amplitude contains the phase of the pulse.
            auto_update:  Whether or not to automatically update the calibrations. By
                default this variable is set to True.
        """
        super().__init__(
            calibrations,
            qubit,
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
        )

    def _metadata(self) -> Dict[str, any]:
        """Add metadata to the experiment data making it more self contained.

        The following keys are added to the experiment's metadata:
            cal_param_value: The value of the pulse amplitude. This value together with
                the fit result will be used to find the new value of the pulse amplitude.
            cal_param_name: The name of the parameter in the calibrations.
            cal_schedule: The name of the schedule in the calibrations.
            cal_group: The calibration group to which the parameter belongs.
        """
        metadata = super()._metadata()
        metadata["cal_param_value"] = self._cals.get_parameter_value(
            self._param_name,
            self._physical_qubits,
            self._sched_name,
            group=self.experiment_options.group,
        )

        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        r"""Update the value of the parameter in the calibrations.

        The parameter that is updated is the phase of the sx pulse. This phase is contained
        in the complex amplitude of the pulse. The update rule for the half angle calibration is
        therefore:

        ..math::

            A \to A \cdot e^{-i{\rm d}\theta_\text{hac}/2}

        where :math:`A` is the complex amplitude of the sx pulse which has an angle which might be
        different from the angle of the x pulse due to the non-linearity in the mixer's skew. The
        angle :math:`{\rm d}\theta_\text{hac}` is the angle deviation measured through the error
        amplifying pulse sequence.

        Args:
            experiment_data: The experiment data from which to extract the measured over/under
                rotation used to adjust the amplitude.
        """

        result_index = self.experiment_options.result_index
        group = experiment_data.metadata["cal_group"]
        prev_amp = experiment_data.metadata["cal_param_value"]

        d_theta = BaseUpdater.get_value(experiment_data, "d_hac", result_index)
        new_amp = prev_amp * np.exp(-1.0j * d_theta / 2)

        BaseUpdater.add_parameter_value(
            self._cals,
            experiment_data,
            new_amp,
            self._param_name,
            self._sched_name,
            group,
        )
