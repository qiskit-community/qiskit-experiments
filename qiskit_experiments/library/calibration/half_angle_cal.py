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

from typing import List, Optional

from qiskit import QuantumCircuit

from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.calibration_management import (
    BaseCalibrationExperiment,
    BackendCalibrations,
)
from qiskit_experiments.library.characterization import HalfAngle
from qiskit_experiments.calibration_management.update_library import BaseUpdater


class HalfAngleCal(BaseCalibrationExperiment, HalfAngle):
    """Calibration version of the half-angle experiment."""

    def __init__(
        self,
        qubit,
        calibrations: BackendCalibrations,
        schedule_name: str,
        cal_parameter_name: Optional[str] = "amp",
        auto_update: bool = True,
    ):
        """see class :class:`HalfAngle` for details.

        Args:
            qubit: The qubit for which to run the half-angle calibration.
            calibrations: The calibrations instance with the schedules.
            schedule_name: The name of the schedule to calibrate.
            cal_parameter_name: The name of the parameter in the schedule to update.
            auto_update:  Whether or not to automatically update the calibrations. By
                default this variable is set to True.
        """
        super().__init__(
            calibrations,
            qubit,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
        )

        self.transpile_options.inst_map = calibrations.default_inst_map

    @classmethod
    def _default_experiment_options(cls):
        """Default values for the fine amplitude calibration experiment.

        Experiment Options:
            result_index (int): The index of the result from which to update the calibrations.
            target_angle (float): The target angle of the pulse.
            group (str): The calibration group to which the parameter belongs. This will default
                to the value "default".

        """
        options = super()._default_experiment_options()

        options.result_index = -1
        options.group = "default"

        return options

    def _add_cal_metadata(self, circuits: List[QuantumCircuit]):
        """Add metadata to the circuit to make the experiment data more self contained.

        The following keys are added to each circuit's metadata:
            cal_param_value: The value of the pulse amplitude. This value together with
                the fit result will be used to find the new value of the pulse amplitude.
            cal_param_name: The name of the parameter in the calibrations.
            cal_schedule: The name of the schedule in the calibrations.
            cal_group: The calibration group to which the parameter belongs.
        """

        param_val = self._cals.get_parameter_value(
            self._param_name,
            self._physical_qubits,
            self._sched_name,
            group=self.experiment_options.group,
        )

        for circuit in circuits:
            circuit.metadata["cal_param_value"] = param_val
            circuit.metadata["cal_param_name"] = self._param_name
            circuit.metadata["cal_schedule"] = self._sched_name
            circuit.metadata["cal_group"] = self.experiment_options.group

        return circuits

    def update_calibrations(self, experiment_data: ExperimentData):
        r"""Update the value of the parameter in the calibrations.

        The update rule for the half angle calibration is:

        ..math::

            TODO

        Args:
            experiment_data: The experiment data from which to extract the measured over/under
                rotation used to adjust the amplitude.
        """

        data = experiment_data.data()

        # No data -> no update
        if len(data) > 0:
            result_index = self.experiment_options.result_index
            group = data[0]["metadata"]["cal_group"]
            prev_amp = data[0]["metadata"]["cal_param_value"]

            d_theta = BaseUpdater.get_value(experiment_data, "d_theta", result_index)
            new_amp = None  # TODO implement me

            BaseUpdater.add_parameter_value(
                self._cals,
                experiment_data,
                new_amp,
                self._param_name,
                self._sched_name,
                group,
            )
