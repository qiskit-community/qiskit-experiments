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

"""Utility to test calibration module."""

import datetime
from typing import Sequence

from qiskit_experiments.calibration_management import (
    BaseCalibrationExperiment,
    ParameterValue,
    Calibrations,
)
from qiskit_experiments.framework import BaseExperiment, BaseAnalysis, AnalysisResultData


class DoNothingAnalysis(BaseAnalysis):
    """Analysis just returns return_value set in the options."""

    @classmethod
    def _default_options(cls):
        options = super()._default_options()
        options.return_value = None
        return options

    def _run_analysis(
        self,
        experiment_data,
    ):
        ret = AnalysisResultData(
            name="return_value",
            value=self.options.return_value,
        )
        return [ret], []


class DoNothingExperiment(BaseExperiment):
    """Experiment doesn't provide any circuit to run."""

    def __init__(self, physical_qubits: Sequence[int], return_value: float):
        super().__init__(physical_qubits=physical_qubits, analysis=DoNothingAnalysis())
        self.analysis.set_options(return_value=return_value)

    def circuits(self):
        return []


class MockCalExperiment(BaseCalibrationExperiment, DoNothingExperiment):
    """Mock calibration experiment.

    This experiment only invokes update_calibrations method and
    adds new entry to the Calibrations instance.
    Added entry can be directly managed by the constructor arguments.
    """

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        new_value: float,
        param_name: str,
        sched_name: str,
    ):
        """Create mock calibration experiment.

        Args:
            qubits: Qubit to update calibration.
            calibrations: Calibrations instance to update.
            new_value: New parameter value obtained by the calibration experiment.
            param_name: Name of parameter to update.
            sched_name: Name of schedule to update.
        """
        super().__init__(
            physical_qubits=physical_qubits,
            calibrations=calibrations,
            return_value=new_value,
        )
        self.to_update = {
            "param": param_name,
            "qubits": physical_qubits,
            "schedule": sched_name,
        }

    def update_calibrations(self, experiment_data):
        new_value = experiment_data.analysis_results("return_value", block=False).value

        param_value = ParameterValue(
            value=new_value,
            date_time=datetime.datetime.now(),
            group="default",
            exp_id="0123456789",
        )
        self.calibrations.add_parameter_value(value=param_value, **self.to_update)

    def _attach_calibrations(self, circuit):
        pass
