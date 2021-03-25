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

"""Base experiment class for calibration."""

from abc import abstractmethod
from typing import Dict, List

from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments import ExperimentData
from qiskit_experiments.calibration.exceptions import CalibrationError


class BaseCalibrationExperiment(BaseExperiment):
    """Base class for all calibration experiments."""

    # Gates and pulse parameters to update
    __calibration_objective__ = {
        'gates': [],
        'options': [],
        'parameter_names': []
    }

    @property
    def calibration_objective(self):
        """Return the calibration objective of the experiment."""
        return self.__calibration_objective__

    @calibration_objective.setter
    def calibration_objective(self, objective_dict: Dict[str, List]):
        """
        Args:
            objective_dict: The objective dictionary specifies the gates, parameter_names
                and options of the calibration.
        """
        if 'parameter_names' not in objective_dict:
            raise CalibrationError('The objective must specify the parameter_names.')

        self.__calibration_objective__ = dict(objective_dict)

    @abstractmethod
    def update_calibrations(self, experiment_data: ExperimentData, index: int = -1):
        """
        Update the parameters in the calibration table.

        Args:
            experiment_data: The experiment data to use to update the pulse amplitudes.
            index: The index of analysis result to use in experiment_data. If this is not
                specified then the latest added analysis result is used.
        """
