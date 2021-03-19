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

"""Rough amplitude calibration."""

from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

from qiskit.pulse import DriveChannel
from qiskit import QuantumCircuit
from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.calibration.analysis import CosineFit
from qiskit_experiments.calibration.analysis.utils import get_period_fraction
from qiskit_experiments.calibration.metadata import CalibrationMetadata
from qiskit_experiments.calibration.calibration_definitions import CalibrationsDefinition
from qiskit_experiments.calibration.data_processing.data_processor import DataProcessor
from qiskit_experiments.calibration.parameter_value import ParameterValue
from qiskit_experiments import ExperimentData


class RoughAmplitude(BaseExperiment):
    """Rough amplitude calibration that scans the amplitude."""

    # Analysis class for experiment
    __analysis_class__ = CosineFit

    def __init__(self,
                 qubit: int,
                 cals: CalibrationsDefinition,
                 gate_name: str,
                 parameter_name: str,
                 data_processor: DataProcessor,
                 amplitudes = List[float],
                 group: Optional[str] = 'default'):
        """
        Args:
            qubit: The qubit on which to run the calibration.
            cals: The instance to manage the calibrated pulse schedules and their
                parameters.
            gate_name: The name of the gate in cals to use when constructing the
                rough amplitude calibration.
            parameter_name: The name of the parameter to sweep, typically this will
                be the amplitude of the pulse.
            data_processor: The data processor used to process the measured data.
            amplitudes: The amplitudes over which to sweep.
            group: The calibration group that will be updated which defaults to 'default'.
        """
        circuit_options = ['initial_layout']
        super().__init__([qubit], self.__class__.__name__, circuit_options)

        self._cal_def = cals
        self.amplitudes = amplitudes
        self.parameter = parameter_name
        self._qubit = qubit
        self._data_processor = data_processor
        self._calibration_group = group
        self._gate_name = gate_name

    def circuits(self, backend=None, **circuit_options) -> List[QuantumCircuit]:
        """
        Create the circuits for a rough rabi amplitude calibration and add the
        required metadata.

        Args:
            backend (Backend): Not used.
            circuit_options: Not used.

        Returns:
            A list of quantum circuits where a parameter is scanned.
        """
        circuits = []
        for amplitude in self.amplitudes:
            meta = CalibrationMetadata(
                experiment_type=self._type,
                pulse_schedule_name=self.__class__.__name__,
                x_values=amplitude,
                qubits=self._qubit
            )

            template_qc = self._cal_def.get_circuit(self._gate_name, (self._qubit,), [self.parameter])
            template_qc.measure(0, 0)
            template_qc.name = 'circuit'

            qc = template_qc.assign_parameters({template_qc.parameters[0]: amplitude})
            qc.metadata = asdict(meta)
            circuits.append(qc)

        return circuits

    def update_calibrations(self, experiment_data: ExperimentData,
                            update_pulses: Optional[Dict[str, float]] = None, index: int = -1):
        """
        Updates the amplitude of the pulses. This will preserve the existing phase.

        Args:
            experiment_data: The experiment data to use to update the pulse amplitudes.
            update_pulses: The pulse to update. The key is the name of the pulse parameter and
                the value is the angle to extract from the cosine fit. For example, {'amp_xp':
                np.pi, 'amp_x90p': np.pi/2} will update the amplitude ot the xp and x90p pulses.
            index: The index of analysis result to use in experiment_data. If this is not
                specified then the latest added analysis result is used.
        """
        if update_pulses is None:
            update_pulses = {self._gate_name: np.pi/2}

        fit_result = experiment_data.analysis_result(index)['default']

        for param_name, angle in update_pulses.items():
            amp = self._cal_def.parameter_value(param_name,
                                                DriveChannel(self._qubit),
                                                group=self._calibration_group)

            phase = np.exp(1.0j*np.angle(amp))
            value = phase*get_period_fraction(self.__analysis_class__, angle, fit_result)

            param_val = ParameterValue(value, datetime.now(), exp_id=experiment_data.experiment_id,
                                       group=self._calibration_group)

            self._cal_def.add_parameter_value(param_name, param_val, DriveChannel(self._qubit))
