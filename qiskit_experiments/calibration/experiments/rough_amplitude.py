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
from typing import List, Optional
import numpy as np

from qiskit.pulse import DriveChannel
from qiskit import QuantumCircuit
from qiskit_experiments.calibration.analysis import CosineFit
from qiskit_experiments.calibration.analysis.utils import get_period_fraction
from qiskit_experiments.calibration.metadata import CalibrationMetadata
from qiskit_experiments.calibration.calibration_definitions import CalibrationsDefinition
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.calibration.parameter_value import ParameterValue
from qiskit_experiments import ExperimentData
from .base_calibration_experiment import BaseCalibrationExperiment


class RoughAmplitude(BaseCalibrationExperiment):
    """Rough amplitude calibration that scans the amplitude."""

    # Analysis class for experiment
    __analysis_class__ = CosineFit

    # Gates and pulse parameters to update
    __calibration_objective__ = {
        'gates': ['x90p', 'xp'],
        'options': [np.pi/2, np.pi],
        'parameter_names': ['amp_x90p', 'amp_xp']
    }

    def __init__(self,
                 qubit: int,
                 cals: CalibrationsDefinition,
                 gate_name: str,
                 parameter_name: str,
                 data_processor: DataProcessor,
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
            group: The calibration group that will be updated which defaults to 'default'.
        """
        circuit_options = ['initial_layout', 'amplitudes']
        super().__init__([qubit], self.__class__.__name__, circuit_options)

        self._cal_def = cals
        self.parameter = parameter_name
        self._data_processor = data_processor
        self._calibration_group = group
        self._gate_name = gate_name

    def circuits(self, backend=None, **circuit_options) -> List[QuantumCircuit]:
        """
        Create the circuits for a rough rabi amplitude calibration and add the
        required metadata.

        Args:
            backend (Backend): Not used.
            circuit_options: Can contain 'amplitudes' a list of floats specifying
                the amplitude values to scan. If amplitudes is not given the free
                parameter will be scanned from -0.9 to 0.9 in 51 steps.

        Returns:
            A list of quantum circuits where a parameter is scanned.
        """
        template = self._cal_def.get_circuit(self._gate_name, self._physical_qubits, [self.parameter])
        template.measure(0, 0)
        template.name = 'circuit'

        circuits = []
        for amplitude in circuit_options.get('amplitudes', np.linspace(-0.9, 0.9, 51)):
            meta = CalibrationMetadata(
                experiment_type=self._type,
                pulse_schedule_name=self.__class__.__name__,
                x_values=amplitude,
                qubits=self._physical_qubits
            )

            qc = template.assign_parameters({template.parameters[0]: amplitude})
            qc.metadata = asdict(meta)
            circuits.append(qc)

        return circuits

    def update_calibrations(self, experiment_data: ExperimentData, index: int = -1):
        """
        Updates the amplitude of the pulses. This will preserve the existing phase.

        Args:
            experiment_data: The experiment data to use to update the pulse amplitudes.
            index: The index of analysis result to use in experiment_data. If this is not
                specified then the latest added analysis result is used.
        """
        fit_result = experiment_data.analysis_result(index)['default']

        for idx, gate in enumerate(self.calibration_objective['gates']):
            angle = self.calibration_objective['options'][idx]
            param_name = self.calibration_objective['parameter_names'][idx]
            amp = self._cal_def.parameter_value(param_name,
                                                DriveChannel(self._physical_qubits[0]),
                                                group=self._calibration_group)

            phase = np.exp(1.0j*np.angle(amp))
            value = phase*get_period_fraction(self.__analysis_class__, angle, fit_result)

            param_val = ParameterValue(value, datetime.now(), exp_id=experiment_data.experiment_id,
                                       group=self._calibration_group)

            self._cal_def.add_parameter_value(param_name, param_val,
                                              DriveChannel(self._physical_qubits[0]))
