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

"""Class that defines the structure of the metadata for calibration experiments."""

from dataclasses import dataclass
from typing import Dict, List, Union


@dataclass
class CalibrationMetadata:
    """
    Defines the structure of the meta data that describes
    calibration experiments. Calibration analysis routines will
    use variables of this class to tie together the results from
    different quantum circuits.
    """

    # The name of the calibration experiment.
    name: str = None

    # Type of the calibration experiment
    experiment_type: str = None

    # Name of the pulse schedule that was used in the calibration experiment.
    pulse_schedule_name: str = None

    # A dictionary of x-values the structure of this dict will
    # depend on the experiment being run.
    x_values: Dict[str, Union[int, float, complex]] = None

    # The series of the Experiment to which the circuit is
    # attached to. E.g. 'X' or 'Y' for Ramsey measurements.
    series: Union[str, int, float] = None

    # ID of the experiment to which this circuit is attached.
    exp_id: str = None

    # Physical qubits used.
    qubits: List[int] = None
