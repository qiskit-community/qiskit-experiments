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

"""Ramsey XY frequency calibration experiment."""

from typing import List, Optional

from qiskit.providers.backend import Backend

from qiskit_experiments.framework import fix_class_docs
from qiskit_experiments.library import RamseyXY
from qiskit_experiments.calibration_management import BaseCalibrationExperiment
from qiskit_experiments.calibration_management.backend_calibrations import BackendCalibrations


@fix_class_docs
class FrequencyCal(BaseCalibrationExperiment, RamseyXY):
    """A qubit frequency calibration experiment based on the Ramsey XY experiment.

    # section: see_also
        qiskit_experiments.library.characterization.ramsey_xy.RamseyXY
    """

    def __init__(
        self,
        calibrations: BackendCalibrations,
        qubit: int,
        backend: Optional[Backend] = None,
        delays: Optional[List] = None,
        unit: str = "s",
        osc_freq: float = 2e6,
        auto_update: bool = True,
    ):
        """
        Args:
            calibrations:
            qubit: The qubit on which to run the frequency calibration.
            backend: Optional, the backend to run the experiment on.
            delays: The list of delays that will be scanned in the experiment.
            unit: The unit of the delays. Accepted values are dt, i.e. the
                duration of a single sample on the backend, seconds, and sub-units,
                e.g. ms, us, ns.
            osc_freq: A frequency shift in Hz that will be applied by means of
                a virtual Z rotation to increase the frequency of the measured oscillation.
            auto_update: If set to True, which is the default, then the experiment will
                automatically update the frequency in the calibrations.
        """
        super().__init__(
            calibrations,
            qubit,
            backend=backend,
            delays=delays,
            unit=unit,
            osc_freq=osc_freq,
            auto_update=auto_update,
        )

        # Instruction schedule map to bring in the calibrations for the sx gate.
        self.set_transpile_options(inst_map=calibrations.default_inst_map)
