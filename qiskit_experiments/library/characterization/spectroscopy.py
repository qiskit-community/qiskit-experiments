# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Abstract spectroscopy experiment base class."""

from abc import ABC, abstractmethod
from typing import Iterable, Optional

import numpy as np
import qiskit.pulse as pulse
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.providers import Backend
from qiskit.qobj.utils import MeasLevel

from qiskit_experiments.framework import BaseAnalysis, BaseExperiment, Options
from qiskit_experiments.curve_analysis import ResonanceAnalysis


class Spectroscopy(BaseExperiment, ABC):
    """An abstract class for spectroscopy experiments."""

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default option values used for the spectroscopy pulse.

        Experiment Options:
            amp (float): The amplitude of the spectroscopy pulse. Defaults to 0.1 and must
                be between 0 and 1.
            duration (int): The duration of the spectroscopy pulse. Defaults to 1024 samples.
            sigma (float): The standard deviation of the flanks of the spectroscopy pulse.
                Defaults to 256.
            width (int): The width of the flat-top part of the GaussianSquare pulse.
                Defaults to 0.
        """
        options = super()._default_experiment_options()

        options.amp = 0.1
        options.duration = 240e-9
        options.sigma = 60e-9
        options.width = 0

        return options

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default options values for the experiment :meth:`run` method."""
        options = super()._default_run_options()

        options.meas_level = MeasLevel.KERNELED
        options.meas_return = "avg"

        return options

    def __init__(
        self,
        qubit: int,
        frequencies: Iterable[float],
        backend: Optional[Backend] = None,
        absolute: bool = True,
        analysis: Optional[BaseAnalysis] = None,
        **experiment_options,
    ):
        """A spectroscopy experiment where the frequency of a pulse is scanned.

        Args:
            qubit: The qubit on which to run spectroscopy.
            frequencies: The frequencies to scan in the experiment, in Hz.
            backend: Optional, the backend to run the experiment on.
            absolute: Boolean to specify if the frequencies are absolute or relative to the
                qubit frequency in the backend.
            analysis: An instance of the analysis class to use.
            experiment_options: Key word arguments used to set the experiment options.

        Raises:
            QiskitError: if there are less than three frequency shifts.

        """
        analysis = analysis or ResonanceAnalysis()

        super().__init__([qubit], analysis=analysis, backend=backend)

        if len(frequencies) < 3:
            raise QiskitError("Spectroscopy requires at least three frequencies.")

        self._frequencies = frequencies
        self._absolute = absolute

        self.set_experiment_options(**experiment_options)

    def _set_backend(self, backend: Backend):
        """Set the backend for the experiment and extract config information."""
        super()._set_backend(backend)

        self._dt = self._backend_data.dt
        self._granularity = self._backend_data.granularity

        if self._dt is None or self._granularity is None:
            raise QiskitError(f"{self.__class__.__name__} needs both dt and sample granularity.")

    @property
    @abstractmethod
    def _backend_center_frequency(self) -> float:
        """The default frequency for the channel of the spectroscopy pulse.

        This frequency is used to calculate the appropriate frequency shifts to apply to the
        spectroscopy pulse as its frequency is scanned in the experiment. Spectroscopy experiments
        should implement schedules using frequency shifts. Therefore, if an absolute frequency
        range is given the frequency shifts need to be corrected by the backend default frequency
        which depends on the nature of the spectroscopy experiment.
        """

    def _add_metadata(self, circuit: QuantumCircuit, freq: float, sched: pulse.ScheduleBlock):
        """Helper method to add the metadata to avoid code duplication with subclasses."""

        if not self._absolute:
            freq += self._backend_center_frequency

        circuit.metadata = {
            "experiment_type": self._type,
            "qubits": self.physical_qubits,
            "xval": np.round(freq, decimals=3),
            "unit": "Hz",
            "schedule": str(sched),
        }

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        return metadata
