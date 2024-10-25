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

import warnings
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.providers import Backend
from qiskit.qobj.utils import MeasLevel
from qiskit.utils.deprecation import deprecate_func

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

    @deprecate_func(
        since="0.8",
        package_name="qiskit-experiments",
        additional_msg=(
            "Due to the deprecation of Qiskit Pulse, experiments involving pulse "
            "gate calibrations like this one have been deprecated."
        ),
    )
    def __init__(
        self,
        physical_qubits: Sequence[int],
        frequencies: Iterable[float],
        backend: Optional[Backend] = None,
        absolute: bool = True,
        analysis: Optional[BaseAnalysis] = None,
        **experiment_options,
    ):
        """A spectroscopy experiment where the frequency of a pulse is scanned.

        Args:
            physical_qubits: List containing the qubit on which to run spectroscopy.
            frequencies: The frequencies to scan in the experiment, in Hz.
            backend: Optional, the backend to run the experiment on.
            absolute: Boolean to specify if the frequencies are absolute or relative to the
                qubit frequency in the backend.
            analysis: An instance of the analysis class to use.
            experiment_options: Key word arguments used to set the experiment options.

        Raises:
            QiskitError: If there are less than three frequency shifts.

        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="deprecation of Qiskit Pulse",
                module="qiskit_experiments",
                category=DeprecationWarning,
            )
            analysis = analysis or ResonanceAnalysis()

        super().__init__(physical_qubits, analysis=analysis, backend=backend)

        if len(frequencies) < 3:
            raise QiskitError("Spectroscopy requires at least three frequencies.")

        self._frequencies = frequencies
        self._absolute = absolute

        self.set_experiment_options(**experiment_options)

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

    def _add_metadata(self, circuit: QuantumCircuit, freq: float):
        """Helper method to add the metadata to avoid code duplication with subclasses."""

        if not self._absolute:
            freq += self._backend_center_frequency

        circuit.metadata = {"xval": np.round(freq, decimals=3)}

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        return metadata
