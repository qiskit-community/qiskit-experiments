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

"""Base class for calibration-type experiments."""

from abc import ABC, abstractmethod
from typing import Optional

from qiskit.providers.options import Options
from qiskit.providers.backend import Backend

from qiskit_experiments.framework.base_experiment import BaseExperiment
from qiskit_experiments.framework.experiment_data import ExperimentData


class BaseCalibrationExperiment(BaseExperiment, ABC):
    """An abstract base class for calibration experiments.

    This abstract base class specifies an experiment and how to update an
    optional instance of :class:`Calibrations` specified in the experiment options
    under calibrations. Furthermore, the experiment options also specifies
    an auto_update variable which, by default, is set to True. If this variable,
    is True then the run method of the experiment will call :meth:`block_for_results`
    and update the calibrations instance.
    """

    # The updater class that updates the Calibrations instance
    __updater__ = None

    @abstractmethod
    def update_calibrations(self, experiment_data: ExperimentData):
        """Update parameter values in the :class:`Calibrations` instance.

        Subclasses must implement this method which will call the :meth:`update`
        method of the updater.
        """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default options for experiment

        Experiment Options:
            calibrations (Calibrations): An optional instance of :class:`Calibrations` if this
                instance is specified then the experiment will try and update the calibrations.
            auto_update (bool): A boolean which defaults to True. If this variable is set to
                True then running the calibration experiment will block for the results and
                update the calibrations if the calibrations is not None.
        """
        options = super()._default_experiment_options()
        options.calibrations = None
        options.auto_update = True

        return options

    def run(
        self,
        backend: Backend,
        analysis: bool = True,
        experiment_data: Optional[ExperimentData] = None,
        **run_options,
    ) -> ExperimentData:
        """Run an experiment, perform analysis, and update any calibrations.

        Args:
            backend: The backend to run the experiment on.
            analysis: If True run analysis on the experiment data.
            experiment_data: Optional, add results to existing experiment data.
                If None a new ExperimentData object will be returned.
            run_options: backend runtime options used for circuit execution.

        Returns:
            The experiment data object.
        """
        experiment_data = super().run(backend, analysis, experiment_data, **run_options)

        if self.experiment_options.auto_update:
            if self.experiment_options.calibrations is not None:
                experiment_data = experiment_data.block_for_results()
                self.update_calibrations(experiment_data)

        return experiment_data
