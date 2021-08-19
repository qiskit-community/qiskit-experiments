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

"""
EventHook for experiment runner.
"""

import importlib

from qiskit.exceptions import QiskitError
from qiskit_experiments.framework.experiment_data import ExperimentData


class ExperimentRunner:
    """Event handler of the experiment run."""

    def __init__(self, experiment: "BaseExperiment"):
        """Create new runner.

        Args:
            experiment: Associated experiment instance.
        """
        self.__experiment = experiment

        self.__handlers = list()
        self.__scope_vars = dict()

    def add_handler(self, handler: str, module: str):
        """Add new event handler.

        Args:
            handler: Name of event handler.
            module: Name of event module.
        """
        try:
            callback = importlib.import_module(f"{module}.{handler}")
            self.__handlers.append(callback)
        except ModuleNotFoundError:
            QiskitError(f"Event handler {module}.{handler} is not found.")

    def run(self, **kwargs) -> ExperimentData:
        """Run experiment.

        Args:
            kwargs: User provided runtime options.

        Returns:
            Experiment data.
        """
        self.__scope_vars.update(**kwargs)

        for handler in self.__handlers:
            scope_args = handler(self.__experiment, **self.__scope_vars)
            if scope_args:
                self.__scope_vars.update(scope_args)

        # return experiment data
        try:
            return self.__scope_vars["experiment_data"]
        except KeyError:
            # TODO no logger
            raise QiskitError("Experiment result data is not generated. Check log.")
