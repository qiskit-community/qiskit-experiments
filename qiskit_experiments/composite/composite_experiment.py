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
Composite Experiment abstract base class.
"""

from abc import abstractmethod
import warnings

from qiskit_experiments.base_experiment import BaseExperiment
from .composite_experiment_data import CompositeExperimentData
from .composite_analysis import CompositeAnalysis


class CompositeExperiment(BaseExperiment):
    """Composite Experiment base class"""

    __analysis_class__ = CompositeAnalysis
    __experiment_data__ = CompositeExperimentData

    def __init__(self, experiments, qubits, experiment_type=None):
        """Initialize the composite experiment object.

        Args:
            experiments (List[BaseExperiment]): a list of experiment objects.
            qubits (int or Iterable[int]): the number of qubits or list of
                                           physical qubits for the experiment.
            experiment_type (str): Optional, composite experiment subclass name.
        """
        self._experiments = experiments
        self._num_experiments = len(experiments)
        super().__init__(qubits, experiment_type=experiment_type)

    @abstractmethod
    def circuits(self, backend=None):
        pass

    @property
    def num_experiments(self):
        """Return the number of sub experiments"""
        return self._num_experiments

    def component_experiment(self, index):
        """Return the component Experiment object"""
        return self._experiments[index]

    def component_analysis(self, index):
        """Return the component experiment Analysis object"""
        return self.component_experiment(index).analysis()

    def _add_job_metadata(self, experiment_data, job, **run_options):
        # Add composite metadata
        super()._add_job_metadata(experiment_data, job, **run_options)

        # Add sub-experiment options
        for i in range(self.num_experiments):
            sub_exp = self.component_experiment(i)

            # Run and transpile options are always overridden
            if (
                sub_exp.run_options != sub_exp._default_run_options()
                or sub_exp.transpile_options != sub_exp._default_transpile_options()
            ):

                warnings.warn(
                    "Sub-experiment run and transpile options"
                    " are overridden by composite experiment options."
                )
            sub_data = experiment_data.component_experiment_data(i)
            sub_exp._add_job_metadata(sub_data, job, **run_options)

    def _postprocess_transpiled_circuits(self, circuits, backend, **run_options):
        for expr in self._experiments:
            if not isinstance(expr, CompositeExperiment):
                expr._postprocess_transpiled_circuits(circuits, backend, **run_options)
