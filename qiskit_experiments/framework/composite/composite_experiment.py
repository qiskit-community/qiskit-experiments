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

from typing import List, Sequence, Optional
from abc import abstractmethod
import warnings
from qiskit.providers.backend import Backend
from qiskit_experiments.framework import BaseExperiment, ExperimentData
from .composite_analysis import CompositeAnalysis


class CompositeExperiment(BaseExperiment):
    """Composite Experiment base class"""

    def __init__(
        self,
        experiments: List[BaseExperiment],
        qubits: Sequence[int],
        backend: Optional[Backend] = None,
        experiment_type: Optional[str] = None,
    ):
        """Initialize the composite experiment object.

        Args:
            experiments: a list of experiment objects.
            qubits: list of physical qubits for the experiment.
            backend: Optional, the backend to run the experiment on.
            experiment_type: Optional, composite experiment subclass name.
        """
        self._experiments = experiments
        self._num_experiments = len(experiments)
        super().__init__(
            qubits,
            analysis=CompositeAnalysis(),
            backend=backend,
            experiment_type=experiment_type,
        )

    @abstractmethod
    def circuits(self):
        pass

    @property
    def num_experiments(self):
        """Return the number of sub experiments"""
        return self._num_experiments

    def component_experiment(self, index=None):
        """Return the component Experiment object.
        Args:
            index (int): Experiment index, or ``None`` if all experiments are to be returned.
        Returns:
            BaseExperiment: The component experiment(s).
        """
        if index is None:
            return self._experiments
        return self._experiments[index]

    def component_analysis(self, index):
        """Return the component experiment Analysis object"""
        return self.component_experiment(index).analysis()

    def copy(self) -> "BaseExperiment":
        """Return a copy of the experiment"""
        ret = super().copy()
        # Recursively call copy of component experiments
        ret._experiments = [exp.copy() for exp in self._experiments]
        return ret

    def _set_backend(self, backend):
        super()._set_backend(backend)
        for subexp in self._experiments:
            subexp._set_backend(backend)

    def _initialize_experiment_data(self):
        """Initialize the return data container for the experiment run"""
        experiment_data = ExperimentData(experiment=self)
        # Initialize child experiment data
        for sub_exp in self._experiments:
            sub_data = sub_exp._initialize_experiment_data()
            experiment_data.add_child_data(sub_data)
        experiment_data.metadata["component_child_index"] = list(range(self.num_experiments))
        return experiment_data

    def _additional_metadata(self):
        """Add component experiment metadata"""
        return {
            "component_metadata": [sub_exp._metadata() for sub_exp in self.component_experiment()]
        }

    def _add_job_metadata(self, metadata, jobs, **run_options):
        super()._add_job_metadata(metadata, jobs, **run_options)
        # Add sub-experiment options
        for sub_metadata, sub_exp in zip(
            metadata["component_metadata"], self.component_experiment()
        ):
            # Run and transpile options are always overridden
            if (
                sub_exp.run_options != sub_exp._default_run_options()
                or sub_exp.transpile_options != sub_exp._default_transpile_options()
            ):
                warnings.warn(
                    "Sub-experiment run and transpile options"
                    " are overridden by composite experiment options."
                )
            sub_exp._add_job_metadata(sub_metadata, jobs, **run_options)

    def _postprocess_transpiled_circuits(self, circuits, **run_options):
        for expr in self._experiments:
            if not isinstance(expr, CompositeExperiment):
                expr._postprocess_transpiled_circuits(circuits, **run_options)
