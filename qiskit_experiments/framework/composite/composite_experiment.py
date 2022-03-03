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

from typing import List, Sequence, Optional, Union
from abc import abstractmethod
import warnings
from qiskit.providers.backend import Backend
from qiskit_experiments.framework import BaseExperiment, ExperimentData
from qiskit_experiments.framework.base_analysis import BaseAnalysis
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
        analysis = CompositeAnalysis([exp.analysis for exp in self._experiments])
        super().__init__(
            qubits,
            analysis=analysis,
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

    def component_experiment(self, index=None) -> Union[BaseExperiment, List[BaseExperiment]]:
        """Return the component Experiment object.

        Args:
            index (int): Experiment index, or ``None`` if all experiments are to be returned.
        Returns:
            BaseExperiment: The component experiment(s).
        """
        if index is None:
            return self._experiments
        return self._experiments[index]

    def component_analysis(self, index=None) -> Union[BaseAnalysis, List[BaseAnalysis]]:
        """Return the component experiment Analysis object"""
        warnings.warn(
            "The `component_analysis` method is deprecated as of "
            "qiskit-experiments 0.3.0 and will be removed in the 0.4.0 release."
            " Use `analysis.component_analysis` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.component_analysis(index)

    def copy(self) -> "BaseExperiment":
        """Return a copy of the experiment"""
        ret = super().copy()
        # Recursively call copy of component experiments
        ret._experiments = [exp.copy() for exp in self._experiments]

        # Check if the analysis in CompositeAnalysis was a reference to the
        # original component experiment analyses and if so update the copies
        # to preserve this relationship
        if isinstance(self.analysis, CompositeAnalysis):
            for i, orig_exp in enumerate(self._experiments):
                if orig_exp.analysis is self.analysis._analyses[i]:
                    # Update copies analysis with reference to experiment analysis
                    ret.analysis._analyses[i] = ret._experiments[i].analysis
        return ret

    def set_run_options(self, **fields):
        super().set_run_options(**fields)
        for subexp in self._experiments:
            subexp.set_run_options(**fields)

    def _set_backend(self, backend):
        super()._set_backend(backend)
        for subexp in self._experiments:
            subexp._set_backend(backend)

    def _finalize(self):
        # NOTE: When CompositeAnalysis is updated to support level-1
        # measurements this method should be updated to validate that all
        # sub-experiments have the same meas level and meas return types,
        # and update the composite experiment run option to that value.

        for i, subexp in enumerate(self._experiments):
            # Raise warning if different run options were set for individual
            # component experiments and not through the composite experiments
            # set_run_options
            for key in subexp._set_run_options:
                subval = getattr(subexp.run_options, key)
                compval = getattr(self.run_options, key, None)
                if subval != compval:
                    warnings.warn(
                        f"Component experiment {i} run option {key}={subval} is "
                        f" differs from the the composite experiment value {compval}"
                        " and will be overridden."
                    )
            subexp._finalize()

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
            sub_exp._add_job_metadata(sub_metadata, jobs, **run_options)
