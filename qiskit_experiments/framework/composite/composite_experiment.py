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
from qiskit_experiments.exceptions import QiskitError
from qiskit_experiments.framework import BaseExperiment
from .composite_analysis import CompositeAnalysis


class CompositeExperiment(BaseExperiment):
    """Composite Experiment base class"""

    def __init__(
        self,
        experiments: List[BaseExperiment],
        physical_qubits: Sequence[int],
        backend: Optional[Backend] = None,
        experiment_type: Optional[str] = None,
        flatten_results: bool = True,
        analysis: Optional[CompositeAnalysis] = None,
    ):
        """Initialize the composite experiment object.

        Args:
            experiments: a list of experiment objects.
            physical_qubits: list of physical qubits for the experiment.
            backend: Optional, the backend to run the experiment on.
            experiment_type: Optional, composite experiment subclass name.
            flatten_results: If True flatten all component experiment results
                             into a single ExperimentData container, including
                             nested composite experiments. If False save each
                             component experiment results as a separate child
                             ExperimentData container. This kwarg is ignored
                             if the analysis kwarg is used.
            analysis: Optional, the composite analysis class to use. If not
                      provided this will be initialized automatically from the
                      supplied experiments.

        Raises:
            QiskitError: If the provided analysis class is not a CompositeAnalysis
                         instance.
        """
        self._experiments = experiments
        self._num_experiments = len(experiments)
        if analysis is None:
            analysis = CompositeAnalysis(
                [exp.analysis for exp in self._experiments], flatten_results=flatten_results
            )
        super().__init__(
            physical_qubits,
            analysis=analysis,
            backend=backend,
            experiment_type=experiment_type,
        )

    @abstractmethod
    def circuits(self):
        pass

    def set_transpile_options(self, **fields):
        super().set_transpile_options(**fields)
        # Recursively set transpile options of component experiments
        for exp in self._experiments:
            exp.set_transpile_options(**fields)

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

    @property
    def analysis(self) -> Union[CompositeAnalysis, None]:
        """Return the analysis instance for the experiment"""
        return self._analysis

    @analysis.setter
    def analysis(self, analysis: Union[CompositeAnalysis, None]) -> None:
        """Set the analysis instance for the experiment"""
        if analysis is not None and not isinstance(analysis, CompositeAnalysis):
            raise TypeError("Input is not a None or a CompositeAnalysis subclass.")
        self._analysis = analysis

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
        #
        # In addition, we raise an error if we detect inconsistencies in
        # the usage of BatchExperiment separate_job experiment option.

        for i, subexp in enumerate(self._experiments):
            # Validate set and default run options in component experiment
            # against and component experiment run options and raise a
            # warning if any are different and will be overridden
            overridden_keys = []
            sub_vals = []
            comp_vals = []
            for key, sub_val in subexp.run_options.__dict__.items():
                comp_val = getattr(self.run_options, key, None)
                if sub_val != comp_val:
                    overridden_keys.append(key)
                    sub_vals.append(sub_val)
                    comp_vals.append(comp_val)

            if overridden_keys:
                warnings.warn(
                    f"Component {i} {subexp.experiment_type} experiment run options"
                    f" {overridden_keys} values {sub_vals} will be overridden with"
                    f" {self.experiment_type} values {comp_vals}.",
                    UserWarning,
                )
                # Update sub-experiment options with actual run option values so
                # they can be used by that sub experiments _finalize method.
                subexp.set_run_options(**dict(zip(overridden_keys, comp_vals)))

            if not self.experiment_options.get(
                "separate_jobs", False
            ) and subexp.experiment_options.get("separate_jobs", False):
                raise QiskitError(
                    "It is not allowed to request to separate jobs in a child experiment,"
                    " if its parent does not separate jobs as well"
                )

            # Call sub-experiments finalize method
            subexp._finalize()

    def _metadata(self):
        """Add component experiment metadata"""
        metadata = super()._metadata()
        metadata["component_types"] = [
            sub_exp.experiment_type for sub_exp in self.component_experiment()
        ]
        metadata["component_metadata"] = [
            sub_exp._metadata() for sub_exp in self.component_experiment()
        ]
        return metadata
