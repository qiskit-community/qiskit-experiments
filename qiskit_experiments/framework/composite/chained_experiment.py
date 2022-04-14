
from abc import ABC, abstractmethod
from typing import List, Optional, Union

from qiskit.providers import Backend
from qiskit_experiments.framework.base_experiment import BaseExperiment, ExperimentData, BaseAnalysis
from qiskit_experiments.framework.composite.composite_experiment import CompositeExperiment
from qiskit.providers.options import Options
from qiskit import QuantumCircuit


class BaseTransitionCallable(ABC):
    """A method to determine how to transition between experiments."""

    @abstractmethod
    def __call__(
        self,
        experiment_data: ExperimentData,
        **kwargs,
    ) -> int:
        """The call method determines if the ChainedExperiment can transition to the next experiment.

        Args:
            current_exp_idx: The index of the experiment that just run.
            experiment_data: The experiment data. Typically, the children will contain the
                experiment data of the sub-experiments.
            args: Additional arguments.
            kwargs: Additional keyword arguments.
        """


class GoodExperimentTransition(BaseTransitionCallable):
    """A simple experiment transition callable."""

    def __call__(
        self,
        experiment_data: ExperimentData,
        **kwargs
    ) -> int:
        """A simple experiment transition callback based on the quality of the result."""

        if experiment_data.child_data(-1).analysis_results(0).quality == "good":
            return 1
        else:
            return 0


class ChainedExperiment(CompositeExperiment):
    """An experiment that is made of several experiments that are run one after another."""

    def __init__(self, experiments: List[BaseExperiment], transition_callback: BaseTransitionCallable):
        """

        Args:
            experiments: The list of experiments to run.
            transition_callback: The method that determines which experiment to run next.
        """
        qubits = tuple()  # TODO
        super().__init__(experiments, qubits)

        self._current_index = 0
        self._transition_callback = transition_callback
        self._transition_options = self._default_transition_options()

    def current_index(self) -> int:
        """Returns the index of the experiment that is currently being run in the chain."""
        return self._current_index

    @classmethod
    def _default_transition_options(cls) -> Options:
        """Default options to control how transitions between experiments occur."""
        return Options()

    @property
    def transition_options(self) -> Options:
        """Options that can be given to the transition callback."""
        return self._transition_options

    def run(
        self,
        backend: Optional[Backend] = None,
        analysis: Optional[Union[BaseAnalysis, None]] = "default",
        timeout: Optional[float] = None,
        **run_options,
    ) -> ExperimentData:
        """Run the chained experiment by transitioning between the sub-experiments using a counter."""
        experiment_data = super().run(backend=backend, analysis=analysis, timeout=timeout, **run_options)
        return self._run_index(experiment_data)

    def _run_jobs(self, circuits: List[QuantumCircuit], **run_options) -> List[BaseJob]:
        """Do not run anything yet."""
        return []

    def _run_index(self, experiment_data):
        if self._current_index >= self.num_experiments:
            return experiment_data

        exp_data = self.component_experiment(self._current_index).run()
        experiment_data.add_child_data(exp_data)

        # transition callback thingy
        experiment_data.add_analysis_callback(self._transition_callback)

        # recursion
        experiment_data.add_analysis_callback(self._run_index)

    def _transition_callack(self, experiment_data, **kwargs):
        """Define how the index is updated."""
        self._current_index += self.experiment_options.callback(experiment_data, **kwargs)

    def circuits(self) -> List[QuantumCircuit]:
        """Returns the circuits of the current experiment."""
        return self.component_experiment(self._current_index).circuits()
