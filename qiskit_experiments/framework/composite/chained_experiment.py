
from abc import ABC, abstractmethod
from typing import List, Optional, Union

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.providers import BaseJob
from qiskit.providers.options import Options

from qiskit_experiments.framework.base_experiment import BaseExperiment, ExperimentData, BaseAnalysis
from qiskit_experiments.framework.composite.composite_experiment import CompositeExperiment
from qiskit_experiments.exceptions import AnalysisError


class BaseTransitionCallable(ABC):
    """A method to determine how to transition between experiments.

    TODO These should be serializable.
    """

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
    """An experiment that is made of several experiments that are run one after another.

    The chained experiment works using analysis callback functions and an index to the experiment
    in the chain of experiments that needs to be run. Each time an experiment is run there are
    two analysis callback functions that are added to the experiment data as follows.

    1. An experiment transition callback. This is a function thar determines relative changes to
    the experiment index. For example, if the next experiment in the chain is to be executed then
    this function returns 1. However, if the current experiment is to be repeated thes this
    function will return 0.

    2. A callback to launch the next experiment based on the value of the experiment index.

    The experiment data from this class will contain all the sub-experiments as children.
    """

    def __init__(
        self,
        experiments: List[BaseExperiment],
        transition_callback: Optional[BaseTransitionCallable] = None
    ):
        """

        Args:
            experiments: The list of experiments to run.
            transition_callback: The method that determines which experiment to run next.
        """
        qubits = set()
        for exp in experiments:
            qubits.update(exp.physical_qubits)

        super().__init__(experiments, list(qubits))

        # The index that points to the experiment that is running.
        self._current_index = 0

        # A counter to safe-guard against endless experiment runs.
        self._number_of_runs = 0

        # The arguments for the transition callback function.
        self._transition_options = self._default_transition_options()

        if transition_callback:
            self.set_experiment_options(callback=transition_callback)

    def current_index(self) -> int:
        """Returns the index of the experiment that is currently being run in the chain."""
        return self._current_index

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """The default options for this experiment.

        Experiment Options:
            callback: The function that determines how to transition between experiments
                in the chain.
            max_runs: The maximum number of times that the run method of the experiments
                can be called. By default this value is set to 100.
        """
        options = super()._default_experiment_options()
        options.callback = GoodExperimentTransition
        options.max_runs = 100
        return options

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

    def _run_index(self, experiment_data: ExperimentData):
        """Run the experiment at the current index.

        Args:
            experiment_data: The experiment data to which the experiment data of the
                run experiment will be added as a child.
        """
        if self._current_index >= self.num_experiments:
            return experiment_data

        if self._number_of_runs > self.experiment_options.max_runs:
            raise AnalysisError("The maximum allowed number of runs has been exceeded.")

        exp_data = self.component_experiment(self._current_index).run()
        experiment_data.add_child_data(exp_data)

        # Add the transition callback as an analysis callback to the experiment data
        experiment_data.add_analysis_callback(self._transition_callback)

        # recursion
        experiment_data.add_analysis_callback(self._run_index)

    def _transition_callback(self, experiment_data, **kwargs):
        """Define how the index is updated."""
        self._current_index += self.experiment_options.callback(experiment_data, **kwargs)

    def circuits(self) -> List[QuantumCircuit]:
        """Returns the circuits of the current experiment."""
        return self.component_experiment(self._current_index).circuits()
