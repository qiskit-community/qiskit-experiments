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

"""Restless experiment class."""

from abc import ABC
from typing import Union

from qiskit.providers import BaseBackend
from qiskit.test.mock.fake_backend import FakeBackend
from qiskit_experiments.framework.base_experiment import BaseExperiment

from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.data_processing import nodes


class RestlessEnabledExperiment(BaseExperiment, ABC):
    """Restless experiment class."""

    def enable_restless(self, rep_delay: float, backend: Union[BaseBackend, FakeBackend]):
        """Enables a restless experiment by setting the restless run options and
        the restless data processor.

            Args:
                rep_delay: The repetition delay.
                backend: The experiment backend.

            Raises:
                DataProcessorError: if the rep_delay is equal to or greater than the
                    T1 time of one of the physical qubits in the experiment.
        """
        if self._is_restless(rep_delay, backend):
            self.set_run_options(rep_delay=rep_delay, init_qubit=False, memory=True, meas_level=2)
            if self.analysis.options.get("data_processor", None):
                pass
            else:
                self.analysis.set_options(data_processor=self._get_restless_processor())
        else:
            raise DataProcessorError(
                f"The specified repetition delay {rep_delay} is equal to or greater"
                f"than the T1 time of one of the physical qubits"
                f"{self.physical_qubits} in the experiment. Consider choosing"
                f"a smaller repetition delay for the restless experiment."
            )

    def _get_restless_processor(self) -> DataProcessor:
        """Returns the restless experiments data processor."""
        outcome = self.analysis.options.get("outcome", "1" * self.num_qubits)
        return DataProcessor(
            "memory",
            [
                nodes.RestlessToCounts(self.num_qubits),
                nodes.Probability(outcome),
            ],
        )

    def _is_restless(self, rep_delay: float, backend: Union[BaseBackend, FakeBackend]) -> bool:
        """Checks if the specified repetition delay is smaller than the T1
        times of the physical qubits in the experiment.

        Args:
            rep_delay: The repetition delay.
            backend: The experiment backend.

        Returns:
            True if the repetition delay is smaller than the qubit T1 times.

        Raises:
            DataProcessorError: if excited state promotion readout is enabled in the restless setting.
        """

        t1_values = [
            backend.properties().qubit_property(physical_qubit)["T1"][0]
            for physical_qubit in self.physical_qubits
        ]

        if all(rep_delay / t1_value < 1.0 for t1_value in t1_values):
            esp_enabled = self.analysis.options.get("use_measure_esp", False)
            if esp_enabled:
                raise DataProcessorError(
                    "Restless experiments are not compatible with the excited "
                    "state promotion readout analysis option."
                )
            return True

        return False
