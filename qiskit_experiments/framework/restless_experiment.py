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

"""Restless mix in class."""

from abc import ABC
from qiskit_experiments.framework.base_experiment import BaseExperiment

from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.data_processing import nodes


class RestlessEnabledExperiment(BaseExperiment, ABC):
    """Restless enabled class."""

    def enable_restless(self, rep_delay: float):
        """Enables a restless experiment by setting the restless run options and
        the restless data processor.

            Args:
                rep_delay: The repetition delay.
        """
        outcome = self.analysis.options.get("outcome", "1" * self.num_qubits)
        self.set_run_options(rep_delay=rep_delay, init_qubit=False, memory=True, meas_level=2)
        self.analysis.set_options(data_processor=self._get_restless_processor(outcome))
        # print([
        #     self.backend.properties().qubit_property(physical_qubit)["T1"][0]
        #     for physical_qubit in self.physical_qubits
        # ])

    def _get_restless_processor(self, outcome: str) -> DataProcessor:
        return DataProcessor(
                 "memory",
                 [
                     nodes.RestlessToCounts(self.num_qubits),
                     nodes.Probability(outcome),
                 ],
             )

    def _is_restless(self, rep_delay: float) -> bool:
        """Checks if the specified repetition delay is much smaller than the T1
        times of the physical qubits in the experiment.

        Args:
            rep_delay: The repetition delay.

        Returns:
            True if the repetition delay is much smaller than the qubit T1 times.
        """

        esp_enabled = self.analysis.options.get("use_measure_esp", False)

        # Todo: include properties in mock backend.
        t1_values = [
            self.backend.properties().qubits[physical_qubit][0].value * 1e-6
            for physical_qubit in self.physical_qubits
        ]

        t1_values = [
            self.backend.properties().qubit_property(physical_qubit)["T1"][0]
            for physical_qubit in self.physical_qubits
        ]

        if all(rep_delay / t1_value < 1. for t1_value in t1_values):
            if esp_enabled:
                raise DataProcessorError(
                    "Restless experiments are not compatible with the excited "
                    "state promotion readout analysis option."
                )
            return True

        return False
