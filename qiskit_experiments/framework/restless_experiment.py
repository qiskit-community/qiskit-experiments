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
from qiskit_experiments.framework.base_experiment import BaseExperiment

from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.data_processing import nodes


class RestlessEnabledExperiment(BaseExperiment, ABC):
    """Restless enabled experiment class.

    A restless enabled experiment is an experiment that can be run in a restless
    measurement setting. In restless measurements, the qubit is not reset after
    each measurement. Instead, the outcome of the previous quantum non-demolition
    measurement is the initial state for the current circuit. Restless measurements
    therefore require special data processing which is provided by sub-classes of
    the :code:`RestlessNode`. Restless experiments are a fast alternative for
    several calibration and characterization tasks, for details see
    https://arxiv.org/pdf/2202.06981.pdf.
    This class makes it possible for users to enter a restless run-mode without having
    to set all the required run options and the data processor. Furthermore, subclasses
    can override the :meth:`_get_restless_processor` method if they require more
    complex restless data processing such as two-qubit calibrations. In addition, this
    class makes it easy to determine if restless measurements are supported for a given
    experiments.
    """

    def enable_restless(self, rep_delay: float, override_restless_processor: bool = False):
        """Enables a restless experiment by setting the restless run options and
        the restless data processor.

            Args:
                rep_delay: The repetition delay. This is the delay between a measurement
                    and the subsequent quantum circuit. Since IBM Quantum backends have
                    dynamic repetition rates, the repetition delay can be set to a small
                    value which is required for restless experiments. Typical values are
                    1 us or less.
                override_restless_processor: If True, a data processor that is specified in the
                    analysis options of the experiment can override the restless data
                    processor.

            Raises:
                DataProcessorError: if the rep_delay is equal to or greater than the
                    T1 time of one of the physical qubits in the experiment.
                DataProcessorError: if excited state promotion readout is enabled in the
                    restless setting.
        """

        # If the excited state promotion readout analysis option is enabled,
        # it will be set to False because it is not compatible with a
        # restless experiment.

        if self._t1_check(rep_delay):
            if not self.analysis.options.get("data_processor", None):
                self.set_run_options(
                    rep_delay=rep_delay,
                    init_qubit=False,
                    memory=True,
                    meas_level=2,
                    use_measure_esp=False,
                )
                self.analysis.set_options(data_processor=self._get_restless_processor())
            else:
                if override_restless_processor:
                    self.set_run_options(
                        rep_delay=rep_delay,
                        init_qubit=False,
                        memory=True,
                        meas_level=2,
                        use_measure_esp=False,
                    )
                else:
                    raise DataProcessorError(
                        "Cannot enable restless. Data processor has already been set and "
                        "override_restless_processor is False."
                    )
        else:
            raise DataProcessorError(
                f"The specified repetition delay {rep_delay} is equal to or greater "
                f"than the T1 time of one of the physical qubits"
                f"{self.physical_qubits} in the experiment. Consider choosing "
                f"a smaller repetition delay for the restless experiment."
            )

    def _get_restless_processor(self) -> DataProcessor:
        """Returns the restless experiments data processor.

        Notes:
            Sub-classes can override this method if they need more complex data processing.
        """
        outcome = self.analysis.options.get("outcome", "1" * self.num_qubits)
        return DataProcessor(
            "memory",
            [
                nodes.RestlessToCounts(self.num_qubits),
                nodes.Probability(outcome),
            ],
        )

    def _t1_check(self, rep_delay: float) -> bool:
        """Check that repetition delay < T1 of the physical qubits in the experiment.

        Args:
            rep_delay: The repetition delay. This is the delay between a measurement
                    and the subsequent quantum circuit.

        Returns:
            True if the repetition delay is smaller than the qubit T1 times.
        """

        t1_values = [
            self.backend.properties().qubit_property(physical_qubit)["T1"][0]
            for physical_qubit in self.physical_qubits
        ]

        if all(rep_delay / t1_value < 1.0 for t1_value in t1_values):
            return True

        return False
