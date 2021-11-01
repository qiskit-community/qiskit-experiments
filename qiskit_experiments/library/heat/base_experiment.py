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
Base Class for general Hamiltonian Error Amplifying Tomography experiments.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

from qiskit import circuit, QuantumCircuit
from qiskit.providers import Backend

from qiskit_experiments.framework import BaseExperiment, BatchExperiment, Options
from .base_analysis import HeatAnalysis


class HeatElement(BaseExperiment):
    """Base class of HEAT experiment elements.

    This class implements a single error amplification sequence.

    Subclasses must implement :py:meth:`_echo_circuit` to provide echo sequence that
    selectively amplifies a specific Pauli component local to the target qubit.

    """

    __analysis_class__ = HeatAnalysis

    def __init__(
        self,
        qubits: Tuple[int, int],
        prep_circ: QuantumCircuit,
        echo_circ: QuantumCircuit,
        meas_circ: QuantumCircuit,
        **kwargs
    ):
        """Create new HEAT sub experiment.

        Args:
            qubits: Index of control and target qubit, respectively.
            prep_circ: A circuit to prepare qubit before the echo sequence.
            echo_circ: A circuit to selectively amplify the specific error term.
            meas_circ: A circuit to project target qubit onto the basis of interest.
        """
        super().__init__(qubits)
        self._prep_circuit = prep_circ
        self._echo_circuit = echo_circ
        self._meas_circuit = meas_circ
        self.set_experiment_options(**kwargs)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            repetitions (Sequence[int]): A list of the number of echo repetitions.
            cr_gate (Gate): A gate instance representing the ZX(pi/2).
        """
        options = super()._default_experiment_options()
        options.repetitions = list(range(21))
        options.heat_gate = circuit.Gate("heat", num_qubits=2, params=[])

        return options

    @classmethod
    def _default_transpile_options(cls) -> Options:
        """Default transpile options."""
        options = super()._default_transpile_options()
        options.basis_gates = ["sx", "x", "rz", "cr"]

        return options

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        opt = self.experiment_options

        circs = list()
        for repetition in opt.repetitions:
            circ = circuit.QuantumCircuit(2, 1)
            circ.compose(self._prep_circuit, qubits=[0, 1], inplace=True)
            circ.barrier()
            for _ in range(repetition):
                circ.append(self.experiment_options.heat_gate, [0, 1])
                circ.compose(self._echo_circuit, qubits=[0, 1], inplace=True)
                circ.barrier()
            circ.compose(self._meas_circuit, qubits=[0, 1], inplace=True)
            circ.measure(1, 0)

            # add metadata
            circ.metadata = {
                "experiment_type": self.experiment_type,
                "qubits": self.physical_qubits,
                "xval": repetition,
            }

            circs.append(circ)

        return circs


class BaseCompositeHeat(BatchExperiment, ABC):
    """Base class of HEAT experiments.

    This class implements a batch experiment consisting of multiple HEAT element experiments
    to compute specific unitary error terms from extracted `d_theta` parameters.

    Class Attributes:
        - ``__heat_elements__``: A dictionary of fit parameter name and associated experiment
            class.

    """

    __heat_elements__ = {}

    def __init__(self, heat_experiments: List[HeatElement]):
        """Create new HEAT experiment.

        Args:
            heat_experiments: A list of configured HEAT experiments.
        """
        super().__init__(heat_experiments)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            repetitions (Sequence[int]): A list of the number of echo repetitions.
            cr_gate (Gate): A gate instance representing the ZX(pi/2).
        """
        options = super()._default_experiment_options()
        options.repetitions = list(range(21))
        options.cr_gate = circuit.Gate("cr", num_qubits=2, params=[])

        return options

    def set_experiment_options(self, **fields):
        """Set the analysis options for :meth:`run` method.

        Same experiment options are applied to all subset HEAT experiments.

        Args:
            fields: The fields to update the options
        """
        for comp_exp in self.component_experiment():
            comp_exp.set_experiment_options(**fields)

        super().set_experiment_options(**fields)

    @classmethod
    def _default_transpile_options(cls) -> Options:
        """Default transpile options."""
        options = super()._default_transpile_options()
        options.basis_gates = ["sx", "x", "rz", "cr"]

        return options

    def set_transpile_options(self, **fields):
        """Set the transpiler options for :meth:`run` method.

        Same transpile options are applied to all subset HEAT experiments.

        Args:
            fields: The fields to update the options
        """
        # TODO wait for #380 to apply individual transpile options to nested experiments
        for comp_exp in self.component_experiment():
            comp_exp.set_transpile_options(**fields)

        super().set_transpile_options(**fields)
