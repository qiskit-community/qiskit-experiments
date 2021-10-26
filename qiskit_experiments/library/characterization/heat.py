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
Hamiltonian Error Amplifying Tomography Experiments.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import numpy as np
from qiskit import circuit, QuantumCircuit
from qiskit.providers import Backend

from qiskit_experiments.curve_analysis import ParameterRepr
from qiskit_experiments.framework import BaseExperiment, BatchExperiment, Options
from .heat_analysis import HeatAnalysis, HeatYAnalysis, HeatZAnalysis


class BaseHeat(BaseExperiment, ABC):
    """Base class of HEAT experiment elements.

    This class implements a single error amplification sequence.

    Subclasses must implement :py:meth:`_echod_circuit` to provide echo sequence that
    selectively amplifies a specific Pauli component local to the target qubit.

    """

    __analysis_class__ = HeatAnalysis

    def __init__(
        self,
        qubits: Tuple[int, int],
        **kwargs
    ):
        """Create new HEAT sub experiment."""
        super().__init__(qubits)
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
        options.cr_gate = circuit.Gate("cr", num_qubits=2, params=[])

        return options

    @classmethod
    def _default_transpile_options(cls) -> Options:
        """Default transpile options."""
        options = super()._default_transpile_options()
        options.basis_gates = ["sx", "x", "rz", "cr"]

        return options

    @abstractmethod
    def _echoed_circuit(self) -> QuantumCircuit:
        pass

    def _prep_circuit(self) -> QuantumCircuit:
        circ = QuantumCircuit(2)
        return circ

    def _meas_circuit(self) -> QuantumCircuit:
        circ = QuantumCircuit(2)
        return circ

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        opt = self.experiment_options

        circs = list()
        for repetition in opt.repetitions:
            circ = circuit.QuantumCircuit(2, 1)
            circ.compose(self._prep_circuit(), qubits=[0, 1], inplace=True)
            circ.barrier()
            for _ in range(repetition):
                circ.compose(self._echoed_circuit(), qubits=[0, 1], inplace=True)
                circ.barrier()
            circ.compose(self._meas_circuit(), qubits=[0, 1], inplace=True)
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
        - ``__heat_elements__``: A list of HEAT element experiments.

    """

    __heat_elements__ = []

    def __init__(self, qubits: Tuple[int, int]):
        """Create new HEAT experiment.

        Args:
            qubits: A tuple of control and target qubit index.
        """
        heat_experiments = [expcls(qubits=qubits) for expcls in self.__heat_elements__]
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

        super().set_transpile_options(**fields)

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


class HeatElementY0(BaseHeat):
    r"""A single error amplification sequence of Y error with the control qubit in 0 state.

    # section: overview

        This experiment generates a following circuit.

        .. parsed-literal::

                             ░ ┌─────┐      ░
            q_0: ────────────░─┤0    ├──────░────
                 ┌─────────┐ ░ │  cr │┌───┐ ░ ┌─┐
            q_1: ┤ Ry(π/2) ├─░─┤1    ├┤ Y ├─░─┤M├
                 └─────────┘ ░ └─────┘└───┘ ░ └╥┘
            c: 1/══════════════════════════════╩═
                                               0

        Circuit block in the middle is repeated N times to amplify the target error.
        The ``cr`` gate represents a unitary of :math:`ZX(\pi/2)`, and its pulse schedule
        should be provided by users.

    """

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_analysis_options()
        options.result_parameters = [ParameterRepr("d_theta", "d_heat_y0", "rad")]

        return options

    def _prep_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.ry(np.pi/2, 1)

        return circ

    def _echoed_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.append(self.experiment_options.cr_gate, [0, 1])
        circ.y(1)

        return circ


class HeatElementY1(BaseHeat):
    r"""A single error amplification sequence of Y error with the control qubit in 1 state.

    # section: overview

        This experiment generates a following circuit.

        .. parsed-literal::

                    ┌───┐    ░ ┌─────┐      ░
            q_0: ───┤ X ├────░─┤0    ├──────░────
                 ┌──┴───┴──┐ ░ │  cr │┌───┐ ░ ┌─┐
            q_1: ┤ Ry(π/2) ├─░─┤1    ├┤ Y ├─░─┤M├
                 └─────────┘ ░ └─────┘└───┘ ░ └╥┘
            c: 1/══════════════════════════════╩═
                                               0

        Circuit block in the middle is repeated N times to amplify the target error.
        The ``cr`` gate represents a unitary of :math:`ZX(\pi/2)`, and its pulse schedule
        should be provided by users.

    """

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_analysis_options()
        options.result_parameters = [ParameterRepr("d_theta", "d_heat_y1", "rad")]

        return options

    def _prep_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.x(0)
        circ.ry(np.pi/2, 1)

        return circ

    def _echoed_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.append(self.experiment_options.cr_gate, [0, 1])
        circ.y(1)

        return circ


class HeatElementZ0(BaseHeat):
    r"""A single error amplification sequence of Z error with the control qubit in 0 state.

    # section: overview

        This experiment generates a following circuit.

        .. parsed-literal::

                             ░ ┌─────┐      ░
            q_0: ────────────░─┤0    ├──────░───────────────
                 ┌─────────┐ ░ │  cr │┌───┐ ░ ┌─────────┐┌─┐
            q_1: ┤ Ry(π/2) ├─░─┤1    ├┤ Z ├─░─┤ Rx(π/2) ├┤M├
                 └─────────┘ ░ └─────┘└───┘ ░ └─────────┘└╥┘
            c: 1/═════════════════════════════════════════╩═
                                                          0

        Circuit block in the middle is repeated N times to amplify the target error.
        The ``cr`` gate represents a unitary of :math:`ZX(\pi/2)`, and its pulse schedule
        should be provided by users.

    """

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_analysis_options()
        options.result_parameters = [ParameterRepr("d_theta", "d_heat_z0", "rad")]

        return options

    def _prep_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.ry(np.pi/2, 1)

        return circ

    def _meas_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.rx(np.pi/2, 1)

        return circ

    def _echoed_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.append(self.experiment_options.cr_gate, [0, 1])
        circ.z(1)

        return circ


class HeatElementZ1(BaseHeat):
    r"""A single error amplification sequence of Z error with the control qubit in 1 state.

    # section: overview

        This experiment generates a following circuit.

        .. parsed-literal::

                    ┌───┐    ░ ┌─────┐      ░
            q_0: ───┤ X ├────░─┤0    ├──────░───────────────
                 ┌──┴───┴──┐ ░ │  cr │┌───┐ ░ ┌─────────┐┌─┐
            q_1: ┤ Ry(π/2) ├─░─┤1    ├┤ Z ├─░─┤ Rx(π/2) ├┤M├
                 └─────────┘ ░ └─────┘└───┘ ░ └─────────┘└╥┘
            c: 1/═════════════════════════════════════════╩═
                                                          0

        Circuit block in the middle is repeated N times to amplify the target error.
        The ``cr`` gate represents a unitary of :math:`ZX(\pi/2)`, and its pulse schedule
        should be provided by users.

    """

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_analysis_options()
        options.result_parameters = [ParameterRepr("d_theta", "d_heat_z1", "rad")]

        return options

    def _prep_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.x(0)
        circ.ry(np.pi / 2, 1)

        return circ

    def _meas_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.rx(np.pi / 2, 1)

        return circ

    def _echoed_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.append(self.experiment_options.cr_gate, [0, 1])
        circ.z(1)

        return circ


class HeatYError(BaseCompositeHeat):
    """HEAT experiments for Y error amplification.

    # section: overview
        TODO
    """

    __heat_elements__ = [HeatElementY0, HeatElementY1]
    __analysis_class__ = HeatYAnalysis


class HeatZError(BaseCompositeHeat):
    """HEAT experiments for Z error amplification.

    # section: overview
        TODO
    """

    __heat_elements__ = [HeatElementZ0, HeatElementZ1]
    __analysis_class__ = HeatZAnalysis
