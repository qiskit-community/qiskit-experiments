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
from .heat_analysis import HeatAnalysis


class BaseHeat(BaseExperiment, ABC):
    """Base class of HEAT experiments.

    Subclasses must implement :py:meth:`_echo_circuit` to provide echo sequence that
    selectively amplifies specific error component.
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
        options = super()._default_experiment_options()
        options.repetitions = list(range(21))
        options.cr_gate = circuit.Gate("cr", num_qubits=2, params=[])

        return options

    @classmethod
    def _default_transpile_options(cls) -> Options:
        options = super()._default_transpile_options()
        options.basis_gates = ["sx", "x", "rz", "cr"]

        return options

    @abstractmethod
    def _echo_circuit(self) -> QuantumCircuit:
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
                circ.compose(self._echo_circuit(), qubits=[0, 1], inplace=True)
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


class HeatY0(BaseHeat):
    """"""

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

    def _echo_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.append(self.experiment_options.cr_gate, [0, 1])
        circ.y(1)

        return circ


class HeatY1(BaseHeat):
    """"""

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

    def _echo_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.append(self.experiment_options.cr_gate, [0, 1])
        circ.y(1)

        return circ


class HeatZ0(BaseHeat):
    """"""

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

    def _echo_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.append(self.experiment_options.cr_gate, [0, 1])
        circ.z(1)

        return circ


class HeatZ1(BaseHeat):
    """"""

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

    def _echo_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.append(self.experiment_options.cr_gate, [0, 1])
        circ.z(1)

        return circ


class HeatCompositeZY(BatchExperiment):
    """"""

    def __init__(self, qubits: Tuple[int, int]):

        # configure sub echo experiments.
        exp_y0 = HeatY0(qubits=qubits)
        exp_y1 = HeatY1(qubits=qubits)
        exp_z0 = HeatZ0(qubits=qubits)
        exp_z1 = HeatZ1(qubits=qubits)

        super().__init__(experiments=[exp_y0, exp_y1, exp_z0, exp_z1])

    @classmethod
    def set_transpile_options(self, **fields):
        for comp_exp in self.component_experiment():
            comp_exp.set_transpile_options(**fields)

        super().set_transpile_options(**fields)
