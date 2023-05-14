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

"""Spectroscopy for the e-f transition."""

from typing import Iterable, Optional, Sequence
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.circuit import Gate

from qiskit_experiments.curve_analysis import ParameterRepr
from qiskit_experiments.library.characterization.qubit_spectroscopy import QubitSpectroscopy
from qiskit_experiments.warnings import qubit_deprecate


class EFSpectroscopy(QubitSpectroscopy):
    """A spectroscopy experiment to obtain a frequency sweep of the qubit's e-f transition.

    # section: overview
        The circuits produced by spectroscopy, i.e.

        .. parsed-literal::

                       ┌───┐┌────────────┐ ░ ┌─┐
                  q_0: ┤ X ├┤ Spec(freq) ├─░─┤M├
                       └───┘└────────────┘ ░ └╥┘
            measure: 1/═══════════════════════╩═
                                              0

    """

    @qubit_deprecate()
    def __init__(
        self,
        physical_qubits: Sequence[int],
        frequencies: Iterable[float],
        backend: Optional[Backend] = None,
        absolute: bool = True,
    ):
        super().__init__(physical_qubits, frequencies, backend=backend, absolute=absolute)
        self.analysis.set_options(result_parameters=[ParameterRepr("freq", "f12")])

    def _template_circuit(self, freq_param) -> QuantumCircuit:
        """Return the template quantum circuit."""
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.append(Gate(name=self.__spec_gate_name__, num_qubits=1, params=[freq_param]), (0,))
        circuit.measure_active()

        return circuit
