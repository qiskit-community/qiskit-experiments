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

from typing import Iterable, Optional
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.circuit import Gate

from qiskit_experiments.curve_analysis import ParameterRepr
from qiskit_experiments.library.characterization.qubit_spectroscopy import QubitSpectroscopy


class EFSpectroscopy(QubitSpectroscopy):
    """Class that runs spectroscopy on the e-f transition by scanning the frequency.

    # section: overview
        The circuits produced by spectroscopy, i.e.

        .. parsed-literal::

                       ┌───┐┌────────────┐ ░ ┌─┐
                  q_0: ┤ X ├┤ Spec(freq) ├─░─┤M├
                       └───┘└────────────┘ ░ └╥┘
            measure: 1/═══════════════════════╩═
                                              0

    """

    def __init__(
        self,
        qubit: int,
        frequencies: Iterable[float],
        backend: Optional[Backend] = None,
        absolute: bool = True,
    ):
        super().__init__(qubit, frequencies, backend=backend, absolute=absolute)
        self.analysis.set_options(result_parameters=[ParameterRepr("freq", "f12")])

    def _template_circuit(self, freq_param) -> QuantumCircuit:
        """Return the template quantum circuit."""
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.append(Gate(name=self.__spec_gate_name__, num_qubits=1, params=[freq_param]), (0,))
        circuit.measure_active()

        return circuit
