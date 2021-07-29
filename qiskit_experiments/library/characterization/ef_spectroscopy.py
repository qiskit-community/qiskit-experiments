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

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.providers.options import Options

from qiskit_experiments.library.characterization.qubit_spectroscopy import QubitSpectroscopy


class EFSpectroscopy(QubitSpectroscopy):
    """Class that runs spectroscopy on the e-f transition by scanning the frequency.

    The circuits produced by spectroscopy, i.e.

    .. parsed-literal::

                   ┌───┐┌────────────┐ ░ ┌─┐
              q_0: ┤ X ├┤ Spec(freq) ├─░─┤M├
                   └───┘└────────────┘ ░ └╥┘
        measure: 1/═══════════════════════╩═
                                          0

    """

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_analysis_options()
        options.db_parameters = {"freq": ("f12", "Hz")}

        return options

    def _template_circuit(self, freq_param) -> QuantumCircuit:
        """Return the template quantum circuit."""
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.append(Gate(name=self.__spec_gate_name__, num_qubits=1, params=[freq_param]), (0,))
        circuit.measure_active()

        return circuit
