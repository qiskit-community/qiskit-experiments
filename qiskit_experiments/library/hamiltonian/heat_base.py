# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
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

from typing import List, Tuple, Optional

from qiskit import circuit, QuantumCircuit
from qiskit.providers import Backend

from qiskit_experiments.framework import BaseExperiment, Options
from .heat_analysis import HeatElementAnalysis


class HeatElement(BaseExperiment):
    """Base class of HEAT experiment elements.

    # section: overview

        Hamiltonian error amplifying tomography (HEAT) is designed to amplify
        the dynamics of entangler circuit on a qubit along a specific axis.

        The basic form of HEAT circuit is represented as follows.

        .. parsed-literal::

                                    (xN)
                 ┌───────┐ ░ ┌───────┐┌───────┐ ░ ┌───────┐
            q_0: ┤0      ├─░─┤0      ├┤0      ├─░─┤0      ├───
                 │  prep │ ░ │  heat ││  echo │ ░ │  meas │┌─┐
            q_1: ┤1      ├─░─┤1      ├┤1      ├─░─┤1      ├┤M├
                 └───────┘ ░ └───────┘└───────┘ ░ └───────┘└╥┘
            c: 1/═══════════════════════════════════════════╩═
                                                            0

        The circuit in the middle is repeated ``N`` times to amplify the Hamiltonian
        coefficients along a specific axis of the second qubit. The ``prep`` circuit is
        carefully chosen based on the generator of the ``heat`` gate under consideration.
        The ``echo`` and ``meas`` circuits depend on the axis of the error to amplify.
        Only the second qubit is measured.

        The measured qubit population containing the amplified error typically has contributions
        from both local (e.g. IZ) and non-local rotations (e.g. ZX).
        Thus, multiple error amplification experiments with different control qubit states
        are usually combined to resolve these rotation terms.
        This experiment just provides a single error amplification sequence, and therefore
        you must combine multiple instances instantiated with different ``prep``, ``echo``,
        and ``meas`` circuits designed to resolve the different error terms.
        This class can be wrapped with hard-coded circuits to define new experiment class
        to provide HEAT experiment with respect to the error axis and Hamiltonian of interest.

        The ``heat`` gate is a custom gate representing the entangling pulse sequence.
        One must thus provide its definition through the backend or a custom transpiler
        configuration, i.e. with the instruction schedule map. This gate name can be overridden
        via the experiment options.

    # section: note

        This class is usually not exposed to end users.
        The developer of a new HEAT experiment must design the amplification sequences and
        create instances of this class implicitly in the batch experiment.

    # section: analysis_ref
        :py:class:`HeatElementAnalysis`

    # section: reference
        .. ref_arxiv:: 1 2007.02925
    """

    def __init__(
        self,
        qubits: Tuple[int, int],
        prep_circ: QuantumCircuit,
        echo_circ: QuantumCircuit,
        meas_circ: QuantumCircuit,
        backend: Optional[Backend] = None,
        **kwargs,
    ):
        """Create new HEAT sub experiment.

        Args:
            qubits: Index of control and target qubit, respectively.
            prep_circ: A circuit to prepare qubit before the echo sequence.
            echo_circ: A circuit to selectively amplify the specific error term.
            meas_circ: A circuit to project target qubit onto the basis of interest.
            backend: Optional, the backend to run the experiment on.

        Keyword Args:
            See :meth:`experiment_options` for details.
        """
        super().__init__(qubits=qubits, backend=backend, analysis=HeatElementAnalysis())
        self.set_experiment_options(**kwargs)

        # These are not user configurable options. Be frozen once assigned.
        self._prep_circuit = prep_circ
        self._echo_circuit = echo_circ
        self._meas_circuit = meas_circ

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            repetitions (Sequence[int]): A list of the number of echo repetitions.
            heat_gate (Gate): A gate instance representing the entangler sequence.
        """
        options = super()._default_experiment_options()
        options.repetitions = list(range(21))
        options.heat_gate = circuit.Gate("heat", num_qubits=2, params=[])

        return options

    @classmethod
    def _default_transpile_options(cls) -> Options:
        """Default transpile options."""
        options = super()._default_transpile_options()
        options.basis_gates = ["sx", "x", "rz", "heat"]
        options.optimization_level = 1

        return options

    def circuits(self) -> List[QuantumCircuit]:
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
