# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Quantum State Tomography experiment
"""

from typing import Union, Optional, List, Sequence
from qiskit.providers.backend import Backend
from qiskit.circuit import QuantumCircuit, Instruction, Clbit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_experiments.framework import BatchExperiment, BaseAnalysis
from qiskit_experiments.library.characterization.local_readout_error import LocalReadoutError
from .qst_experiment import StateTomography
from .mit_tomography_analysis import MitigatedTomographyAnalysis
from . import basis


class MitigatedStateTomography(BatchExperiment):
    """A batched experiment to characterize readout error then perform state tomography
    for doing readout error mitigated state tomography.

    # section: overview
        Readout error mitigated quantum state tomography is a batch
        experiment consisting of a :class:`~.LocalReadoutError` characterization
        experiments, followed by a :class:`~.StateTomography` experiment.

        During analysis the assignment matrix local readout error model is
        used to automatically construct a noisy Pauli measurement basis for
        performing readout error mitigated state tomography fitting.

    # section: note
        Performing readout error mitigation full state tomography on an
        `N`-qubit circuit requires running 2 readout error characterization
        circuits and :math:`3^N` measurement circuits using the Pauli
        measurement basis.

    # section: analysis_ref
        :py:class:`MitigatedTomographyAnalysis`

    # section: see_also
        * :py:class:`qiskit_experiments.library.tomography.StateTomography`
        * :py:class:`qiskit_experiments.library.characterization.LocalReadoutError`

    # section: example
        .. jupyter-execute::
            :hide-code:

            # backend
            from qiskit_aer import AerSimulator
            from qiskit_ibm_runtime.fake_provider import FakePerth
            from qiskit_aer.noise import NoiseModel

            noise_model = NoiseModel.from_backend(FakePerth(),
                                                  thermal_relaxation=False,
                                                  gate_error=False,
                                                  readout_error=True,
                                                 )

            backend = AerSimulator.from_backend(FakePerth(), noise_model=noise_model)

        .. jupyter-execute::

            from qiskit import QuantumCircuit
            from qiskit_experiments.library import MitigatedStateTomography
            from qiskit.visualization import plot_state_city

            nq = 2
            qc_ghz = QuantumCircuit(nq)
            qc_ghz.h(0)
            qc_ghz.s(0)

            for i in range(1, nq):
                qc_ghz.cx(0, i)

            mitqstexp = MitigatedStateTomography(qc_ghz)
            mitqstexp.set_run_options(shots=1000)
            mitqstdata = mitqstexp.run(backend=backend,
                                       seed_simulator=100,).block_for_results()
            state_result = mitqstdata.analysis_results("state")
            plot_state_city(state_result.value, title="mitigated Density Matrix")
    """

    def __init__(
        self,
        circuit: Union[QuantumCircuit, Instruction, BaseOperator],
        backend: Optional[Backend] = None,
        physical_qubits: Optional[Sequence[int]] = None,
        measurement_indices: Optional[Sequence[int]] = None,
        basis_indices: Optional[Sequence[List[int]]] = None,
        conditional_circuit_clbits: Union[bool, Sequence[int], Sequence[Clbit]] = False,
        analysis: Union[BaseAnalysis, None, str] = "default",
    ):
        """Initialize a quantum process tomography experiment.

        Args:
            circuit: the quantum process circuit. If not a quantum circuit
                it must be a class that can be appended to a quantum circuit.
            backend: The backend to run the experiment on.
            physical_qubits: Optional, the physical qubits for the initial state circuit.
                If None this will be qubits [0, N) for an N-qubit circuit.
            measurement_indices: Optional, the `physical_qubits` indices to be measured.
                If None all circuit physical qubits will be measured.
            basis_indices: Optional, a list of basis indices for generating partial
                tomography measurement data. Each item should be given as a list of
                measurement basis configurations ``[m[0], m[1], ...]`` where ``m[i]``
                is the measurement basis index for qubit-i. If not specified full
                tomography for all indices of the measurement basis will be performed.
            conditional_circuit_clbits: Optional, the clbits in the source circuit to
                be conditioned on when reconstructing the state. If True all circuit
                clbits will be conditioned on. Enabling this will return a list of
                reconstructed state components conditional on the values of these clbit
                values.
            analysis: Optional, a custom tomography analysis instance to use.
                If ``"default"`` :class:`~.ProcessTomographyAnalysis` will be
                used. If None no analysis instance will be set.
        """
        tomo_exp = StateTomography(
            circuit,
            backend=backend,
            physical_qubits=physical_qubits,
            measurement_basis=basis.PauliMeasurementBasis(),
            measurement_indices=measurement_indices,
            basis_indices=basis_indices,
            conditional_circuit_clbits=conditional_circuit_clbits,
            analysis=analysis,
        )

        roerror_exp = LocalReadoutError(
            tomo_exp.physical_qubits,
            backend=backend,
        )

        if analysis is None:
            mit_analysis = (None,)
        else:
            mit_analysis = MitigatedTomographyAnalysis(roerror_exp.analysis, tomo_exp.analysis)

        super().__init__(
            [roerror_exp, tomo_exp], backend=backend, flatten_results=True, analysis=mit_analysis
        )
