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
Quantum Process Tomography experiment
"""

from typing import Union, Optional, Iterable, List, Tuple, Sequence
from qiskit.providers.backend import Backend
from qiskit.circuit import QuantumCircuit, Instruction, Clbit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_experiments.framework import BatchExperiment, BaseAnalysis
from qiskit_experiments.library.characterization.local_readout_error import LocalReadoutError
from .qpt_experiment import ProcessTomography
from .mit_tomography_analysis import MitigatedTomographyAnalysis
from . import basis


class MitigatedProcessTomography(BatchExperiment):
    """A batched experiment to characterize readout error then perform process tomography
    for doing readout error mitigated process tomography.

    # section: overview
        Readout error mitigated Quantum process tomography is a batch
        experiment consisting of a :class:`~.LocalReadoutError` characterization
        experiments, followed by a :class:`~.ProcessTomography` experiment.

        During analysis the assignment matrix local readout error model is
        used to automatically construct a noisy Pauli measurement basis for
        performing readout error mitigated process tomography fitting.

    # section: note
        Performing readout error mitigation full process tomography on an
        `N`-qubit circuit requires running 2 readout error characterization
        circuits and :math:`4^N 3^N` measurement circuits using the Pauli
        preparation and measurement bases.

    # section: analysis_ref
        :py:class:`MitigatedTomographyAnalysis`

    # section: see_also
        * :py:class:`qiskit_experiments.library.tomography.ProcessTomography`
        * :py:class:`qiskit_experiments.library.characterization.LocalReadoutError`

    # section: example
        .. jupyter-execute::
            :hide-code:

            # backend
            from qiskit_aer import AerSimulator
            from qiskit_ibm_runtime.fake_provider import FakePerth

            backend = AerSimulator.from_backend(FakePerth())

        .. jupyter-execute::

            import numpy as np
            from qiskit import QuantumCircuit
            from qiskit_experiments.library import MitigatedProcessTomography

            num_qubits = 2
            qc_ghz = QuantumCircuit(num_qubits)
            qc_ghz.h(0)
            qc_ghz.s(0)

            for i in range(1, num_qubits):
                qc_ghz.cx(0, i)

            mitqptexp = MitigatedProcessTomography(qc_ghz)
            mitqptexp.set_run_options(shots=1000)
            mitqptdata = mitqptexp.run(backend=backend,
                                       seed_simulator=100,).block_for_results()
            mitigated_choi_out = mitqptdata.analysis_results("state").value

            # extracting a densitymatrix from mitigated_choi_out
            from qiskit.visualization import plot_state_city
            import qiskit.quantum_info as qinfo

            _rho_exp_00 = np.array([[None, None, None, None],
                                    [None, None, None, None],
                                    [None, None, None, None],
                                    [None, None, None, None]])

            for i in range(4):
                for j in range(4):
                    _rho_exp_00[i][j] = mitigated_choi_out.data[i][j]

            mitigated_rho_exp_00 = qinfo.DensityMatrix(_rho_exp_00)
            display(plot_state_city(mitigated_rho_exp_00, title="mitigated Density Matrix"))
    """

    def __init__(
        self,
        circuit: Union[QuantumCircuit, Instruction, BaseOperator],
        backend: Optional[Backend] = None,
        physical_qubits: Optional[Sequence[int]] = None,
        measurement_indices: Optional[Sequence[int]] = None,
        preparation_indices: Optional[Sequence[int]] = None,
        basis_indices: Optional[Iterable[Tuple[List[int], List[int]]]] = None,
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
            preparation_indices: Optional, the `physical_qubits` indices to be prepared.
                If None all circuit physical qubits will be prepared.
            basis_indices: Optional, a list of basis indices for generating partial
                tomography measurement data. Each item should be given as a pair of
                lists of preparation and measurement basis configurations
                ``([p[0], p[1], ...], [m[0], m[1], ...])``, where ``p[i]`` is the
                preparation basis index, and ``m[i]`` is the measurement basis index
                for qubit-i. If not specified full tomography for all indices of the
                preparation and measurement bases will be performed.
            conditional_circuit_clbits: Optional, the clbits in the source circuit to
                be conditioned on when reconstructing the state. If True all circuit
                clbits will be conditioned on. Enabling this will return a list of
                reconstructed state components conditional on the values of these clbit
                values.
            analysis: Optional, a custom tomography analysis instance to use.
                If ``"default"`` :class:`~.ProcessTomographyAnalysis` will be
                used. If None no analysis instance will be set.
        """
        tomo_exp = ProcessTomography(
            circuit,
            backend=backend,
            physical_qubits=physical_qubits,
            measurement_basis=basis.PauliMeasurementBasis(),
            measurement_indices=measurement_indices,
            preparation_basis=basis.PauliPreparationBasis(),
            preparation_indices=preparation_indices,
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
