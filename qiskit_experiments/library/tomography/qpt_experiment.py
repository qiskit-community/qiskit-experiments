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
Quantum Process Tomography experiment
"""

from typing import Union, Optional, List, Tuple, Sequence
import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction, Clbit
from qiskit.providers.backend import Backend
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import Choi, Operator, Statevector, DensityMatrix, partial_trace

from qiskit_experiments.exceptions import QiskitError
from .tomography_experiment import TomographyExperiment, TomographyAnalysis, BaseAnalysis
from .qpt_analysis import ProcessTomographyAnalysis
from . import basis


class ProcessTomography(TomographyExperiment):
    """An experiment to reconstruct the quantum channel from measurement data.

    # section: overview
        Quantum process tomography (QPT) is a method for experimentally
        reconstructing the quantum channel from measurement data.

        A QPT experiment prepares multiple input states, evolves them by the
        circuit, then performs multiple measurements in different measurement
        bases. The resulting measurement data is then post-processed by a
        tomography fitter to reconstruct the quantum channel.

    # section: note
        Performing full process tomography on an `N`-qubit circuit requires
        running :math:`4^N 3^N` measurement circuits when using the default
        preparation and measurement bases.

    # section: analysis_ref
        :class:`ProcessTomographyAnalysis`

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
            from qiskit_experiments.library import ProcessTomography

            num_qubits = 2
            qc_ghz = QuantumCircuit(num_qubits)
            qc_ghz.h(0)
            qc_ghz.s(0)

            for i in range(1, num_qubits):
                qc_ghz.cx(0, i)

            qptexp = ProcessTomography(qc_ghz)
            qptdata = qptexp.run(backend=backend,
                                 shots=1000,
                                 seed_simulator=100,).block_for_results()
            choi_out = qptdata.analysis_results("state").value

            # extracting a densitymatrix from choi_out
            from qiskit.visualization import plot_state_city
            import qiskit.quantum_info as qinfo

            _rho_exp_00 = np.array([[None, None, None, None],
                                    [None, None, None, None],
                                    [None, None, None, None],
                                    [None, None, None, None]])

            for i in range(4):
                for j in range(4):
                    _rho_exp_00[i][j] = choi_out.data[i][j]

            rho_exp_00 = qinfo.DensityMatrix(_rho_exp_00)
            display(plot_state_city(rho_exp_00, title="Density Matrix"))
    """

    def __init__(
        self,
        circuit: Union[QuantumCircuit, Instruction, BaseOperator],
        backend: Optional[Backend] = None,
        physical_qubits: Optional[Sequence[int]] = None,
        measurement_basis: basis.MeasurementBasis = basis.PauliMeasurementBasis(),
        measurement_indices: Optional[Sequence[int]] = None,
        preparation_basis: basis.PreparationBasis = basis.PauliPreparationBasis(),
        preparation_indices: Optional[Sequence[int]] = None,
        basis_indices: Optional[Sequence[Tuple[List[int], List[int]]]] = None,
        conditional_circuit_clbits: Union[bool, Sequence[int], Sequence[Clbit]] = False,
        analysis: Union[BaseAnalysis, None, str] = "default",
        target: Union[Statevector, DensityMatrix, None, str] = "default",
    ):
        """Initialize a quantum process tomography experiment.

        Args:
            circuit: the quantum process circuit. If not a quantum circuit
                it must be a class that can be appended to a quantum circuit.
            backend: The backend to run the experiment on.
            physical_qubits: Optional, the physical qubits for the initial state circuit.
                If None this will be qubits [0, N) for an N-qubit circuit.
            measurement_basis: Tomography basis for measurements. If not specified the
                default basis is the :class:`~basis.PauliMeasurementBasis`.
            measurement_indices: Optional, the `physical_qubits` indices to be measured.
                If None all circuit physical qubits will be measured.
            preparation_basis: Tomography basis for measurements. If not specified the
                default basis is the :class:`~basis.PauliPreparationBasis`.
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
                be conditioned on when reconstructing the channel. If True all circuit
                clbits will be conditioned on. Enabling this will return a list of
                reconstructed channel components conditional on the values of these clbit
                values.
            analysis: Optional, a custom analysis instance to use. If ``"default"``
                :class:`~.ProcessTomographyAnalysis` will be used. If None no analysis
                instance will be set.
            target: Optional, a custom quantum state target for computing the
                state fidelity of the fitted density matrix during analysis.
                If "default" the state will be inferred from the input circuit
                if it contains no classical instructions.
        """
        if analysis == "default":
            analysis = ProcessTomographyAnalysis()

        super().__init__(
            circuit,
            backend=backend,
            physical_qubits=physical_qubits,
            measurement_basis=measurement_basis,
            measurement_indices=measurement_indices,
            preparation_basis=preparation_basis,
            preparation_indices=preparation_indices,
            basis_indices=basis_indices,
            conditional_circuit_clbits=conditional_circuit_clbits,
            analysis=analysis,
        )

        # Set target quantum channel
        if isinstance(self.analysis, TomographyAnalysis):
            if target == "default":
                target = self._target_quantum_channel()
            self.analysis.set_options(target=target)

    def _target_quantum_channel(self) -> Union[Choi, Operator]:
        """Return the process tomography target"""
        # Check if circuit contains measure instructions
        # If so we cannot return target state
        circuit_ops = self._circuit.count_ops()
        if "measure" in circuit_ops:
            return None

        try:
            circuit = self._permute_circuit()
            if "reset" in circuit_ops or "kraus" in circuit_ops or "superop" in circuit_ops:
                channel = Choi(circuit)
            else:
                channel = Operator(circuit)
        except QiskitError:
            # Circuit couldn't be simulated
            return None

        total_qubits = self._circuit.num_qubits
        num_meas = total_qubits if not self._meas_indices else len(self._meas_indices)
        num_prep = total_qubits if not self._prep_indices else len(self._prep_indices)

        # If all qubits are prepared or measurement we are done
        if num_meas == total_qubits and num_prep == total_qubits:
            return channel

        # Convert channel to a state to project and trace out non-tomography
        # input and output qubits
        if isinstance(channel, Operator):
            chan_state = Statevector(np.ravel(channel, order="F"))
        else:
            chan_state = DensityMatrix(channel.data)

        # Get qargs for non measured and prepared subsystems
        non_meas_qargs = list(range(num_meas, total_qubits))
        non_prep_qargs = list(range(total_qubits + num_prep, 2 * total_qubits))

        # Project non-prepared subsystems on to the zero state
        if non_prep_qargs:
            proj0 = Operator([[1, 0], [0, 0]])
            for qarg in non_prep_qargs:
                chan_state = chan_state.evolve(proj0, [qarg])

        # Trace out indices to remove
        tr_qargs = non_meas_qargs + non_prep_qargs
        chan_state = partial_trace(chan_state, tr_qargs)
        channel = Choi(chan_state.data, input_dims=[2] * num_prep, output_dims=[2] * num_meas)
        return channel
