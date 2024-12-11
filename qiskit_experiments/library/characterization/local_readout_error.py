# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Local readout error calibration experiment class.
"""
from typing import Iterable, List, Optional
from qiskit import QuantumCircuit
from qiskit.providers.backend import BackendV2, Backend
from qiskit.exceptions import QiskitError
from qiskit_experiments.framework import BaseExperiment
from qiskit_experiments.library.characterization.analysis.local_readout_error_analysis import (
    LocalReadoutErrorAnalysis,
)
from .correlated_readout_error import calibration_circuit


class LocalReadoutError(BaseExperiment):
    r"""An experiment for characterizing local readout error.

    # section: overview

        This class constructs a :class:`~qiskit.result.LocalReadoutMitigator` containing a sequence
        of assignment matrices :math:`A` characterizing the readout error for the given qubits
        from the experiment results. The full assignment matrix is accessible via the
        :meth:`~qiskit.result.LocalReadoutMitigator.assignment_matrix` method.

        Readout errors affect quantum computation during the measurement
        of the qubits in a quantum device. By characterizing the readout errors,
        it is possible to construct a *readout error mitigator* that is used both
        to obtain a more accurate distribution of the outputs, and more accurate
        measurements of expectation value for measurables.

        The readout mitigator is generated from an *assignment matrix*:
        a :math:`2^n \times 2^n` matrix :math:`A` such that :math:`A_{y,x}` is the probability
        to observe :math:`y` given the true outcome should be :math:`x`. The assignment matrix is used
        to compute the *mitigation matrix* used in the readout error mitigation process itself.

        A *Local readout mitigator* works under the assumption that readout errors are mostly
        *local*, meaning readout errors for different qubits are independent of each other.
        In this case, the assignment matrix is the tensor product of :math:`n` :math:`2 \times 2`
        matrices, one for each qubit, making it practical to store the assignment matrix in implicit
        form, by storing the individual :math:`2 \times 2` assignment matrices.
        The corresponding class in Qiskit is the :class:`~qiskit.result.LocalReadoutMitigator`
        in :mod:`qiskit.result`.

        The experiment generates 2 circuits, corresponding to the states
        :math:`|0^n\rangle` and :math:`|1^n\rangle`, measuring the error in all
        the qubits at once, and constructs the assignment matrix and local mitigator from the results.

        See :class:`LocalReadoutErrorAnalysis`
        documentation for additional information on local readout error experiment analysis.

    # section: analysis_ref
        :class:`LocalReadoutErrorAnalysis`

    # section: example
        .. jupyter-execute::
            :hide-code:

            # backend
            from qiskit_aer import AerSimulator
            from qiskit_ibm_runtime.fake_provider import FakePerth
            backend = AerSimulator.from_backend(FakePerth())

        .. jupyter-execute::

            from qiskit_experiments.library import LocalReadoutError

            qubits = list(range(4))
            exp = LocalReadoutError(physical_qubits=qubits, backend=backend)
            exp.analysis.set_options(plot=True)
            exp.set_run_options(shots=10000)

            exp_data = exp.run().block_for_results()
            display(exp_data.figure(0))
            exp_data.analysis_results(dataframe=True)

    # section: manual
        :doc:`/manuals/measurement/readout_mitigation`

    # section: reference
        .. ref_arxiv:: 1 2006.14044
    """

    def __init__(
        self,
        physical_qubits: Optional[Iterable[int]] = None,
        backend: Optional[Backend] = None,
    ):
        """Initialize a local readout error characterization experiment.

        Args:
            physical_qubits: Optional, the backend qubits being characterized
                for readout error. If None all qubits on the provided backend
                will be characterized.
            backend: Optional, the backend to characterize.

        Raises:
            QiskitError: If args are not valid.
        """
        if physical_qubits is None:
            if backend is None:
                raise QiskitError("`physical_qubits` and `backend` kwargs cannot both be None.")
            num_qubits = 0
            if isinstance(backend, BackendV2):
                num_qubits = backend.target.num_qubits
            elif isinstance(backend, Backend):
                num_qubits = backend.configuration().num_qubits
            if num_qubits:
                physical_qubits = range(num_qubits)
            else:
                raise QiskitError(f"Cannot infer backend qubits from backend {backend}")
        super().__init__(physical_qubits, backend=backend)
        self.analysis = LocalReadoutErrorAnalysis()

    def circuits(self) -> List[QuantumCircuit]:
        """Returns the experiment's circuits"""
        labels = ["0" * self.num_qubits, "1" * self.num_qubits]
        return [calibration_circuit(self.num_qubits, label) for label in labels]
