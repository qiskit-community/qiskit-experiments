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
Standard RB Experiment class.
"""
from typing import Union, Iterable, Optional, List, Dict, Tuple, Callable
import itertools as itert
import numpy as np
from numpy.random import Generator, default_rng
import copy
from qiskit import QuantumCircuit, QiskitError
from qiskit.providers import Backend
from qiskit.circuit import Gate

from qiskit_experiments.framework import BaseExperiment, ParallelExperiment, Options
from qiskit_experiments.curve_analysis.data_processing import probability
from gst_analysis import GSTAnalysis
from gatesetbasis import GateSetBasis, default_gateset_basis, gatesetbasis_constrction_from_basis_gates


class GateSetTomography(BaseExperiment):
    """Gate set tomography experiment.

    # section: overview

        Gate-Set Tomography is a powerful method that, similarly
        to Quantum Process Tomography, is used to perform full
        characterization of quantum processes using measurement data of a specific set of quantum
        circuits.
        Unlike the quantum process tomography, the gate set tomography deals with
        the state preparation and measurements (SPAM) errors self-consistently by
        including the gates used for both initializing and measuring the qubits in
        the gate set which is reconstructed from processing measurement data.
        Therefore, it does not need any pre-calibration of the initialized qubits
        or the measurements and provides full characterization of the states,
        quantum processes and measurements.

        Gate set tomography is performed on a gate set (G0, G1,...,Gm)
        with the additional information of SPAM circuits (F0,F1,...,Fn)
        that are constructed from the gates in the gate set.
        In gate set tomography, we assume a single initial state rho which is
        the native state in which all states are initialized in 0 state
        and a single POVM measurement operator E, a measurement of all qubits
        in state 0. The SPAM circuits provide us with a complete set of initial
        state F_j|rho> and measurements <E|F_i.
        We perform three types of experiments:
        1) :math:`\langle E  | F_i G_k F_j |\rho \rangle` for 1 <= i,j <= n
            and 1 <= k <= m:
            This experiment enables us to obtain data on the gate G_k
        2) :math:`\langle E  | F_i F_j |\rho \rangle`  for 1 <= i,j <= n:
            This experiment enables us to obtain the Gram matrix required
            to "invert" the results of experiments of type 1 in order to
            reconstruct (a matrix similar to) the gate G_k
        3) :math:`\langle E  | F_j |\rho \rangle` for 1 <= j <= n:
            This experiment enables us to reconstruct <E| and rho

        This experiment protocol is based on the protocol described in section 3.5.2 in
        arXiv:1509.02921.

    # section: reference
        .. ref_arxiv:: 1 1509.02921
    """

    # Analysis class for experiment
    __analysis_class__ = GSTAnalysis

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default analysis options.
        """
        options = super()._default_analysis_options()

        return options

    def __init__(
            self,
            qubits: List[int],
            gateset: Union[str, GateSetBasis, Tuple[str, Dict[str, Union[Callable, Gate]]]] = 'default',
            additional_gates: List[Union[Callable, Gate]] = None,
            only_basis_gates: bool = False,
    ):
        """Initialize a gate set tomography experiment.

        Args:
            qubits: List of qubits labels GST is performed on.
            gateset: The gateset and SPAM data provided as a GatesetBasis or only a set of basis
                gates from which SPAM gates should be constructed, provided as
                ('Only basis gates', a dictionary of the basis gates). If 'default', the default
                gateset corresponding to the number of qubits = qubits will be used.
            additional_gates: The list of gates to add to a gateset to be characterized using GST.
            only_basis_gates: A boolean variable indicating whether the gateset provided includes
            only a set of gates from which the algorithm is required to construct the SPAM gates, or
            if it is full, meaning that it includes both the gate set and the SPAM gates.

        """
        # Initialize base experiment
        super().__init__(qubits)

        if only_basis_gates is False or None:
            if isinstance(gateset, GateSetBasis):
                self._gateset_basis = gateset
            elif any((gateset == 'default', gateset is None)):
                if self.num_qubits < 3:
                    self._gateset_basis = default_gateset_basis(self.num_qubits)
                else:
                    raise QiskitError("Only 1-qubit and 2-qubit default gate sets are available"
                                      " in Qiskit")

        elif only_basis_gates is True and isinstance(gateset, Dict):
            self._gateset_basis = gatesetbasis_constrction_from_basis_gates(gateset, self.num_qubits)[0]
        else:
            raise QiskitError("Can not run GST experiment with the provided gateset format.")

        if additional_gates is not None:
            for gate in additional_gates:
                self._gateset_basis.add_gate(gate)

    def _metadata(self):
        metadata = super()._metadata()
        # A dding measured_qubits and gateset basis to metadata
        if self.physical_qubits:
            metadata["measured_qubits"] = copy.copy(self.physical_qubits)
        if self._gateset_basis:
            metadata["gateset_basis"] = copy.copy(self._gateset_basis)
        return metadata

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Return a list of GST circuits.

        Args:
            backend (Backend): Optional, a backend object.

        Returns:
            A list of :class:`QuantumCircuit`.

        Additional Information:
            It returns the circuits corresponding to measuring :math:`\langle E  | F_i G_k F_j |\rho \rangle`,
             :math:`\langle E  | F_i F_j |\rho \rangle` and :math:`\langle E  | F_j |\rho \rangle`
             as explained in further details in the overview of GateSetTomography class.

        """
        all_circuits = []

        # Experiments of the form <E|F_i G_k F_j|rho>
        FGF_circuits = []
        for gate in self._gateset_basis.gate_labels:
            for (fprep, fmeas) in itert.product(self._gateset_basis.spam_labels, self._gateset_basis.spam_labels):
                circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
                self._gateset_basis.add_spam_to_circuit(circuit, self.physical_qubits, fprep)
                circuit.barrier()
                self._gateset_basis.add_gate_to_circuit(circuit, self.physical_qubits, gate)
                circuit.barrier()
                self._gateset_basis.add_spam_to_circuit(circuit, self.physical_qubits, fmeas)
                circuit.measure(self.physical_qubits, self.physical_qubits)
                circuit.name = str((fprep, gate, fmeas))
                metadata = {
                    "experiment_type": self._type,
                    "circuit_name": circuit.name,
                }
                circuit.metadata = metadata
                FGF_circuits.append(circuit)

        all_circuits = all_circuits + FGF_circuits

        # Experiments of the form <E|F_i F_j|rho>
        # Can be skipped if one of the gates is ideal identity

        FF_circuits = []
        for (fprep, fmeas) in itert.product(self._gateset_basis.spam_labels, self._gateset_basis.spam_labels):
            circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
            self._gateset_basis.add_spam_to_circuit(circuit, self.physical_qubits, fprep)
            circuit.barrier()
            self._gateset_basis.add_spam_to_circuit(circuit, self.physical_qubits, fmeas)
            circuit.measure(self.physical_qubits, self.physical_qubits)
            circuit.name = str((fprep, fmeas))
            metadata = {
                "experiment_type": self._type,
                "circuit_name": circuit.name,
            }
            circuit.metadata = metadata
            FF_circuits.append(circuit)
        all_circuits = all_circuits + FF_circuits

        # Experiments of the form <E|F_j|rho>

        F_circuits = []
        for fprep in self._gateset_basis.spam_labels:
            circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
            self._gateset_basis.add_spam_to_circuit(circuit, self.physical_qubits, fprep)
            circuit.barrier()
            circuit.measure(self.physical_qubits, self.physical_qubits)
            circuit.name = str((fprep,))
            metadata = {
                "experiment_type": self._type,
                "circuit_name": circuit.name,
            }
            circuit.metadata = metadata
            F_circuits.append(circuit)
        all_circuits = all_circuits + F_circuits
        # print(all_circuits)

        return all_circuits
