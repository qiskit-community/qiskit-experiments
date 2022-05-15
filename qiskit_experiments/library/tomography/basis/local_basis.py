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
Circuit basis for tomography preparation and measurement circuits
"""
import functools
from typing import Sequence, Optional, Tuple, Union, List, Dict
import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.quantum_info import DensityMatrix, Statevector, Operator, SuperOp
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.exceptions import QiskitError
from .base_basis import PreparationBasis, MeasurementBasis

# Typing object for POVM args of measurement basis
POVM = Union[List[Statevector], List[DensityMatrix], QuantumChannel]


class LocalPreparationBasis(PreparationBasis):
    """Local tensor-product preparation basis.

    This basis consists of a set of 1-qubit instructions which
    are used to define a tensor-product basis on N-qubits.
    """

    def __init__(
        self,
        name: str,
        instructions: Optional[Sequence[Instruction]] = None,
        default_states: Optional[Sequence[Union[Statevector, DensityMatrix]]] = None,
        qubit_states: Optional[Dict[int, Sequence[Union[Statevector, DensityMatrix]]]] = None,
    ):
        """Initialize a fitter preparation basis.

        Args:
            name: a name to identity the basis.
            instructions: list of 1-qubit instructions for preparing states
                          from the :math:`|0^{\\otimes n}\\rangle` state.
            default_states: Optional, default density matrices prepared by the
                            input instructions. If None these will be determined by
                            ideal simulation of the preparation instructions.
            qubit_states: Optional, a dict with physical qubit keys and a list of
                          density matrices prepared by the list of basis instructions
                          for a specific qubit. The default states will be used for any
                          qubits not specified in this dict.

        Raises:
            QiskitError: if input states or instructions are not valid, or no
                         instructions or states are provided.
        """
        if instructions is None and default_states is None and qubit_states is None:
            raise QiskitError(
                "LocalPreparationBasis must define at least one of instructions, "
                "default_states, or qubit_states."
            )
        super().__init__(name)

        # Internal variables
        self._instructions = tuple()
        self._size = None
        self._default_states = None
        self._default_dim = None
        self._qubit_states = {}
        self._qubit_dim = {}
        self._qubits = set()
        self._custom_defaults = True

        # Format instructions so compatible types can be converted to
        # Instruction instances.
        if instructions is not None:
            self._instructions = _format_instructions(instructions)
            self._size = len(instructions)
            if default_states is None:
                default_states = self._instructions
                self._custom_defaults = False

        # Construct default states
        if default_states is not None:
            self._default_states = tuple(DensityMatrix(i).data for i in default_states)
            self._default_dim = self._default_states[0].shape[0]
            if self._size is None:
                self._size = len(self._default_states)
            elif len(self._default_states) != self._size:
                raise QiskitError(
                    "Number of instructions and number of default states must be equal."
                )

        # Construct states of specific qubits if provided
        qubit_states = qubit_states or {}
        for qubit, states in qubit_states.items():
            if self._size is None:
                self._size = len(states)
            elif len(states) != self._size:
                raise QiskitError("Number of instructions and number of states must be equal.")

            qstates = tuple(DensityMatrix(i).data for i in states)
            self._qubit_states[qubit] = qstates
            self._qubit_dim[qubit] = qstates[0].shape[0]
            self._qubits.add(qubit)

        # Pseudo hash value to make basis hashable for LRU cached functions
        self._hash = hash(
            (
                type(self),
                self._name,
                self._size,
                self._default_dim,
                self._custom_defaults,
                tuple(self._qubits),
                tuple(self._qubit_dim.values()),
                (type(i) for i in self._instructions),
            )
        )

    def __repr__(self):
        return f"<{type(self).__name__}: {self.name}>"

    def __hash__(self):
        return self._hash

    def __eq__(self, value):
        return (
            super().__eq__(value)
            and self._size == getattr(value, "_size", None)
            and self._default_dim == getattr(value, "_default_dim", None)
            and self._custom_defaults == getattr(value, "_custom_defaults", None)
            and self._qubits == getattr(value, "_qubits", None)
            and self._qubit_dim == getattr(value, "_qubit_dim", None)
            and self._instructions == getattr(value, "_instructions", None)
        )

    def index_shape(self, qubits: Sequence[int]) -> Tuple[int, ...]:
        return len(qubits) * (self._size,)

    def matrix_shape(self, qubits: Sequence[int]) -> Tuple[int, ...]:
        return tuple(self._qubit_dim.get(i, self._default_dim) for i in qubits)

    def circuit(
        self, index: Sequence[int], qubits: Optional[Sequence[int]] = None
    ) -> QuantumCircuit:
        # pylint: disable = unused-argument
        if not self._instructions:
            raise NotImplementedError(
                f"Basis {self.name} does not define circuits so can only be "
                " used as a fitter basis for analysis."
            )
        return _tensor_product_circuit(self._instructions, index, self._name)

    def matrix(self, index: Sequence[int], qubits: Optional[Sequence[int]] = None):
        if qubits is None:
            qubits = tuple(range(len(index)))
        try:
            mat = np.eye(1)
            for i, qubit in zip(index, qubits):
                states = self._qubit_states.get(qubit, self._default_states)
                mat = np.kron(states[i], mat)
            return mat
        except TypeError as ex:
            # This occurs if basis is constructed with qubit_states
            # kwarg but no default_states or instructions and is called for
            # a qubit not in the specified kwargs.
            raise ValueError(f"Invalid qubits for basis {self.name}") from ex

    def __json_encode__(self):
        value = {
            "name": self._name,
            "instructions": list(self._instructions) if self._instructions else None,
        }
        if self._custom_defaults:
            value["default_states"] = list(self._default_states)
        if self._qubit_states:
            value["qubit_states"] = self._qubit_states
        return value


class LocalMeasurementBasis(MeasurementBasis):
    """Local tensor-product measurement basis.

    This basis consists of a set of 1-qubit instructions which
    are used to define a tensor-product basis on N-qubits to
    rotate a desired multi-qubit measurement basis to the Z-basis
    measurement.
    """

    def __init__(
        self,
        name: str,
        instructions: Optional[Sequence[Instruction]] = None,
        default_povms: Optional[Sequence[POVM]] = None,
        qubit_povms: Optional[Dict[int, Sequence[POVM]]] = None,
    ):
        """Initialize a fitter preparation basis.

        Args:
            name: a name to identity the basis.
            instructions: list of instructions for rotating a desired
                          measurement basis to the standard :math:`Z^{\\otimes n}`
                          computational basis measurement.
            default_povms: Optional, list if positive operators valued measures (POVM)
                           for of the measurement basis instructions. A POVM can be
                           input as a list of effects (Statevector or DensityMatrix)
                           for each possible measurement outcome of that basis, or as
                           a single QuantumChannel. For the channel case the effects
                           will be calculated by evolving the computation basis states
                           by the adjoint of the channel. If None the input instructions
                           will be used as the POVM channel.
            qubit_povms: Optional, a dict with physical qubit keys and a list of POVMs
                         corresponding to each basis measurement instruction for the
                         specific qubit. The default POVMs will be used for any qubits
                         not specified in this dict.

        Raises:
            QiskitError: if the input instructions or POVMs are not valid, or if no
                         instructions or POVMs are provided.
        """
        if instructions is None and default_povms is None and qubit_povms is None:
            raise QiskitError(
                "LocalMeasurementBasis must define at least one of instructions, "
                "default_povms, or qubit_povms."
            )
        super().__init__(name)

        # Internal variables
        self._instructions = tuple()
        self._size = None
        self._default_povms = None
        self._default_num_outcomes = None
        self._default_dim = None
        self._qubit_povms = {}
        self._qubit_num_outcomes = {}
        self._qubit_dim = {}
        self._qubits = set()
        self._custom_defaults = True

        # Format instructions so compatible types can be converted to
        # Instruction instances.
        if instructions is not None:
            self._instructions = _format_instructions(instructions)
            self._size = len(self._instructions)
            if default_povms is None:
                default_povms = instructions
                self._custom_defaults = False

        # Format default POVMs
        if default_povms is not None:
            self._default_povms = _format_povms(default_povms)
            self._default_num_outcomes = len(self._default_povms[0])
            self._default_dim = self._default_povms[0][0].shape[0]
            if self._size is None:
                self._size = len(self._default_povms)
            elif len(self._default_povms) != self._size:
                raise QiskitError("Number of instructions and number of states must be equal.")
            if any(len(povm) != self._default_num_outcomes for povm in self._default_povms):
                raise QiskitError(
                    "LocalMeasurementBasis default POVM elements must all have "
                    "the same number of outcomes."
                )

        # Format qubit POVMS
        qubit_povms = qubit_povms or {}
        for qubit, povms in qubit_povms.items():
            f_povms = _format_povms(povms)
            num_outcomes = len(f_povms[0])
            if any(len(povm) != num_outcomes for povm in f_povms):
                raise QiskitError(
                    "LocalMeasurementBasis POVM elements must all have the "
                    "same number of outcomes."
                )
            self._qubit_povms[qubit] = f_povms
            self._qubit_num_outcomes[qubit] = num_outcomes
            self._qubit_dim[qubit] = f_povms[0][0].shape[0]
            self._qubits.add(qubit)

        # Pseudo hash value to make basis hashable for LRU cached functions
        self._hash = hash(
            (
                type(self),
                self._name,
                self._size,
                self._default_dim,
                self._default_num_outcomes,
                self._custom_defaults,
                tuple(self._qubits),
                tuple(self._qubit_dim.values()),
                tuple(self._qubit_num_outcomes.values()),
                (type(i) for i in self._instructions),
            )
        )

    def __repr__(self):
        return f"<{type(self).__name__}: {self.name}>"

    def __hash__(self):
        return self._hash

    def __eq__(self, value):
        return (
            super().__eq__(value)
            and self._size == getattr(value, "_size", None)
            and self._default_dim == getattr(value, "_default_dim", None)
            and self._default_num_outcomes == getattr(value, "_default_num_outcomes", None)
            and self._custom_defaults == getattr(value, "_custom_defaults", None)
            and self._qubit_dim == getattr(value, "_qubit_dim", None)
            and self._qubit_num_outcomes == getattr(value, "_qubit_num_outcomes", None)
            and self._qubits == getattr(value, "_qubits", None)
            and self._instructions == getattr(value, "_instructions", None)
        )

    def index_shape(self, qubits: Sequence[int]) -> Tuple[int, ...]:
        return len(qubits) * (self._size,)

    def matrix_shape(self, qubits: Sequence[int]) -> Tuple[int, ...]:
        return tuple(self._qubit_dim.get(i, self._default_dim) for i in qubits)

    def outcome_shape(self, qubits: Sequence[int]) -> Tuple[int, ...]:
        return tuple(self._qubit_num_outcomes.get(i, self._default_num_outcomes) for i in qubits)

    def circuit(self, index: Sequence[int], qubits: Optional[Sequence[int]] = None):
        # pylint: disable = unused-argument
        if not self._instructions:
            raise NotImplementedError(
                f"Basis {self.name} does not define circuits so can only be "
                " used as a fitter basis for analysis."
            )
        circuit = _tensor_product_circuit(self._instructions, index, self._name)
        circuit.measure_all()
        return circuit

    def matrix(self, index: Sequence[int], outcome: int, qubits: Optional[Sequence[int]] = None):
        if qubits is None:
            qubits = tuple(range(len(index)))
        try:
            outcome_index = self._outcome_indices(outcome, tuple(qubits))
            mat = np.eye(1)
            for idx, odx, qubit in zip(index, outcome_index, qubits):
                povms = self._qubit_povms.get(qubit, self._default_povms)
                mat = np.kron(povms[idx][odx], mat)
            return mat
        except TypeError as ex:
            # This occurs if basis is constructed with qubit_states
            # kwarg but no default_states or instructions and is called for
            # a qubit not in the specified kwargs.
            raise ValueError(f"Invalid qubits for basis {self.name}") from ex

    @functools.lru_cache(None)
    def _outcome_indices(self, outcome: int, qubits: Tuple[int, ...]) -> Tuple[int, ...]:
        """Convert an outcome integer to a tuple of single-qubit outcomes"""
        num_outcomes = self._qubit_num_outcomes.get(qubits[0], self._default_num_outcomes)
        try:
            value = (outcome % num_outcomes,)
            if len(qubits) == 1:
                return value
            return value + self._outcome_indices(outcome // num_outcomes, qubits[1:])
        except TypeError as ex:
            raise ValueError("Invalid qubits for basis") from ex

    def __json_encode__(self):
        value = {
            "name": self._name,
            "instructions": list(self._instructions) if self._instructions else None,
        }
        if self._custom_defaults:
            value["default_povms"] = list(self._default_povms)
        if self._qubit_povms:
            value["qubit_povms"] = self._qubit_povms
        return value


def _tensor_product_circuit(
    instructions: Sequence[Instruction],
    index: Sequence[int],
    name: str = "",
) -> QuantumCircuit:
    """Return tensor product of 1-qubit basis instructions"""
    size = len(instructions)
    circuit = QuantumCircuit(len(index), name=f"{name}{list(index)}")
    for i, elt in enumerate(index):
        if elt >= size:
            raise QiskitError("Invalid basis element index")
        circuit.append(instructions[elt], [i])
    return circuit


def _format_instructions(instructions: Sequence[any]) -> Tuple[Instruction, ...]:
    """Parse multiple input formats for list of instructions"""
    ret = tuple()
    for inst in instructions:
        # Convert to instructions if object is not an instruction
        # This allows converting raw unitary matrices and other operator
        # types like Pauli or Clifford into instructions.
        if not isinstance(inst, Instruction):
            if hasattr(inst, "to_instruction"):
                inst = inst.to_instruction()
            else:
                inst = Operator(inst).to_instruction()

        # Validate that instructions are single qubit
        if inst.num_qubits != 1:
            raise QiskitError(f"Input instruction {inst.name} is not a 1-qubit instruction.")
        ret += (inst,)
    return ret


def _format_povms(povms: Sequence[any]) -> Tuple[Tuple[np.ndarray, ...], ...]:
    """Format sequence of basis POVMs"""
    formatted_povms = []
    # Convert from operator/channel to POVM effects
    for povm in povms:
        if isinstance(povm, (list, tuple)):
            # POVM is already an effect
            formatted_povms.append(povm)
            continue

        # Convert POVM to operator of quantum channel
        try:
            chan = Operator(povm)
        except QiskitError:
            chan = SuperOp(povm)
        adjoint = chan.adjoint()
        dims = adjoint.input_dims()
        dim = np.prod(dims)
        effects = tuple(DensityMatrix.from_int(i, dims).evolve(adjoint) for i in range(dim))
        formatted_povms.append(effects)

    # Format POVM effects to density matrix matrices
    return tuple(tuple(DensityMatrix(effect).data for effect in povm) for povm in formatted_povms)
