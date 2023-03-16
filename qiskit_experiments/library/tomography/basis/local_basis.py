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
from typing import Sequence, Optional, Tuple, Union, List, Dict
import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info import DensityMatrix, Statevector, Operator, SuperOp
from qiskit.exceptions import QiskitError

from .base_basis import PreparationBasis, MeasurementBasis
from .cache_method import cache_method, _method_cache_name


# Typing
Povm = Union[List[Statevector], List[DensityMatrix], QuantumChannel]
States = Union[List[QuantumState], Dict[Tuple[int, ...], QuantumState]]


class LocalPreparationBasis(PreparationBasis):
    """Local tensor-product preparation basis.

    This basis consists of a set of 1-qubit instructions which
    are used to define a tensor-product basis on N-qubits.
    """

    def __init__(
        self,
        name: str,
        instructions: Optional[Sequence[Instruction]] = None,
        default_states: Optional[States] = None,
        qubit_states: Optional[Dict[Tuple[int, ...], States]] = None,
    ):
        """Initialize a fitter preparation basis.

        Args:
            name: a name to identify the basis.
            instructions: list of 1-qubit instructions for preparing states
                          from the :math:`|0\\rangle` state.
            default_states: Optional, default density matrices prepared by the
                            input instructions. If None these will be determined by
                            ideal simulation of the preparation instructions.
            qubit_states: Optional, a dict with physical qubit keys and a list of
                          density matrices prepared by the list of basis instructions
                          for a specific qubit. The default states will be used for any
                          qubits not specified in this dict.

        Raises:
            QiskitError: If input states or instructions are not valid, or no
                         instructions or states are provided.
        """
        if instructions is None and default_states is None and qubit_states is None:
            raise QiskitError(
                "LocalPreparationBasis must define at least one of instructions, "
                "default_states, or qubit_states."
            )
        super().__init__(name)
        # POVM element variables
        self._instructions = _format_instructions(instructions)
        self._default_states = _format_states(default_states, (0,), self._instructions)
        self._qubit_states = _format_qubit_states(qubit_states)
        self._custom_defaults = bool(default_states)

        # Other attributes derived from povms and instructions
        # that need initializing
        self._qubits = set()
        self._size = None
        self._default_dim = None
        self._qubit_dim = {}
        self._hash = None

        # Initialize attributes
        self._initialize()

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
        qubits = tuple(qubits)
        if len(qubits) > 1 and qubits in self._qubit_dim:
            return self._qubit_dim[qubits]
        dims = tuple()
        for i in qubits:
            dims += self._qubit_dim.get((i,), (self._default_dim,))
        return dims

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
        # Convert args to hashable tuples
        if qubits is None:
            qubits = tuple(range(len(index)))
        else:
            qubits = tuple(qubits)
        index = tuple(index)

        try:
            # Look for custom POVM for specified qubits
            state = self._generate_qubits_state(index, qubits)
            if state is not None:
                return state.data

            mat = np.eye(1)
            for idx, qubit in zip(index, qubits):
                qubit_state = self._generate_qubits_state((idx,), (qubit,))
                mat = np.kron(qubit_state, mat)
            return mat

        except TypeError as ex:
            # This occurs if basis is constructed with qubit_states
            # kwarg but no default_states or instructions and is called for
            # a qubit not in the specified kwargs.
            raise ValueError(f"Invalid qubits for basis {self.name}") from ex

    @cache_method()
    def _generate_qubits_state(self, index: Tuple[int, ...], qubits: Tuple[int, ...]):
        """LRU cached function for returning POVMS"""
        num_qubits = len(qubits)

        # Check for N-qubit states
        if qubits in self._qubit_states:
            # Get states for specified qubits
            # TODO: In the future we could add support for different orderings
            #       of qubits by permuting the returned POVMS
            states = self._qubit_states[qubits]
            if index in states:
                return states[index]

            # Look up custom 0 init state for specified qubits
            # TODO: Add support for noisy instuctions
            if not self._instructions:
                raise NotImplementedError(
                    f"Basis {self.name} does not define circuits to construct POVMs from"
                )
            key0 = num_qubits * (0,)
            if key0 in states:
                circuit = _tensor_product_circuit(self._instructions, index, self._name)
                return _generate_state(circuit, states[key0])

        # No match, so if 1-qubit use default, otherwise return None
        if num_qubits == 1 and self._default_states:
            return self._default_states[index]
        return None

    def _initialize(self):
        """Initialize dimension and num outcomes"""
        if self._instructions:
            self._size = len(self._instructions)

        # Format default POVMs
        if self._default_states:
            default_state = next(iter(self._default_states.values()))
            self._default_dim = np.prod(default_state.dims())
            if self._size is None:
                self._size = len(self._default_states)
            elif len(self._default_states) != self._size:
                raise QiskitError("Number of instructions and number of states must be equal.")

        # Format qubit states
        for qubits, states in self._qubit_states.items():
            state = next(iter(states.values()))
            num_qubits = len(qubits)
            dims = state.dims()
            if num_qubits == 1:
                qubit_dim = (np.prod(dims),)
            elif len(dims) == num_qubits:
                qubit_dim = tuple(dims)
            else:
                # Assume all subsystems have the same dimension if the provided
                # state dimension don't match number of qubits
                ave_dim = np.prod(dims) ** (1 / num_qubits)
                if int(ave_dim) != ave_dim:
                    raise QiskitError("Cannot infer unequal subsystem dimensions from input states")
                qubit_dim = num_qubits * (int(ave_dim),)
            self._qubit_dim[qubits] = qubit_dim
            self._qubits.update(qubits)

        # Pseudo hash value to make basis hashable for LRU cached functions
        self._hash = hash(
            (
                type(self),
                self._name,
                self._size,
                self._default_dim,
                self._custom_defaults,
                tuple(self._qubits),
                tuple(sorted(self._qubit_dim.items())),
                tuple(type(i) for i in self._instructions),
            )
        )

    def __json_encode__(self):
        value = {
            "name": self._name,
            "instructions": list(self._instructions) if self._instructions else None,
        }
        if self._custom_defaults:
            value["default_states"] = self._default_states
        if self._qubit_states:
            value["qubit_states"] = self._qubit_states
        return value

    def __getstate__(self):
        # override get state to skip class cache when pickling
        state = self.__dict__.copy()
        state.pop(_method_cache_name(self), None)
        return state


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
        default_povms: Optional[Sequence[Povm]] = None,
        qubit_povms: Optional[Dict[Tuple[int, ...], Sequence[Povm]]] = None,
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
            QiskitError: If the input instructions or POVMs are not valid, or if no
                         instructions or POVMs are provided.
        """
        if instructions is None and default_povms is None and qubit_povms is None:
            raise QiskitError(
                "LocalMeasurementBasis must define at least one of instructions, "
                "default_povms, or qubit_povms."
            )
        super().__init__(name)

        # POVM element variables
        self._instructions = _format_instructions(instructions)
        self._default_povms = _format_default_povms(default_povms, self._instructions)
        self._qubit_povms = _format_qubit_povms(qubit_povms)
        self._custom_defaults = bool(default_povms)

        # Other attributes derived from povms and instructions
        # that need initializing
        self._qubits = set()
        self._size = None
        self._default_num_outcomes = None
        self._default_dim = None
        self._qubit_num_outcomes = {}
        self._qubit_dim = {}
        self._hash = None

        # Initialize attributes
        self._initialize()

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
        qubits = tuple(qubits)
        if len(qubits) > 1 and qubits in self._qubit_dim:
            return self._qubit_dim[qubits]
        dims = tuple()
        for i in qubits:
            dims += self._qubit_dim.get((i,), (self._default_dim,))
        return dims

    def outcome_shape(self, qubits: Sequence[int]) -> Tuple[int, ...]:
        qubits = tuple(qubits)
        if qubits in self._qubit_num_outcomes:
            return self._qubit_num_outcomes[qubits]
        shape = tuple()
        for i in qubits:
            shape += self._qubit_num_outcomes.get((i,), (self._default_num_outcomes,))
        return shape

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
        # Convert args to hashable tuples
        if qubits is None:
            qubits = tuple(range(len(index)))
        else:
            qubits = tuple(qubits)
        index = tuple(index)

        try:
            # Look for custom POVM for specified qubits
            qubit_povm = self._generate_qubits_povm(index, qubits)
            if qubit_povm:
                return qubit_povm[outcome].data

            # Otherwise construct tensor product POVM
            outcome_index = self._outcome_indices(outcome, qubits)
            mat = np.eye(1)
            for idx, odx, qubit in zip(index, outcome_index, qubits):
                povm = self._generate_qubits_povm((idx,), (qubit,))
                mat = np.kron(povm[odx], mat)
            return mat

        except TypeError as ex:
            # This occurs if basis is constructed with qubit_states
            # kwarg but no default_states or instructions and is called for
            # a qubit not in the specified kwargs.
            raise ValueError(f"Invalid qubits for basis {self.name}") from ex

    def _initialize(self):
        """Initialize dimension and num outcomes"""
        if self._instructions:
            self._size = len(self._instructions)

        # Format default POVMs
        if self._default_povms:
            default_povm = next(iter(self._default_povms.values()))
            self._default_num_outcomes = len(default_povm)
            self._default_dim = np.prod(default_povm[0].dims())
            if self._size is None:
                self._size = len(self._default_povms)
            elif len(self._default_povms) != self._size:
                raise QiskitError("Number of instructions and number of states must be equal.")
            if any(
                len(povm) != self._default_num_outcomes for povm in self._default_povms.values()
            ):
                raise QiskitError(
                    "LocalMeasurementBasis default POVM elements must all have "
                    "the same number of outcomes."
                )

        # Format qubit POVMS
        for qubits, povms in self._qubit_povms.items():
            povm = next(iter(povms.values()))
            num_povms = len(povm)
            if any(len(povm) != num_povms for povm in povms.values()):
                raise QiskitError(
                    "LocalMeasurementBasis POVM elements must all have the "
                    "same number of outcomes."
                )
            num_qubits = len(qubits)
            dims = povm[0].dims()
            dim = np.prod(dims)
            if num_qubits == 1:
                qubit_dim = (dim,)
                num_outcomes = (num_povms,)
            elif len(dims) == num_qubits:
                qubit_dim = tuple(dims)
                if dim != num_povms:
                    raise QiskitError("POVMs dimensions don't match number of outcomes")
                num_outcomes = qubit_dim
            else:
                # Assume all subsystems have the same dimension if the provided
                # operator dimension don't match number of qubits
                ave_dim = np.prod(dims) ** (1 / num_qubits)
                if int(ave_dim) != ave_dim:
                    raise QiskitError("Cannot infer unequal subsystem dimensions from input POVMs")
                qubit_dim = num_qubits * (int(ave_dim),)
                ave_num_outcomes = num_povms ** (1 / num_qubits)
                if int(ave_num_outcomes) != ave_num_outcomes:
                    raise QiskitError("Cannot infer unequal subsystem num_outcome from input POVMs")
                num_outcomes = num_qubits * (int(ave_dim),)
            self._qubit_num_outcomes[qubits] = num_outcomes
            self._qubit_dim[qubits] = qubit_dim
            self._qubits.update(qubits)

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
                tuple(sorted(self._qubit_dim.items())),
                tuple(sorted(self._qubit_num_outcomes.items())),
                tuple(type(i) for i in self._instructions),
            )
        )

    @cache_method()
    def _outcome_indices(self, outcome: int, qubits: Tuple[int, ...]) -> Tuple[int, ...]:
        """Convert an outcome integer to a tuple of single-qubit outcomes"""
        num_outcomes = np.prod(self._qubit_num_outcomes.get(qubits[:1], self._default_num_outcomes))
        try:
            value = (outcome % num_outcomes,)
            if len(qubits) == 1:
                return value
            return value + self._outcome_indices(outcome // num_outcomes, qubits[1:])
        except TypeError as ex:
            raise ValueError("Invalid qubits for basis") from ex

    @cache_method()
    def _generate_qubits_povm(self, index: Tuple[int, ...], qubits: Tuple[int, ...]):
        """LRU cached function for returning POVMS"""
        num_qubits = len(qubits)
        if qubits in self._qubit_povms:
            # Get POVMS for specified qubits
            # TODO: In the future we could add support for different orderings
            #       of qubits by permuting the returned POVMS
            povms = self._qubit_povms[qubits]
            if index in povms:
                return povms[index]

            # Look up custom Z-default POVM for specified qubits
            # TODO: Add support for noisy instuctions
            if not self._instructions:
                raise NotImplementedError(
                    f"Basis {self.name} does not define circuits to construct POVMs from"
                )
            key0 = num_qubits * (0,)
            if key0 in povms:
                circuit = _tensor_product_circuit(self._instructions, index, self._name)
                return _generate_povm(circuit, povms[key0])

        # No match, so if 1-qubit use default, otherwise return None
        if num_qubits == 1 and self._default_povms:
            return self._default_povms[index[0]]
        return None

    def __json_encode__(self):
        value = {
            "name": self._name,
            "instructions": self._instructions if self._instructions else None,
        }
        if self._custom_defaults:
            value["default_povms"] = self._default_povms
        if self._qubit_povms:
            value["qubit_povms"] = self._qubit_povms
        return value

    def __getstate__(self):
        # override get state to skip class cache when pickling
        state = self.__dict__.copy()
        state.pop(_method_cache_name(self), None)
        return state


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


def _format_instructions(instructions: Sequence[any]) -> List[Instruction]:
    """Parse multiple input formats for list of instructions"""
    ret = []
    if instructions is None:
        return ret
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
        ret.append(inst)

    return ret


def _generate_povm(
    value: Union[List[DensityMatrix], Instruction, Operator, SuperOp],
    default_z: Optional[List[DensityMatrix]] = None,
    dims: Optional[Tuple[int, ...]] = None,
) -> List[DensityMatrix]:
    """Format a POVM into list of density matrix effects"""
    # If already a list convert to DensityMatrix objects
    if isinstance(value, (list, tuple)):
        return [DensityMatrix(i, dims=dims) for i in value]

    # Otherwise convert from operator/channel to POVM effects
    try:
        chan = Operator(value)
    except QiskitError:
        chan = SuperOp(value)
    adjoint = chan.adjoint()
    if dims is None:
        dims = adjoint.input_dims()

    if default_z is not None:
        z_states = [DensityMatrix(i, dims) for i in default_z]
    else:
        z_states = [DensityMatrix.from_int(i, dims) for i in range(np.prod(dims))]

    return [state.evolve(adjoint) for state in z_states]


def _format_default_povms(
    default_povms: any, instructions: Optional[Sequence[Instruction]] = None
) -> Dict[Tuple[int, ...], List[DensityMatrix]]:
    "Format default POVM data"
    # Parse data into a dict
    # Legacy data handling
    if isinstance(default_povms, (list, tuple, np.ndarray)):
        povms = dict(enumerate(default_povms))
    elif isinstance(default_povms, dict):
        povms = {int(key): val for key, val in default_povms.items()}
    elif not default_povms:
        povms = {}

    # Add instructions to POVM dict if not specified in data
    if instructions and len(povms) < len(instructions):
        for i, inst in enumerate(instructions):
            if i not in povms:
                povms[i] = inst

    # Look for default POVMs for Z
    if 0 in povms and isinstance(povms[0], (list, tuple)):
        default_z = povms[0]
    else:
        default_z = None

    # Format remaining POVM values and update attribute
    return {key: _generate_povm(val, default_z) for key, val in povms.items()}


def _format_qubit_povms(
    qubit_povms: any,
) -> Dict[Tuple[int, ...], Dict[Tuple[int, ...], List[DensityMatrix]]]:
    """Format qubit POVMs dict"""
    povms = {}
    if not qubit_povms:
        return povms

    # Format POVM keys to be a tuple of (qubits, basis)
    for qubits, povm in qubit_povms.items():
        if isinstance(qubits, int):
            qubits = (qubits,)

        # Convert value to dict if not already
        if not isinstance(povm, dict):
            formatted_povm = {(i,): val for i, val in enumerate(povm)}
        else:
            # Format dict keys
            formatted_povm = {}
            for index, value in povm.items():
                if isinstance(index, int):
                    index = (index,)
                formatted_povm[tuple(index)] = value

        # Add qubit POVM dict to povms
        povms[qubits] = formatted_povm

    # Format POVM values
    for qubits, povm in povms.items():

        # Convert any Z-povm value if present
        key0 = len(qubits) * (0,)
        if key0 in povm and isinstance(povm[key0], (list, tuple)):
            default_z = povm[key0]
        else:
            default_z = None

        # Convert any values from instructions/channels
        # By applying to Z-povm
        for key, value in povm.items():
            povm[key] = _generate_povm(value, default_z)

    return povms


def _generate_state(
    value: Union[Statevector, DensityMatrix, Instruction, Operator, SuperOp],
    init_state: Optional[QuantumState] = None,
    dims: Optional[Tuple[int, ...]] = None,
) -> DensityMatrix:
    """Format a state into list of DensityMatrix"""
    # If already a quantum state convert to a density matrix
    if isinstance(value, (QuantumState, np.ndarray, list)):
        return DensityMatrix(value, dims=dims)

    # Otherwise convert from operator/channel to POVM effects
    try:
        chan = Operator(value)
    except QiskitError:
        chan = SuperOp(value)
    if dims is None:
        dims = chan.input_dims()

    # Default |0> state density matrix
    if init_state is None:
        init_state = DensityMatrix.from_int(0, dims)
    elif not isinstance(init_state, DensityMatrix):
        init_state = DensityMatrix(init_state, dims)
    return init_state.evolve(chan)


def _format_states(
    states: Optional[Union[List[any], Dict[Tuple[int, ...], any]]],
    init_key: Tuple[int, ...] = (0,),
    instructions: Optional[Sequence[Instruction]] = None,
) -> Dict[Tuple[int, ...], DensityMatrix]:
    "Format default state data"
    # Parse data into dict to include legacy handling
    states = _format_data_dict(states, instructions)

    # Look for default state
    init_val = states.get(init_key, None)
    if isinstance(init_val, (QuantumState, np.ndarray, list)):
        init_state = DensityMatrix(init_val)
    else:
        init_state = None

    # Format remaining states and update attribute
    return {key: _generate_state(val, init_state) for key, val in states.items()}


def _format_qubit_states(
    qubit_states: any,
) -> Dict[Tuple[int, ...], Dict[Tuple[int, ...], DensityMatrix]]:
    """Format qubit POVMs dict"""
    if not qubit_states:
        return {}

    # Format POVM keys to be a tuple of (qubits, basis)
    formatted_states = {}
    for qubits, states in qubit_states.items():
        if isinstance(qubits, int):
            qubits = (qubits,)
        else:
            qubits = tuple(qubits)
        init_key = len(qubits) * (0,)
        formatted_states[qubits] = _format_states(states, init_key)

    return formatted_states


def _format_data_dict(
    data: Optional[any], instructions: Optional[Sequence[Union[Instruction, BaseOperator]]] = None
) -> Dict[Tuple[int, ...], any]:
    "Format default state data"
    # Parse data into dict to include legacy handling
    if isinstance(data, (list, tuple, np.ndarray)):
        iter_data = enumerate(data)
    elif isinstance(data, dict):
        iter_data = data.items()
    elif not data:
        iter_data = {}.items()
    else:
        iter_data = data

    # Format arg to tuple keys
    dict_data = {((key,) if isinstance(key, int) else tuple(key)): val for key, val in iter_data}

    # Add instructions for unspecified data
    if instructions and len(dict_data) < len(instructions):
        for i, inst in enumerate(instructions):
            if (i,) not in dict_data:
                dict_data[(i,)] = inst

    return dict_data
