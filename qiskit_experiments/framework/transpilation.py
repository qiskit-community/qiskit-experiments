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
Functions for preparing circuits for execution
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Sequence

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.exceptions import QiskitError
from qiskit.providers import Backend
from qiskit.providers.options import Options
from qiskit.pulse.calibration_entries import CalibrationPublisher
from qiskit.transpiler import Target


LOGGER = logging.getLogger(__file__)

DEFAULT_TRANSPILE_OPTIONS = Options(optimization_level=0, full_transpile=False)
if importlib.metadata.version("qiskit").partition(".")[0] != "0":
    DEFAULT_TRANSPILE_OPTIONS["num_processes"] = 1


def map_qubits(
    circuit: QuantumCircuit,
    physical_qubits: Sequence[int],
    n_qubits: int | None = None,
) -> QuantumCircuit:
    """Generate a new version of a circuit with new qubit indices

    This function iterates through the instructions of ``circuit`` and copies
    them into a new circuit with qubit indices replaced according to the
    entries in ``physical_qubits``. So qubit 0's instructions are applied to
    ``physical_qubits[0]`` and qubit 1's to ``physical_qubits[1]``, etc.

    This function behaves similarly to passing ``initial_layout`` to
    :func:`qiskit.transpile` but does not use a Qiskit
    :class:`~qiskit.transpiler.PassManager` and does not fill the circuit with
    ancillas.

    Args:
        circuit: The :class:`~qiskit.QuantumCircuit` to re-index.
        physical_qubits: The list of new indices for ``circuit``'s qubit indices.
        n_qubits: Optional qubit size to use for the output circuit. If
            ``None``, then the maximum of ``physical_qubits`` will be used.

    Returns:
        The quantum circuit with new qubit indices
    """
    if len(physical_qubits) != circuit.num_qubits:
        raise QiskitError(
            f"Circuit to map has {circuit.num_qubits} qubits, but "
            f"{len(physical_qubits)} physical qubits specified for mapping."
        )

    # if all(p == r for p, r in zip(physical_qubits, range(circuit.num_qubits))):
    #     # No mapping necessary
    #     return circuit

    circ_size = n_qubits if n_qubits is not None else (max(physical_qubits) + 1)
    p_qregs = QuantumRegister(circ_size)
    p_circ = QuantumCircuit(
        p_qregs,
        *circuit.cregs,
        name=circuit.name,
        metadata=circuit.metadata,
        global_phase=circuit.global_phase,
    )
    p_circ.compose(
        circuit,
        qubits=physical_qubits,
        inplace=True,
        copy=False,
    )
    return p_circ


def _has_calibration(target: Target, name: str, qubits: tuple[int, ...]) -> bool:
    """Wrapper to work around bug in Target.has_calibration"""
    try:
        has_cal = target.has_calibration(name, qubits)
    except AttributeError:
        has_cal = False

    return has_cal


def check_transpilation_needed(
    circuits: Sequence[QuantumCircuit],
    backend: Backend,
) -> bool:
    """Test if circuits are already compatible with backend

    This function checks if circuits are able to be executed on ``backend``
    without transpilation. It loops through the circuits to check if any gate
    instructions are not included in the backend's
    :class:`~qiskit.transpiler.Target`.  The :class:`~qiskit.transpiler.Target`
    is also checked for custom pulse gate calibrations for circuit's
    instructions.  If all gates are included in the target and there are no
    custom calibrations, the function returns ``False`` indicating that
    transpilation is not needed.

    This function returns ``True`` if the version of ``backend`` is less than
    2.

    The motivation for this function is that when no transpilation is necessary
    it is faster to check the circuits in this way than to run
    :func:`~qiskit.transpile` and have it do nothing.

    Args:
        circuits: The circuits to prepare for the backend.
        backend: The backend for which the circuits should be prepared.

    Returns:
        ``True`` if transpilation is needed. Otherwise, ``False``.
    """
    transpilation_needed = False

    if getattr(backend, "version", 0) <= 1:
        # Fall back to transpilation for BackendV1
        return True

    target = backend.target

    for circ in circuits:
        for inst in circ.data:
            if inst.operation.name == "barrier" or circ.has_calibration_for(inst):
                continue
            qubits = tuple(circ.find_bit(q).index for q in inst.qubits)
            if not target.instruction_supported(inst.operation.name, qubits):
                transpilation_needed = True
                break
            if _has_calibration(target, inst.operation.name, qubits):
                cal = target.get_calibration(inst.operation.name, qubits, *inst.operation.params)
                if (
                    cal.metadata.get("publisher", CalibrationPublisher.QISKIT)
                    != CalibrationPublisher.BACKEND_PROVIDER
                ):
                    transpilation_needed = True
                    break
        if transpilation_needed:
            break

    return transpilation_needed


def minimal_transpile(
    circuits: Sequence[QuantumCircuit],
    backend: Backend,
    options: Options,
) -> list[QuantumCircuit]:
    """Prepare circuits for execution on a backend

    This function  is a wrapper around :func:`~qiskit.transpile` to prepare
    circuits for execution ``backend`` that tries to do less work in the case
    in which the ``circuits`` can already be executed on the backend without
    modification.

    The instructions in ``circuits`` are checked to see if they can be executed
    by the ``backend`` using :func:`check_transpilation_needed`. If the
    circuits can not be executed, :func:`~qiskit.transpile` is called on them.
    ``options`` is a set of options to pass to the :func:`~qiskit.transpile`
    (see detailed description of ``options``). The special ``full_transpile``
    option can also be set to ``True`` to force calling
    :func:`~qiskit.transpile`.

    Args:
        circuits: The circuits to prepare for the backend.
        backend: The backend for which the circuits should be prepared.
        options:  Options for the transpilation. ``full_transpile`` can be set
            to ``True`` to force this function to pass the circuits to
            :func:`~qiskit.transpile`. Other options are passed as arguments to
            :func:`qiskit.transpile` if it is called.

    Returns:
        The prepared circuits
    """
    options = dict(options.items())

    if "full_transpile" not in options:
        LOGGER.debug(
            "Performing full transpile because base transpile options "
            "were overwritten and full_transpile was not specified."
        )
        full_transpile = True
    else:
        full_transpile = options.pop("full_transpile", False)
    if not full_transpile and set(options) - set(DEFAULT_TRANSPILE_OPTIONS):
        # If an experiment specifies transpile options, it needs to go
        # through transpile()
        full_transpile = True
        LOGGER.debug(
            "Performing full transpile because non-default transpile options are specified."
        )

    if not full_transpile:
        full_transpile = check_transpilation_needed(circuits, backend)

    import inspect
    import unittest
    try:
        test_frame = next(f[0] for f in inspect.stack() if any(isinstance(l, unittest.TestCase) for n, l in f[0].f_locals.items()))
        test = next(v for v in test_frame.f_locals.values())
        print(f"full_transpile={full_transpile} for {test.id()}")
    except StopIteration:
        pass
    if full_transpile:
        transpiled = transpile(circuits, backend, **options)
    else:
        transpiled = circuits

    return transpiled
