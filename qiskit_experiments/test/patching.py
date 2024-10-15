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

"""Temporary monkey-patching test support for BackednSamplerV2"""
from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

import qiskit.primitives.backend_sampler_v2
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.primitives import (
    BackendEstimatorV2,
    BackendSamplerV2,
)
from qiskit.primitives.containers import (
    BitArray,
    DataBin,
    SamplerPubResult,
)
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.result import Result
from qiskit_ibm_runtime.fake_provider.local_service import QiskitRuntimeLocalService


# The rest of this file contains definitions for monkey patching support for
# level 1 data and a noise model run option into BackendSamplerV2
def _patched_run_circuits(
    circuits: QuantumCircuit | list[QuantumCircuit],
    backend: BackendV1 | BackendV2,
    **run_options,
) -> tuple[list[Result], list[dict]]:
    """Remove metadata of circuits and run the circuits on a backend.
    Args:
        circuits: The circuits
        backend: The backend
        monitor: Enable job minotor if True
        **run_options: run_options
    Returns:
        The result and the metadata of the circuits
    """
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
    metadata = []
    for circ in circuits:
        metadata.append(circ.metadata)
        # Commenting out this line is only change from qiskit.primitives.backend_estimator._run_circuits
        # circ.metadata = {}
    if isinstance(backend, BackendV1):
        max_circuits = getattr(backend.configuration(), "max_experiments", None)
    elif isinstance(backend, BackendV2):
        max_circuits = backend.max_circuits
    else:
        raise RuntimeError("Backend version not supported")
    if max_circuits:
        jobs = [
            backend.run(circuits[pos : pos + max_circuits], **run_options)
            for pos in range(0, len(circuits), max_circuits)
        ]
        result = [x.result() for x in jobs]
    else:
        result = [backend.run(circuits, **run_options).result()]
    return result, metadata


def _patched_run_backend_primitive_v2(
    self,  # pylint: disable=unused-argument
    backend: BackendV1 | BackendV2,
    primitive: Literal["sampler", "estimator"],
    options: dict,
    inputs: dict,
) -> PrimitiveJob:
    """Run V2 backend primitive.

    Args:
        backend: The backend to run the primitive on.
        primitive: Name of the primitive.
        options: Primitive options to use.
        inputs: Primitive inputs.

    Returns:
        The job object of the result of the primitive.
    """
    options_copy = copy.deepcopy(options)

    prim_options = {}
    sim_options = options_copy.get("simulator", {})
    if seed_simulator := sim_options.pop("seed_simulator", None):
        prim_options["seed_simulator"] = seed_simulator
    if noise_model := sim_options.pop("noise_model", None):
        prim_options["noise_model"] = noise_model
    if not sim_options:
        options_copy.pop("simulator", None)
    if primitive == "sampler":
        if default_shots := options_copy.pop("default_shots", None):
            prim_options["default_shots"] = default_shots
        if meas_type := options_copy.get("execution", {}).pop("meas_type", None):
            if meas_type == "classified":
                prim_options["meas_level"] = 2
                prim_options["meas_return"] = "single"
            elif meas_type == "kerneled":
                prim_options["meas_level"] = 1
                prim_options["meas_return"] = "single"
            elif meas_type == "avg_kerneled":
                prim_options["meas_level"] = 1
                prim_options["meas_return"] = "avg"
            else:
                options_copy["execution"]["meas_type"] = meas_type

            if not options_copy["execution"]:
                del options_copy["execution"]

        primitive_inst = BackendSamplerV2(backend=backend, options=prim_options)
    else:
        if default_shots := options_copy.pop("default_shots", None):
            inputs["precision"] = 1 / math.sqrt(default_shots)
        if default_precision := options_copy.pop("default_precision", None):
            prim_options["default_precision"] = default_precision
        primitive_inst = BackendEstimatorV2(backend=backend, options=prim_options)

    if options_copy:
        warnings.warn(f"Options {options_copy} have no effect in local testing mode.")

    return primitive_inst.run(**inputs)


@dataclass
class Options:
    """Options for :class:`~.BackendSamplerV2`"""

    default_shots: int = 1024
    """The default shots to use if none are specified in :meth:`~.run`.
    Default: 1024.
    """

    seed_simulator: int | None = None
    """The seed to use in the simulator. If None, a random seed will be used.
    Default: None.
    """

    noise_model: Any | None = None
    meas_level: int | None = None
    meas_return: str | None = None


def _patched_run_pubs(self, pubs: list[SamplerPub], shots: int) -> list[SamplerPubResult]:
    """Compute results for pubs that all require the same value of ``shots``."""
    # prepare circuits
    bound_circuits = [pub.parameter_values.bind_all(pub.circuit) for pub in pubs]
    flatten_circuits = []
    for circuits in bound_circuits:
        flatten_circuits.extend(np.ravel(circuits).tolist())

    # run circuits
    run_opts = {
        k: getattr(self._options, k)
        for k in ("noise_model", "meas_return", "meas_level")
        if getattr(self._options, k) is not None
    }
    results, _ = _patched_run_circuits(
        flatten_circuits,
        self._backend,
        memory=True,
        shots=shots,
        seed_simulator=self._options.seed_simulator,
        **run_opts,
    )
    result_memory = qiskit.primitives.backend_sampler_v2._prepare_memory(results)

    # pack memory to an ndarray of uint8
    results = []
    start = 0
    for pub, bound in zip(pubs, bound_circuits):
        meas_info, max_num_bytes = qiskit.primitives.backend_sampler_v2._analyze_circuit(
            pub.circuit
        )
        end = start + bound.size
        results.append(
            self._postprocess_pub(
                result_memory[start:end],
                shots,
                bound.shape,
                meas_info,
                max_num_bytes,
                pub.circuit.metadata,
                meas_level=self._options.meas_level,
            )
        )
        start = end

    return results


def _patched_postprocess_pub(
    self,  # pylint: disable=unused-argument
    result_memory: list[list[str]],
    shots: int,
    shape: tuple[int, ...],
    meas_info: list[qiskit.primitives.backend_sampler_v2._MeasureInfo],
    max_num_bytes: int,
    circuit_metadata: dict,
    meas_level: int | None = None,
) -> SamplerPubResult:
    """Converts the memory data into an array of bit arrays with the shape of the pub."""
    if meas_level == 2 or meas_level is None:
        arrays = {
            item.creg_name: np.zeros(shape + (shots, item.num_bytes), dtype=np.uint8)
            for item in meas_info
        }
        memory_array = qiskit.primitives.backend_sampler_v2._memory_array(
            result_memory, max_num_bytes
        )

        for samples, index in zip(memory_array, np.ndindex(*shape)):
            for item in meas_info:
                ary = qiskit.primitives.backend_sampler_v2._samples_to_packed_array(
                    samples, item.num_bits, item.start
                )
                arrays[item.creg_name][index] = ary

        meas = {
            item.creg_name: BitArray(arrays[item.creg_name], item.num_bits) for item in meas_info
        }
    elif meas_level == 1:
        raw = np.array(result_memory)
        cplx = raw[..., 0] + 1j * raw[..., 1]
        cplx = np.reshape(cplx, (*shape, *cplx.shape[1:]))
        meas = {item.creg_name: cplx for item in meas_info}
    else:
        raise QiskitError(f"Unsupported meas_level: {meas_level}")
    return SamplerPubResult(
        DataBin(**meas, shape=shape),
        metadata={"shots": shots, "circuit_metadata": circuit_metadata},
    )


def patch_sampler_test_support():
    """Monkey-patching to pass metadata through to test backends and support level 1"""
    warnings.filterwarnings("ignore", ".*Could not determine job completion time.*", UserWarning)
    qiskit.primitives.backend_sampler_v2.Options = Options
    QiskitRuntimeLocalService._run_backend_primitive_v2 = _patched_run_backend_primitive_v2
    BackendSamplerV2._run_pubs = _patched_run_pubs
    BackendSamplerV2._postprocess_pub = _patched_postprocess_pub
