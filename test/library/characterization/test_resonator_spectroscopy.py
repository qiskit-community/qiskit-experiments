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

"""Spectroscopy tests for resonator spectroscopy experiment."""

from test.base import QiskitExperimentsTestCase
from typing import Any, List, Tuple

import numpy as np
from ddt import data, ddt
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.circuit.library import CXGate, Measure, XGate
from qiskit.qobj.utils import MeasLevel

from qiskit_experiments.framework import BackendData, ParallelExperiment
from qiskit_experiments.library import ResonatorSpectroscopy
from qiskit_experiments.database_service import Resonator
from qiskit_experiments.test.mock_iq_backend import MockIQBackend, MockIQParallelBackend
from qiskit_experiments.test.mock_iq_helpers import (
    MockIQParallelExperimentHelper as ParallelExperimentHelper,
)
from qiskit_experiments.test.mock_iq_helpers import (
    MockIQSpectroscopyHelper as ResonatorSpectroscopyHelper,
)


class MockIQBackendDefaults(MockIQBackend):
    """MockIQBackend with defaults() method"""

    def defaults(self):
        """Pulse defaults

        NOTE: ResonatorSpectroscopy still relies on defaults() so we add here.
        Because defaults() is not in the BackendV2 base class, we do not add it
        to Backend classes outside of this test module so that we do not
        introduce new dependencies on it.
        """
        return self._defaults


class MockIQParallelBackendDefaults(MockIQParallelBackend):
    """MockIQParallelBackend with defaults() method"""

    def defaults(self):
        """Pulse defaults

        NOTE: ResonatorSpectroscopy still relies on defaults() so we add here.
        Because defaults() is not in the BackendV2 base class, we do not add it
        to Backend classes outside of this test module so that we do not
        introduce new dependencies on it.
        """
        return self._defaults


def data_valid_initial_circuits() -> List[Tuple[Any, str]]:
    """Returns a list of parameters for ``test_valid_initial_circuits``.

    Returns:
        list: List of tuples containing valid circuits and labels identifying what they represent.
    """

    good_circuit = QuantumCircuit(1)
    good_circuit.x(0)
    good_initial_gate = XGate()

    return [
        (good_circuit, "normal initial circuit"),
        (good_initial_gate, "normal initial single-qubit gate"),
    ]


def data_invalid_initial_circuits() -> List[Tuple[Any, str]]:
    """Returns a list of parameters for ``test_invalid_initial_circuits``.

    Returns:
        list: List of tuples containing invalid circuits and labels identifying what they represent.
    """
    two_qubit_circuit = QuantumCircuit(2)
    two_qubit_circuit.x(1)
    two_qubit_gate_circuit = QuantumCircuit(2)
    two_qubit_gate_circuit.cx(0, 1)
    two_qubit_gate = CXGate()
    one_qubit_one_clbit_gate = Measure()

    return [
        (two_qubit_circuit, "two qubit initial circuit with one single-qubit gate"),
        (two_qubit_gate_circuit, "two qubit initial circuit with a two-qubit gate"),
        (two_qubit_gate, "two qubit gate"),
        (one_qubit_one_clbit_gate, "one qubit and one classical bit gate"),
    ]


@ddt
class TestResonatorSpectroscopy(QiskitExperimentsTestCase):
    """Tests for the resonator spectroscopy experiment."""

    @data(-5e6, -2e6, 0, 1e6, 3e6)
    def test_end_to_end(self, freq_shift):
        """Test the experiment from end to end."""

        qubit = 1
        backend = MockIQBackendDefaults(
            experiment_helper=ResonatorSpectroscopyHelper(
                gate_name="measure",
                freq_offset=freq_shift,
                iq_cluster_centers=[((0.0, 0.0), (-1.0, 0.0))],
                iq_cluster_width=[0.2],
            ),
        )

        res_freq = BackendData(backend).meas_freqs[qubit]

        frequencies = np.linspace(res_freq - 20e6, res_freq + 20e6, 51)
        spec = ResonatorSpectroscopy([qubit], backend=backend, frequencies=frequencies)

        expdata = spec.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results("res_freq0")
        self.assertRoundTripSerializable(result.value)

        self.assertAlmostEqual(result.value.n, res_freq + freq_shift, delta=0.1e6)
        self.assertEqual(str(result.device_components[0]), f"R{qubit}")
        self.assertEqual(expdata.metadata["device_components"], [Resonator(qubit)])

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = ResonatorSpectroscopy([1], frequencies=np.linspace(100, 150, 4) * 1e6)
        loaded_exp = ResonatorSpectroscopy.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqualExtended(exp, loaded_exp)

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = ResonatorSpectroscopy([1], frequencies=np.linspace(int(100e6), int(150e6), 4))
        self.assertRoundTripSerializable(exp)

    def test_circuit_roundtrip_serializable(self):
        """Test circuits data JSON serialization"""
        freq_shift = 20e4
        qubit = 1
        # need backend for dt value in the experiment
        backend = MockIQBackendDefaults(
            experiment_helper=ResonatorSpectroscopyHelper(
                gate_name="measure",
                freq_offset=freq_shift,
                iq_cluster_centers=[((0.0, 0.0), (-1.0, 0.0))],
                iq_cluster_width=[0.2],
            ),
        )
        res_freq = BackendData(backend).meas_freqs[qubit]
        frequencies = np.linspace(res_freq - 20e6, res_freq + 20e6, 3)
        exp = ResonatorSpectroscopy([qubit], backend=backend, frequencies=frequencies)
        self.assertRoundTripSerializable(exp._transpiled_circuits())

    @data(-5e6, 0, 3e6)
    def test_kerneled_expdata_serialization(self, freq_shift):
        """Test experiment data and analysis data JSON serialization"""
        qubit = 1
        backend = MockIQBackendDefaults(
            experiment_helper=ResonatorSpectroscopyHelper(
                gate_name="measure",
                freq_offset=freq_shift,
                iq_cluster_centers=[((0.0, 0.0), (-1.0, 0.0))],
                iq_cluster_width=[0.2],
            ),
        )

        res_freq = BackendData(backend).meas_freqs[qubit]

        frequencies = np.linspace(res_freq - 20e6, res_freq + 20e6, 11)
        exp = ResonatorSpectroscopy([qubit], backend=backend, frequencies=frequencies)

        expdata = exp.run(backend)
        self.assertExperimentDone(expdata)

        # since under _experiment in kwargs there is an argument of the backend which isn't serializable.
        expdata._experiment = None
        # Checking serialization of the experiment data
        self.assertRoundTripSerializable(expdata)

        # Checking serialization of the analysis
        self.assertRoundTripSerializable(expdata.analysis_results("res_freq0"))

    def test_parallel_experiment(self):
        """Test for parallel experiment"""
        # backend initialization
        iq_cluster_centers = [
            ((-1.0, 0.0), (1.0, 0.0)),
            ((0.0, -1.0), (0.0, 1.0)),
            ((3.0, 0.0), (5.0, 0.0)),
        ]

        freq_shift = [-5e6, 3e6]
        exp_helper_list = [
            ResonatorSpectroscopyHelper(
                gate_name="measure",
                freq_offset=freq_shift[0],
                iq_cluster_centers=iq_cluster_centers,
            ),
            ResonatorSpectroscopyHelper(
                gate_name="measure",
                freq_offset=freq_shift[1],
                iq_cluster_centers=iq_cluster_centers,
            ),
        ]

        parallel_backend = MockIQParallelBackendDefaults(
            experiment_helper=None,
            rng_seed=0,
        )

        qubit1 = 0
        qubit2 = 1

        backend_data = BackendData(parallel_backend)
        res_freq1 = backend_data.meas_freqs[qubit1]
        res_freq2 = backend_data.meas_freqs[qubit2]

        frequencies1 = np.linspace(res_freq1 - 20e6, res_freq1 + 20e6, 11)
        frequencies2 = np.linspace(res_freq2 - 20e6, res_freq2 + 20e6, 13)

        res_spect1 = ResonatorSpectroscopy(
            [qubit1], backend=parallel_backend, frequencies=frequencies1
        )
        res_spect2 = ResonatorSpectroscopy(
            [qubit2], backend=parallel_backend, frequencies=frequencies2
        )

        exp_list = [res_spect1, res_spect2]

        # Initializing parallel helper
        parallel_helper = ParallelExperimentHelper(exp_list, exp_helper_list)

        # setting the helper into the backend
        parallel_backend.experiment_helper = parallel_helper

        par_experiment = ParallelExperiment(
            exp_list, flatten_results=False, backend=parallel_backend
        )
        par_experiment.set_run_options(meas_level=MeasLevel.KERNELED, meas_return="single")

        par_data = par_experiment.run()
        self.assertExperimentDone(par_data)

        # since under _experiment in kwargs there is an argument of the backend which isn't serializable.
        par_data._experiment = None
        # Checking serialization of the experiment data
        self.assertRoundTripSerializable(par_data)

        for child_data in par_data.child_data():
            self.assertRoundTripSerializable(child_data)
            for analysis_result in child_data.analysis_results():
                self.assertRoundTripSerializable(analysis_result)

    def test_initial_circuit_transpiled(self):
        """Test that the initial circuit is added to the experiment circuits correctly."""
        # Create backend to assist with circuit creation
        backend = MockIQBackendDefaults(
            experiment_helper=ResonatorSpectroscopyHelper(
                gate_name="measure",
                freq_offset=1e6,
                iq_cluster_centers=[((0.0, 0.0), (-1.0, 0.0))],
                iq_cluster_width=[0.2],
            ),
        )

        # Create arbitrary initial circuit
        initial_circuit = QuantumCircuit(1, name="initial_circuit")
        initial_circuit.x(0)

        # Create resonator spectroscopy experiments. We only use 3 frequencies to reduce the number of
        # circuits to check.
        res_spec_no_initial = ResonatorSpectroscopy(
            [0],
            backend=backend,
            frequencies=[5e9, 5.05e9, 5.1e9],
        )
        res_spec_initial = ResonatorSpectroscopy(
            [0],
            backend=backend,
            frequencies=[5e9, 5.05e9, 5.1e9],
        )
        res_spec_initial.set_experiment_options(initial_circuit=initial_circuit)

        # Check depths and widths to verify that an initial circuit has been added.
        for circ in res_spec_no_initial.circuits():
            self.assertEqual(circ.width(), 2, msg="Circuit width was not as expected.")
            self.assertEqual(circ.depth(), 1, msg="Circuit depth was not as expected.")
        for circ in res_spec_initial.circuits():
            self.assertEqual(
                circ.width(), 2, msg="Circuit, with initial_circuit, width was not as expected."
            )
            self.assertEqual(
                circ.depth(), 2, msg="Circuit, with initial_circuit, depth was not as expected."
            )

        # Check depths and widths for transpiled circuits
        initial_circuit_depth = initial_circuit.depth()
        for circ in res_spec_no_initial._transpiled_circuits():
            self.assertEqual(
                circ.width(),
                # Width is the number of qubits + 1 classical bit.
                backend.num_qubits + 1,
                msg="Transpiled circuit width was not as expected.",
            )
            self.assertEqual(
                circ.depth(),
                1,
                msg="Transpiled circuit depth was not as expected.",
            )
        for circ in res_spec_initial._transpiled_circuits():
            self.assertEqual(
                circ.width(),
                # Width is the number of qubits + 1 classical bit.
                backend.num_qubits + 1,
                msg="Transpiled circuit, with initial_circuit, width was not as expected.",
            )
            self.assertEqual(
                circ.depth(),
                initial_circuit_depth + 1,
                msg="Transpiled circuit, with initial_circuit, depth was not as expected.",
            )

    @data(*data_valid_initial_circuits())
    def test_valid_initial_circuits(self, params):
        """Test successful setting of valid ``initial_circuit`` values."""
        circuit, circuit_label = params
        backend = MockIQBackendDefaults(
            experiment_helper=ResonatorSpectroscopyHelper(
                gate_name="measure",
                freq_offset=1e6,
                iq_cluster_centers=[((0.0, 0.0), (-1.0, 0.0))],
                iq_cluster_width=[0.2],
            ),
        )

        res_spec = ResonatorSpectroscopy([0], backend)
        try:
            res_spec.set_experiment_options(initial_circuit=circuit)
        except QiskitError as exp_exception:
            self.fail(
                f"Setting initial circuit to '{circuit_label}' failed with exception {exp_exception}."
            )

    @data(*data_invalid_initial_circuits())
    def test_invalid_initial_circuits(self, params):
        """Test detection of invalid ``initial_circuit`` values."""
        circuit, circuit_label = params
        backend = MockIQBackendDefaults(
            experiment_helper=ResonatorSpectroscopyHelper(
                gate_name="measure",
                freq_offset=1e6,
                iq_cluster_centers=[((0.0, 0.0), (-1.0, 0.0))],
                iq_cluster_width=[0.2],
            ),
        )

        res_spec = ResonatorSpectroscopy([0], backend)
        with self.assertRaises(
            QiskitError,
            msg=f"Setting initial circuit to invalid '{circuit_label}' did not fail with exception.",
        ):
            res_spec.set_experiment_options(initial_circuit=circuit)
