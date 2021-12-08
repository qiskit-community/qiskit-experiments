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

"""Spectroscopy tests."""
from test.base import QiskitExperimentsTestCase

import numpy as np
from ddt import ddt, data, unpack
from qiskit import QuantumCircuit, circuit, pulse
from qiskit.providers.models import PulseBackendConfiguration
from qiskit.result import Result
from qiskit.test.mock import FakeBackend

from qiskit_experiments.library.characterization import cr_hamiltonian
from qiskit_experiments.library.characterization.analysis import cr_hamiltonian_analysis
from qiskit_experiments.test.utils import FakeJob


class CrossResonanceHamiltonianBackend(FakeBackend):
    """Fake backend to test cross resonance hamiltonian experiment."""

    def __init__(
        self,
        t_off: float = 0.0,
        ix: float = 0.0,
        iy: float = 0.0,
        iz: float = 0.0,
        zx: float = 0.0,
        zy: float = 0.0,
        zz: float = 0.0,
        b: float = 0.0,
        seed: int = 123,
    ):
        """Initialize fake backend.

        Args:
            t_off: Offset of gate duration.
            ix: IX term coefficient.
            iy: IY term coefficient.
            iz: IZ term coefficient.
            zx: ZX term coefficient.
            zy: ZY term coefficient.
            zz: ZZ term coefficient.
            b: Offset term.
            seed: Seed of random number generator used to generate count data.
        """
        self.fit_func_args = {
            "t_off": t_off,
            "px0": 2 * np.pi * (ix + zx),
            "px1": 2 * np.pi * (ix - zx),
            "py0": 2 * np.pi * (iy + zy),
            "py1": 2 * np.pi * (iy - zy),
            "pz0": 2 * np.pi * (iz + zz),
            "pz1": 2 * np.pi * (iz - zz),
            "b": b,
        }
        self.seed = seed
        configuration = PulseBackendConfiguration(
            backend_name="fake_cr_hamiltonian",
            backend_version="0.1.0",
            n_qubits=2,
            basis_gates=["sx", "rz", "x", "cx"],
            gates=None,
            local=True,
            simulator=True,
            conditional=False,
            open_pulse=True,
            memory=True,
            max_shots=10000,
            coupling_map=[[0, 1], [1, 0]],
            n_uchannels=2,
            u_channel_lo=[],
            meas_levels=[2],
            qubit_lo_range=[],
            meas_lo_range=[],
            dt=1,
            dtm=1,
            rep_times=[],
            meas_kernels=[],
            discriminators=[],
            channels={
                "d0": {"operates": {"qubits": [0]}, "purpose": "drive", "type": "drive"},
                "d1": {"operates": {"qubits": [1]}, "purpose": "drive", "type": "drive"},
                "u0": {
                    "operates": {"qubits": [0, 1]},
                    "purpose": "cross-resonance",
                    "type": "control",
                },
                "u1": {
                    "operates": {"qubits": [0, 1]},
                    "purpose": "cross-resonance",
                    "type": "control",
                },
            },
            timing_constraints={"granularity": 16, "min_length": 64},
        )

        super().__init__(configuration)
        setattr(
            self._configuration,
            "_control_channels",
            {
                (0, 1): [pulse.ControlChannel(0)],
                (1, 0): [pulse.ControlChannel(1)],
            },
        )

    def run(self, run_input, **kwargs):
        """Hard-coded experiment result generation based on the series definition."""
        results = []
        shots = kwargs.get("shots", 1024)
        rng = np.random.default_rng(seed=self.seed)

        series_defs = cr_hamiltonian_analysis.CrossResonanceHamiltonianAnalysis.__series__
        filter_kwargs_list = [sdef.filter_kwargs for sdef in series_defs]

        for test_circ in run_input:
            metadata = {
                "control_state": test_circ.metadata["control_state"],
                "meas_basis": test_circ.metadata["meas_basis"],
            }
            curve_ind = filter_kwargs_list.index(metadata)
            xval = test_circ.metadata["xval"]

            expv = series_defs[curve_ind].fit_func(xval, **self.fit_func_args)
            popl = 0.5 * (1 - expv)
            counts = rng.multinomial(shots, [1 - popl, popl])
            results.append(
                {
                    "shots": shots,
                    "success": True,
                    "header": {"metadata": test_circ.metadata},
                    "data": {"counts": dict(zip(["0", "1"], counts))},
                }
            )

        result = {
            "backend_name": self.name(),
            "backend_version": self.configuration().backend_version,
            "qobj_id": "12345",
            "job_id": "12345",
            "success": True,
            "results": results,
        }
        return FakeJob(backend=self, result=Result.from_dict(result))


@ddt
class TestCrossResonanceHamiltonian(QiskitExperimentsTestCase):
    """Test for cross resonance Hamiltonian tomography."""

    def test_circuit_generation(self):
        """Test generated circuits."""

        expr = cr_hamiltonian.CrossResonanceHamiltonian(
            qubits=(0, 1),
            flat_top_widths=[1000],
            amp=0.1,
            sigma=64,
            risefall=2,
        )
        expr.backend = CrossResonanceHamiltonianBackend()

        nearlest_16 = 1248

        with pulse.build(default_alignment="left", name="cr") as ref_cr_sched:
            pulse.play(
                pulse.GaussianSquare(
                    nearlest_16,
                    amp=0.1,
                    sigma=64,
                    width=1000,
                ),
                pulse.ControlChannel(0),
            )
            pulse.delay(nearlest_16, pulse.DriveChannel(0))
            pulse.delay(nearlest_16, pulse.DriveChannel(1))

        cr_gate = circuit.Gate("cr_gate", num_qubits=2, params=[1000])
        expr_circs = expr.circuits()

        x0_circ = QuantumCircuit(2, 1)
        x0_circ.append(cr_gate, [0, 1])
        x0_circ.h(1)
        x0_circ.measure(1, 0)

        x1_circ = QuantumCircuit(2, 1)
        x1_circ.x(0)
        x1_circ.append(cr_gate, [0, 1])
        x1_circ.h(1)
        x1_circ.measure(1, 0)

        y0_circ = QuantumCircuit(2, 1)
        y0_circ.append(cr_gate, [0, 1])
        y0_circ.sdg(1)
        y0_circ.h(1)
        y0_circ.measure(1, 0)

        y1_circ = QuantumCircuit(2, 1)
        y1_circ.x(0)
        y1_circ.append(cr_gate, [0, 1])
        y1_circ.sdg(1)
        y1_circ.h(1)
        y1_circ.measure(1, 0)

        z0_circ = QuantumCircuit(2, 1)
        z0_circ.append(cr_gate, [0, 1])
        z0_circ.measure(1, 0)

        z1_circ = QuantumCircuit(2, 1)
        z1_circ.x(0)
        z1_circ.append(cr_gate, [0, 1])
        z1_circ.measure(1, 0)

        ref_circs = [x0_circ, y0_circ, z0_circ, x1_circ, y1_circ, z1_circ]
        for c in ref_circs:
            c.add_calibration(cr_gate, (0, 1), ref_cr_sched)

        self.assertListEqual(expr_circs, ref_circs)

    @data(
        [1e6, 2e6, 1e3, -3e6, -2e6, 1e4],
        [-1e6, -2e6, 1e3, 3e6, 2e6, 1e4],
        [1e4, 2e4, 1e3, 5e6, 1e6, 2e3],
        [1e4, -1e3, 1e3, 5e5, 1e3, -1e3],  # low frequency test case 1
        [-1.0e5, 1.2e5, 1.0e3, 1.5e5, -1.1e5, -1.0e3],  # low frequency test case 2
    )
    @unpack
    # pylint: disable=invalid-name
    def test_integration(self, ix, iy, iz, zx, zy, zz):
        """Integration test for Hamiltonian tomography."""
        sigma = 20
        toff = np.sqrt(2 * np.pi) * sigma * 1e-9

        backend = CrossResonanceHamiltonianBackend(toff, ix, iy, iz, zx, zy, zz)
        durations = np.linspace(0, 700, 50)
        expr = cr_hamiltonian.CrossResonanceHamiltonian(
            qubits=(0, 1), flat_top_widths=durations, sigma=sigma, risefall=2
        )
        exp_data = expr.run(backend, shots=2000)

        self.assertEqual(exp_data.analysis_results(0).quality, "good")
        self.assertAlmostEqual(exp_data.analysis_results("omega_ix").value.value, ix, delta=2e4)
        self.assertAlmostEqual(exp_data.analysis_results("omega_iy").value.value, iy, delta=2e4)
        self.assertAlmostEqual(exp_data.analysis_results("omega_iz").value.value, iz, delta=2e4)
        self.assertAlmostEqual(exp_data.analysis_results("omega_zx").value.value, zx, delta=2e4)
        self.assertAlmostEqual(exp_data.analysis_results("omega_zy").value.value, zy, delta=2e4)
        self.assertAlmostEqual(exp_data.analysis_results("omega_zz").value.value, zz, delta=2e4)

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = cr_hamiltonian.CrossResonanceHamiltonian(
            qubits=[0, 1],
            flat_top_widths=[1000],
            amp=0.1,
            sigma=64,
            risefall=2,
        )
        loaded_exp = cr_hamiltonian.CrossResonanceHamiltonian.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.experiments_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = cr_hamiltonian.CrossResonanceHamiltonian(
            qubits=[0, 1],
            flat_top_widths=[1000],
            amp=0.1,
            sigma=64,
            risefall=2,
        )
        self.assertRoundTripSerializable(exp, self.experiments_equiv)
