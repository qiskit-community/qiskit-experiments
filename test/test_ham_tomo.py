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
"""Test for Hamiltonian tomography experiments."""

import numpy as np
from ddt import ddt, data, unpack
from qiskit import circuit, pulse
from qiskit.result import Result
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeAthens

from qiskit_experiments.analysis import CurveAnalysisResult
from qiskit_experiments.hamiltonian_tomography import (
    CRHamiltonianTomography,
    CRHamiltonianAnalysis,
)


class CRHamiltonianTomographyBackend(FakeAthens):
    """Mock backend for CR Hamiltonian tomography testing."""

    # pylint: disable = invalid-name
    def __init__(
        self, ix: float, iy: float, iz: float, zx: float, zy: float, zz: float, b: float = 0
    ):
        self.fit_func_args = {
            "px0": 2 * np.pi * (ix + zx),
            "px1": 2 * np.pi * (ix - zx),
            "py0": 2 * np.pi * (iy + zy),
            "py1": 2 * np.pi * (iy - zy),
            "pz0": 2 * np.pi * (iz + zz),
            "pz1": 2 * np.pi * (iz - zz),
            "b": b,
        }
        super().__init__()

    def run(self, run_input, **kwargs):

        try:
            shots = kwargs["shots"]
        except AttributeError as ex:
            raise AttributeError(
                "shots is not specified in run options. "
                "This argument should be explicitly specified to run test."
            ) from ex

        results = []
        for test_circ in run_input:
            circuit_metadata = {
                "control_state": test_circ.metadata["control_state"],
                "meas_basis": test_circ.metadata["meas_basis"],
            }
            xval = test_circ.metadata["xval"]

            for series_def in CRHamiltonianAnalysis.__series__:
                if series_def.filter_kwargs == circuit_metadata:
                    fit_func = series_def.fit_func
                    break
            else:
                assert KeyError(
                    f"Invalid data is provided. Cannot test this input: {circuit_metadata}"
                )

            expv = fit_func(xval, **self.fit_func_args)
            popl = 0.5 * (1 - expv)
            one_count = int(np.round(shots * popl))
            results.append(
                {
                    "shots": shots,
                    "success": True,
                    "header": {"metadata": test_circ.metadata},
                    "data": {"counts": {"0": shots - one_count, "1": one_count}},
                }
            )

        return Result.from_dict(
            {
                "backend_name": self.name(),
                "backend_version": self.configuration().backend_version,
                "qobj_id": "12345",
                "job_id": "12345",
                "success": True,
                "results": results,
            }
        )


@ddt
class TestCRHamiltonianTomography(QiskitTestCase):
    """Test for CR Hamiltonian Tomography."""

    def test_tomo_circs(self):
        """Test generated circuits."""
        durations = np.asarray([20], dtype=int)
        exp = CRHamiltonianTomography(qubits=(0, 1), durations=durations)
        exp.set_experiment_options(amp=0.1, sigma=2.0, risefall=2.0, alignment=1, unit="s", dt=0.1)

        circs = exp.circuits(backend=FakeAthens())

        ref_circ0_meta = {
            "experiment_type": "CRHamiltonianTomography",
            "qubits": (0, 1),
            "xval": (120 + np.sqrt(2 * np.pi * 400)) * 0.1,
            "dt": 0.1,
            "duration": 200,
            "control_state": 0,
            "meas_basis": "x",
            "amplitude": 0.1,
            "sigma": 20.0,
            "risefall": 2.0,
        }
        self.assertDictEqual(circs[0].metadata, ref_circ0_meta)

        with pulse.build() as ref_sched:
            with pulse.align_left():
                pulse.play(
                    pulse.GaussianSquare(duration=200, amp=0.1, sigma=20.0, width=120.0),
                    pulse.ControlChannel(0),
                )
                pulse.delay(duration=200, channel=pulse.DriveChannel(0))
                pulse.delay(duration=200, channel=pulse.DriveChannel(1))

        self.assertEqual(circs[0].calibrations["cr_gate"][((0, 1), (200.0,))], ref_sched)

    def test_tomo_circs_custom_sched(self):
        """Test generated circuits with custom pulse schedule."""
        dur = circuit.Parameter("duration")
        with pulse.build() as ecr_sched:
            with pulse.align_sequential():
                ecr_dur = dur / 2
                pulse.play(
                    pulse.GaussianSquare(
                        duration=ecr_dur,
                        amp=0.1,
                        sigma=20.0,
                        width=ecr_dur - 4 * 20.0,
                    ),
                    pulse.ControlChannel(0),
                )
                pulse.play(pulse.Gaussian(10, 0.1, 2), pulse.DriveChannel(0))
                pulse.play(
                    pulse.GaussianSquare(
                        duration=ecr_dur,
                        amp=-0.1,
                        sigma=20.0,
                        width=ecr_dur - 4 * 20.0,
                    ),
                    pulse.ControlChannel(0),
                )
        xvalues = np.asarray([321.0], dtype=float)

        durations = np.asarray([20], dtype=int)
        exp = CRHamiltonianTomography(
            qubits=(0, 1),
            durations=durations,
            cr_gate_schedule=ecr_sched,
            x_values=xvalues,
        )
        exp.set_experiment_options(amp=0.1, sigma=2.0, risefall=2.0, alignment=1, unit="s", dt=0.1)

        circs = exp.circuits(backend=FakeAthens())

        ref_circ0_meta = {
            "experiment_type": "CRHamiltonianTomography",
            "qubits": (0, 1),
            "xval": 321.0,
            "dt": 0.1,
            "duration": 200,
            "control_state": 0,
            "meas_basis": "x",
        }
        self.assertDictEqual(circs[0].metadata, ref_circ0_meta)

        with pulse.build(backend=FakeAthens()) as ref_sched:
            with pulse.align_sequential():
                pulse.play(
                    pulse.GaussianSquare(duration=100, amp=0.1, sigma=20, width=20),
                    pulse.ControlChannel(0),
                )
                pulse.play(pulse.Gaussian(10, 0.1, 2), pulse.DriveChannel(0))
                pulse.play(
                    pulse.GaussianSquare(duration=100, amp=-0.1, sigma=20, width=20),
                    pulse.ControlChannel(0),
                )

        self.assertEqual(circs[0].calibrations["cr_gate"][((0, 1), (200.0,))], ref_sched)

    def test_hamiltonian_coefficients(self):
        """Test that calculates Hamiltonian coefficients from fit parameters."""

        fit_params_ref = {
            "popt_keys": ["px0", "py0", "pz0", "px1", "py1", "pz1"],
            "popt": [
                2 * np.pi * 1.0,
                2 * np.pi * 2.0,
                2 * np.pi * 3.0,
                2 * np.pi * 4.0,
                2 * np.pi * 5.0,
                2 * np.pi * 6.0,
            ],
            "popt_err": [
                2 * np.pi * 0.01,
                2 * np.pi * 0.02,
                2 * np.pi * 0.03,
                2 * np.pi * 0.04,
                2 * np.pi * 0.05,
                2 * np.pi * 0.06,
            ],
        }
        test_result = CurveAnalysisResult(**fit_params_ref)

        analysis = CRHamiltonianAnalysis()
        processed_result = analysis._post_processing(test_result)

        self.assertEqual(processed_result["IX"], 2.5)
        self.assertEqual(processed_result["IY"], 3.5)
        self.assertEqual(processed_result["IZ"], 4.5)
        self.assertEqual(processed_result["ZX"], -1.5)
        self.assertEqual(processed_result["ZY"], -1.5)
        self.assertEqual(processed_result["ZZ"], -1.5)

        self.assertEqual(processed_result["IX_err"], 0.5 * np.sqrt(0.01 ** 2 + 0.04 ** 2))
        self.assertEqual(processed_result["IY_err"], 0.5 * np.sqrt(0.02 ** 2 + 0.05 ** 2))
        self.assertEqual(processed_result["IZ_err"], 0.5 * np.sqrt(0.03 ** 2 + 0.06 ** 2))
        self.assertEqual(processed_result["ZX_err"], 0.5 * np.sqrt(0.01 ** 2 + 0.04 ** 2))
        self.assertEqual(processed_result["ZY_err"], 0.5 * np.sqrt(0.02 ** 2 + 0.05 ** 2))
        self.assertEqual(processed_result["ZZ_err"], 0.5 * np.sqrt(0.03 ** 2 + 0.06 ** 2))

    # pylint: disable = invalid-name
    @data(
        [1e6, 2e6, 1e3, -3e6, -2e6, 1e4],
        [-1e6, -2e6, 1e3, 3e6, 2e6, 1e4],
        [1e4, 2e4, 1e3, 5e6, 1e6, 2e3],
        [0.0, 0.0, 0.0, 3e6, 0.0, 0.0],
    )
    @unpack
    def test_integration_test(self, ix, iy, iz, zx, zy, zz):
        """Integration test for various expected CR Hamiltonian to reconstruct."""
        backend = CRHamiltonianTomographyBackend(ix, iy, iz, zx, zy, zz)

        durations = np.linspace(0, 2000, 50) + 4 * 20
        exp = CRHamiltonianTomography(qubits=(0, 1), durations=durations)
        exp.set_experiment_options(sigma=20)
        exp.set_run_options(shots=100000)
        exp_data = exp.run(backend=backend)

        result = exp_data.analysis_result(0)
        self.assertTrue(result["success"])

        # check expected value in 1 percent precision or absolute error of 100 Hz
        self.assertLess(np.abs(result["IX"] - ix), max(100, 0.01 * np.abs(ix)))
        self.assertLess(np.abs(result["IY"] - iy), max(100, 0.01 * np.abs(iy)))
        self.assertLess(np.abs(result["IZ"] - iz), max(100, 0.01 * np.abs(iz)))
        self.assertLess(np.abs(result["ZX"] - zx), max(100, 0.01 * np.abs(zx)))
        self.assertLess(np.abs(result["ZY"] - zy), max(100, 0.01 * np.abs(zy)))
        self.assertLess(np.abs(result["ZZ"] - zz), max(100, 0.01 * np.abs(zz)))
