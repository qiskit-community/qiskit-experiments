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
Test T2Ramsey experiment
"""
from test.base import QiskitExperimentsTestCase
from typing import List, Union, Optional
import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.providers.backend import Backend
from qiskit.providers.aer.noise.passes import RelaxationNoisePass
from qiskit.circuit import Delay

from qiskit_experiments.framework import ParallelExperiment
from qiskit_experiments.library import T2Ramsey
from qiskit_experiments.library.characterization import T2RamseyAnalysis
from qiskit_experiments.test.t2ramsey_backend import T2RamseyBackend


class T2RamseyTestExp(T2Ramsey):
    """T2Ramsey Experiment class for testing"""

    def __init__(
        self,
        qubit: int,
        delays: Union[List[float], np.array],
        t2: float,
        t1: float = None,
        dt: float = 1e-9,
        backend: Optional[Backend] = None,
        osc_freq: float = 0.0,
    ):
        """

        Args:
            qubit: The qubit the experiment on.
            delays: List of delays.
            t1: T1 parameter for the noise.
            t2: T2 parameter for the noise
            dt: Time interval for the backend.
            backend: The backend the experiment run on.
            osc_freq: The frequency of the qubit.
        """
        super().__init__(qubit, delays, backend, osc_freq)
        self._t2 = t2
        self._t1 = t1 or np.inf

        if backend and hasattr(backend.configuration(), "dt"):
            self._dt_unit = True
            self._dt_factor = backend.configuration().dt
        else:
            self._dt_unit = False
            self._dt_factor = dt

        self._op_types = [Delay]

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits.

        Each circuit consists of a Hadamard gate, followed by a fixed delay,
        a phase gate (with a linear phase), and an additional Hadamard gate.

        Returns:
            The experiment circuits
        """

        circuits = super().circuits()
        delay_pass = RelaxationNoisePass([self._t1], [self._t2], dt=1e-9, op_types=self._op_types)
        noisy_circuits = []
        for circuit in circuits:
            noisy_circuits.append(delay_pass(circuit))

        return noisy_circuits


class TestT2Ramsey(QiskitExperimentsTestCase):
    """Test T2Ramsey experiment"""

    __tolerance__ = 0.1

    def test_t2ramsey_run_end2end(self):
        """
        Run the T2Ramsey backend
        """
        osc_freq = 0.1
        estimated_t2ramsey = 20

        # induce error
        estimated_freq = osc_freq * 1.001

        # Set up the circuits
        qubit = 0
        delays = np.append(
            (np.linspace(1.0, 15.0, num=15)).astype(float),
            (np.linspace(16.0, 45.0, num=59)).astype(float),
        )

        exp = T2RamseyTestExp(qubit, delays, estimated_t2ramsey, osc_freq=osc_freq)
        default_p0 = {
            "amp": 0.5,
            "tau": estimated_t2ramsey,
            "freq": estimated_freq,
            "phi": 0,
            "base": 0.5,
        }

        backend = AerSimulator()

        for user_p0 in [default_p0, dict()]:
            exp.analysis.set_options(p0=user_p0)
            expdata = exp.run(backend=backend, shots=2000, seed_simulator=1).block_for_results()
            self.assertExperimentDone(expdata)
            self.assertRoundTripSerializable(expdata, check_func=self.experiment_data_equiv)
            self.assertRoundTripPickle(expdata, check_func=self.experiment_data_equiv)

            result = expdata.analysis_results("T2star")
            self.assertAlmostEqual(
                result.value.n,
                estimated_t2ramsey,
                delta=TestT2Ramsey.__tolerance__ * result.value.n,
            )
            self.assertEqual(result.quality, "good", "Result quality bad")
            result = expdata.analysis_results("Frequency")
            self.assertAlmostEqual(
                result.value.n,
                estimated_freq,
                delta=TestT2Ramsey.__tolerance__ * result.value.n,
            )
            self.assertEqual(result.quality, "good", "Result quality bad")

    def test_t2ramsey_parallel(self):
        """
        Test parallel experiments of T2Ramsey using a simulator.
        """
        t2ramsey = [30.0, 25.0]
        estimated_freq = [0.1, 0.12]
        delays = [list(range(1, 61)), list(range(1, 51))]

        osc_freq = [0.11, 0.11]

        exp0 = T2RamseyTestExp(0, delays[0], t2ramsey[0], osc_freq=osc_freq[0])
        exp2 = T2RamseyTestExp(2, delays[1], t2ramsey[1], osc_freq=osc_freq[1])
        par_exp = ParallelExperiment([exp0, exp2])

        exp0_p0 = {
            "A": 0.5,
            "T2star": t2ramsey[0],
            "f": estimated_freq[0],
            "phi": 0,
            "B": 0.5,
        }

        exp2_p0 = {
            "A": 0.5,
            "T2star": t2ramsey[1],
            "f": estimated_freq[1],
            "phi": 0,
            "B": 0.5,
        }

        exp0.analysis.set_options(p0=exp0_p0)
        exp2.analysis.set_options(p0=exp2_p0)

        backend = AerSimulator()
        expdata = par_exp.run(backend=backend, shots=2000, seed_simulator=1).block_for_results()
        self.assertExperimentDone(expdata)

        for i in range(2):
            res_t2star = expdata.child_data(i).analysis_results("T2star")
            self.assertAlmostEqual(
                res_t2star.value.n,
                t2ramsey[i],
                delta=TestT2Ramsey.__tolerance__ * res_t2star.value.n,
            )
            self.assertEqual(
                res_t2star.quality, "good", "Result quality bad for experiment on qubit " + str(i)
            )
            res_freq = expdata.child_data(i).analysis_results("Frequency")
            self.assertAlmostEqual(
                res_freq.value.n,
                estimated_freq[i],
                delta=TestT2Ramsey.__tolerance__ * res_freq.value.n,
            )
            self.assertEqual(
                res_freq.quality, "good", "Result quality bad for experiment on qubit " + str(i)
            )

    def _test_t2ramsey_concat_2_experiments(self):
        """
        Concatenate the data from 2 separate experiments
        """
        estimated_t2ramsey = 30
        estimated_freq = 0.09
        # First experiment
        qubit = 0
        delays0 = list(range(1, 60, 2))
        osc_freq = 0.08

        exp0 = T2Ramsey(qubit, delays0, osc_freq=osc_freq)
        default_p0 = {
            "A": 0.5,
            "T2star": estimated_t2ramsey,
            "f": estimated_freq,
            "phi": 0,
            "B": 0.5,
        }
        exp0.analysis.set_options(p0=default_p0)
        backend = T2RamseyBackend(
            p0={
                "A": [0.5],
                "T2star": [estimated_t2ramsey],
                "f": [estimated_freq],
                "phi": [0.0],
                "B": [0.5],
            },
            initial_prob_plus=[0.0],
            readout0to1=[0.02],
            readout1to0=[0.02],
        )

        # run circuits
        expdata0 = exp0.run(backend=backend, shots=1000, seed_simulator=1)
        self.assertExperimentDone(expdata0)
        res_t2star_0 = expdata0.analysis_results("T2star")

        # second experiment
        delays1 = list(range(2, 65, 2))
        exp1 = T2Ramsey(qubit, delays1)
        exp1.analysis.set_options(p0=default_p0)
        expdata1 = exp1.run(backend=backend, analysis=None, shots=1000, seed_simulator=1)
        self.assertExperimentDone(expdata1)
        expdata1.add_data(expdata0.data())
        exp1.analysis.run(expdata1)

        res_t2star_1 = expdata1.analysis_results("T2star")
        res_freq_1 = expdata1.analysis_results("Frequency")

        self.assertAlmostEqual(
            res_t2star_1.value.n,
            estimated_t2ramsey,
            delta=TestT2Ramsey.__tolerance__ * res_t2star_1.value.n,
        )
        self.assertAlmostEqual(
            res_freq_1.value.n,
            estimated_freq,
            delta=TestT2Ramsey.__tolerance__ * res_freq_1.value.n,
        )
        self.assertLessEqual(res_t2star_1.value.s, res_t2star_0.value.s)
        self.assertEqual(len(expdata1.data()), len(delays0) + len(delays1))

    def _test_experiment_config(self):
        """Test converting to and from config works"""
        exp = T2Ramsey(0, [1, 2, 3, 4, 5])
        loaded_exp = T2Ramsey.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def _test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = T2Ramsey(0, [1, 2, 3, 4, 5])
        self.assertRoundTripSerializable(exp, self.json_equiv)

    def _test_analysis_config(self):
        """ "Test converting analysis to and from config works"""
        analysis = T2RamseyAnalysis()
        loaded = T2RamseyAnalysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())
