import numpy as np

from qiskit.utils import apply_prefix
from qiskit.providers import BaseBackend
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result
from qiskit.test import QiskitTestCase
from qiskit_experiments.composite import ParallelExperiment
from qiskit_experiments.characterization import T2StarExperiment


# Fix seed for simulations
SEED = 9000


class T2starBackend(BaseBackend):
    """
    A simple and primitive backend, to be run by the T2Star tests
    """

    def __init__(
        self, p0=None, initial_prob_plus=None, readout0to1=None, readout1to0=None, dt_factor=1
    ):
        """
        Initialize the T2star backend
        """

        configuration = QasmBackendConfiguration(
            backend_name="t2star_simulator",
            backend_version="0",
            n_qubits=int(1e6),
            basis_gates=["barrier", "h", "p", "delay", "measure"],
            gates=[],
            local=True,
            simulator=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=int(1e6),
            coupling_map=None,
            dt=dt_factor,
        )

        self._t2star = p0["t2star"]
        self._a_guess = p0["a_guess"]
        self._f_guess = p0["f_guess"]
        self._phi_guess = p0["phi_guess"]
        self._b_guess = p0["b_guess"]
        self._initial_prob_plus = initial_prob_plus
        self._readout0to1 = readout0to1
        self._readout1to0 = readout1to0
        self._dt_factor = dt_factor
        super().__init__(configuration)

    # pylint: disable = arguments-differ
    def run(self, qobj):
        """
        Run the T2star backend
        """
        shots = qobj.config.shots
        result = {
            "backend_name": "T2star backend",
            "backend_version": "0",
            "qobj_id": 0,
            "job_id": 0,
            "success": True,
            "results": [],
        }

        for circ in qobj.experiments:
            nqubits = circ.config.n_qubits
            counts = dict()
            if self._readout0to1 is None:
                ro01 = np.zeros(nqubits)
            else:
                ro01 = self._readout0to1
            if self._readout1to0 is None:
                ro10 = np.zeros(nqubits)
            else:
                ro10 = self._readout1to0
            for _ in range(shots):
                if self._initial_prob_plus is None:
                    prob_plus = np.ones(nqubits)
                else:
                    prob_plus = self._initial_prob_plus.copy()

                clbits = np.zeros(circ.config.memory_slots, dtype=int)
                for op in circ.instructions:
                    qubit = op.qubits[0]

                    if op.name == "delay":
                        delay = op.params[0]
                        t2star = self._t2star[qubit] * self._dt_factor
                        freq = self._f_guess[qubit] / self._dt_factor

                        prob_plus[qubit] = (
                            self._a_guess[qubit]
                            * np.exp(-delay / t2star)
                            * np.cos(2 * np.pi * freq * delay + self._phi_guess[qubit])
                            + self._b_guess[qubit]
                        )

                    if op.name == "measure":
                        # we measure in |+> basis which is the same as measuring |0>
                        meas_res = np.random.binomial(
                            1,
                            (1 - prob_plus[qubit]) * (1 - ro10[qubit])
                            + prob_plus[qubit] * ro01[qubit],
                        )
                        clbits[op.memory[0]] = meas_res

                clstr = ""
                for clbit in clbits[::-1]:
                    clstr = clstr + str(clbit)

                if clstr in counts:
                    counts[clstr] += 1
                else:
                    counts[clstr] = 1
            result["results"].append(
                {
                    "shots": shots,
                    "success": True,
                    "header": {"metadata": circ.header.metadata},
                    "data": {"counts": counts},
                }
            )
        return Result.from_dict(result)
