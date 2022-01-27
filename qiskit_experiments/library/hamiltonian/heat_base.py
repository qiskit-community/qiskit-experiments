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
Base Class for general Hamiltonian Error Amplifying Tomography experiments.
"""

from abc import ABC
from typing import List, Tuple, Optional

from qiskit import circuit, QuantumCircuit
from qiskit.providers import Backend

from qiskit_experiments.framework import BaseExperiment, BatchExperiment, Options
from qiskit_experiments.curve_analysis import ParameterRepr
from .heat_analysis import HeatElementAnalysis, HeatAnalysis


class HeatElement(BaseExperiment):
    """Base class of HEAT experiment elements.

    # section: overview

        Hamiltonian error amplifying tomography (HEAT) is designed to amplify
        the dynamics of entangler circuit on the target qubit along a specific axis.

        The basic form of HEAT circuit is represented as follows.

        .. parsed-literal::

                                    (xN)
                 ┌───────┐ ░ ┌───────┐┌───────┐ ░ ┌───────┐
            q_0: ┤0      ├─░─┤0      ├┤0      ├─░─┤0      ├───
                 │  prep │ ░ │  heat ││  echo │ ░ │  meas │┌─┐
            q_1: ┤1      ├─░─┤1      ├┤1      ├─░─┤1      ├┤M├
                 └───────┘ ░ └───────┘└───────┘ ░ └───────┘└╥┘
            c: 1/═══════════════════════════════════════════╩═
                                                            0

        The circuit in the middle is repeated ``N`` times to amplify the Hamiltonian
        coefficients along a specific axis of the target qubit. The ``prep`` circuit is
        carefully chosen based on the generator of the ``heat`` gate under consideration.
        The ``echo`` and ``meas`` circuits depend on the axis of the error to amplify.
        Only the target qubit is measured following to the projection in the ``meas`` circuit.

        The amplified response may consist of the contribution of from the local and
        controlled rotation terms. Thus, usually multiple error amplification experiments
        with different control qubit states are combined to distinguish the terms in the analysis.

        The ``heat`` gate is a special gate kind to represent
        the entangler pulse sequence of interest, thus one must provide the definition of it
        through the backend or custom transpiler configuration, i.e. instruction schedule map.
        This gate name can be overridden via the experiment option of this experiment.

    # section: note

        This class is usually not exposed to end users.
        Developer of new HEAT experiment must design amplification sequence and
        instantiate the class implicitly in the batch experiment.
        The :class:`BatchHeatHelper` provides a convenient wrapper class of
        the :class:`qiskit_experiments.framework.BatchExperiment` for implementing a
        typical HEAT experiment.

    # section: analysis_ref
        :py:class:`HeatElementAnalysis`

    # section: reference
        .. ref_arxiv:: 1 2007.02925
    """

    def __init__(
        self,
        qubits: Tuple[int, int],
        prep_circ: QuantumCircuit,
        echo_circ: QuantumCircuit,
        meas_circ: QuantumCircuit,
        backend: Optional[Backend] = None,
        parameter_name: Optional[str] = "d_theta",
        **kwargs,
    ):
        """Create new HEAT sub experiment.

        Args:
            qubits: Index of control and target qubit, respectively.
            prep_circ: A circuit to prepare qubit before the echo sequence.
            echo_circ: A circuit to selectively amplify the specific error term.
            meas_circ: A circuit to project target qubit onto the basis of interest.
            backend: Optional, the backend to run the experiment on.
            parameter_name: A name that represents angle from fitting.

        Keyword Args:
            See :meth:`experiment_options` for details.
        """
        analysis = HeatElementAnalysis()
        analysis.set_options(result_parameters=[ParameterRepr("d_theta", parameter_name, "rad")])

        super().__init__(qubits=qubits, backend=backend, analysis=analysis)
        self.set_experiment_options(**kwargs)

        # These are not user configurable options. Be frozen once assigned.
        self._prep_circuit = prep_circ
        self._echo_circuit = echo_circ
        self._meas_circuit = meas_circ

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            repetitions (Sequence[int]): A list of the number of echo repetitions.
            cr_gate (Gate): A gate instance representing the entangler sequence.
        """
        options = super()._default_experiment_options()
        options.repetitions = list(range(21))
        options.heat_gate = circuit.Gate("heat", num_qubits=2, params=[])

        return options

    @classmethod
    def _default_transpile_options(cls) -> Options:
        """Default transpile options."""
        options = super()._default_transpile_options()
        options.basis_gates = ["sx", "x", "rz", "heat"]
        options.optimization_level = 1

        return options

    def circuits(self) -> List[QuantumCircuit]:
        opt = self.experiment_options

        circs = list()
        for repetition in opt.repetitions:
            circ = circuit.QuantumCircuit(2, 1)
            circ.compose(self._prep_circuit, qubits=[0, 1], inplace=True)
            circ.barrier()
            for _ in range(repetition):
                circ.append(self.experiment_options.heat_gate, [0, 1])
                circ.compose(self._echo_circuit, qubits=[0, 1], inplace=True)
                circ.barrier()
            circ.compose(self._meas_circuit, qubits=[0, 1], inplace=True)
            circ.measure(1, 0)

            # add metadata
            circ.metadata = {
                "experiment_type": self.experiment_type,
                "qubits": self.physical_qubits,
                "xval": repetition,
            }

            circs.append(circ)

        return circs


class BatchHeatHelper(BatchExperiment, ABC):
    """A wrapper class of ``BatchExperiment`` to implement HEAT experiment.

    # section: overview

        This is a helper class for experiment developers of the HEAT experiment.
        This class overrides :meth:`set_experiment_options` and :meth:`set_transpile_options`
        methods of :class:`BatchExperiment` so that it can override options of
        subsequence amplification experiments to run them on the same set up.
        From end users, this experiment seems as if a single HEAT experiment.

    # section: analysis_ref
        :py:class:`HeatAnalysis`
    """

    def __init__(
        self,
        heat_experiments: List[HeatElement],
        heat_analysis: HeatAnalysis,
        backend: Optional[Backend] = None,
    ):
        """Create new HEAT experiment.

        Args:
            heat_experiments: A list of error amplification sequence that might be
                implemented as :class:``HeatElement`` instance.
            heat_analysis: HEAT analysis instance.
            backend: Optional, the backend to run the experiment on.
        """
        super().__init__(experiments=heat_experiments, backend=backend)

        # override analysis. we expect the instance is initialized with
        # parameter names specific to child amplification experiments.
        self.analysis = heat_analysis

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            repetitions (Sequence[int]): A list of the number of echo repetitions.
            cr_gate (Gate): A gate instance representing the entangler sequence.
        """
        options = super()._default_experiment_options()
        options.repetitions = list(range(21))
        options.heat_gate = circuit.Gate("heat", num_qubits=2, params=[])

        return options

    def set_experiment_options(self, **fields):
        """Set the analysis options for :meth:`run` method.

        Args:
            fields: The fields to update the options
        """
        # propagate options through all nested amplification experiments.
        for comp_exp in self.component_experiment():
            comp_exp.set_experiment_options(**fields)

        super().set_experiment_options(**fields)

    @classmethod
    def _default_transpile_options(cls) -> Options:
        """Default transpile options."""
        options = super()._default_transpile_options()
        options.basis_gates = ["sx", "x", "rz", "heat"]
        options.optimization_level = 1

        return options

    def set_transpile_options(self, **fields):
        """Set the transpiler options for :meth:`run` method.

        Args:
            fields: The fields to update the options
        """
        # propagate options through all nested amplification experiments.
        for comp_exp in self.component_experiment():
            comp_exp.set_transpile_options(**fields)

        super().set_transpile_options(**fields)
