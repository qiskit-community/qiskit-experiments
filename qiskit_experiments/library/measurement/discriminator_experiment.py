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
Standard discriminator experiment class.
"""

from typing import List, Optional

from qiskit_experiments.framework.base_experiment import BaseExperiment

from qiskit.circuit import QuantumCircuit
from qiskit.qobj.utils import MeasLevel
from qiskit.providers.options import Options
from qiskit.providers.backend import Backend


from .discriminator_analysis import DiscriminatorAnalysis


class Discriminator(BaseExperiment):
    """Generic discriminator experiment class"""

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options for the discriminator.

        Experiment Options:
            levels (int): The number of levels to discriminate between. Defaults to 2.
        """
        options = super()._default_experiment_options()

        options.levels = 2
        return options

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default options values for the experiment :meth:`run` method."""
        options = super()._default_run_options()

        options.meas_level = MeasLevel.KERNELED
        options.meas_return = "single"

        return options

    @classmethod
    def _default_transpile_options(cls):
        """Default transpile options.

        Transpile Options:
            basis_gates (list(str)): A list of basis gates needed for this experiment.
            One gate is needed to drive the qubit to each energy level beyond the ground
            state. The schedules for these basis gates will be provided by the instruction
            schedule map(s) at transpile time.
        """
        options = super()._default_transpile_options()
        options.basis_gates = ["x"]

        return options

    def __init__(self, qubit: int, levels: int = 2, backend: Optional[Backend] = None):
        """Initialize a standard discriminator experiment.

        Args:
            qubit: List of physical qubits to discriminate on.
            levels: The number of levels to calibrate. Defaults to 2. If a number
                greater than 2 is selected, the user must provide the instruction map
                during transpile.

        """

        super().__init__([qubit], analysis=DiscriminatorAnalysis(), backend=backend)
        self.set_experiment_options(levels=levels)
        self._levels = levels

        if levels > 2:
            extra_gates = ["".join(("x_", str(i + 1), str(i + 2))) for i in range(levels)]
            self.set_transpile_options(basis_gates=["x"] + extra_gates)

    def circuits(self, backend: Optional["Backend"] = None) -> List[QuantumCircuit]:
        """Return a list of discriminator circuits.
        Args:
            backend (Backend): Optional, a backend object.
        Returns:
            List[QuantumCircuit]: A list of :class:`QuantumCircuit`s.
        """
        circuits = []

        for label in range(self._levels):
            circ = QuantumCircuit(1)
            if label == 0:
                pass
            elif label == 1:
                circ.x(0)
            elif label > 1:
                pass  # need to implement
            circ.measure_all()

            circ.metadata = {
                "experiment_type": self._type,
                "ylabel": str(label),
                "qubit": self.physical_qubits[0],
                "meas_level": self.run_options.meas_level,
                "meas_return": self.run_options.meas_return,
            }
            circuits.append(circ)

        return circuits
