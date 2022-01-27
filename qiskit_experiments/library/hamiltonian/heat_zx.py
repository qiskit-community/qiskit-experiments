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
HEAT experiments for ZX Hamiltonian.
"""

from typing import Tuple, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.providers import Backend

from .heat_base import BatchHeatHelper, HeatElement
from .heat_analysis import HeatAnalysis


class ZXHeat(BatchHeatHelper):
    """HEAT experiment for the ZX-type entangler.

    # section: overview
        This experiment is designed to amplify the error contained in the
        ZX-type generator, a typical Hamiltonian implemented
        by a cross-resonance drive, which is typically used to create a CNOT gate.

        The echo circuit refocuses the ZX rotation to the identity (II) and then applies
        a pi-pulse along the interrogated error axis. X errors and Y errors are
        amplified outward the X-Y plane to draw a ping-pong pattern with
        state flip by the pi-pulse echo, while Z error is amplified outward
        the X-Z plane in the same manner.
        Measurement is projected onto Y-axis in this setup.
        Because the echoed axis anti-commute with other Pauli terms,
        errors in other axes are cancelled out to reduce rotation in the interrogated axis.
        This enables to selectively amplify the Hamiltonian dynamics in the specific axis.
        Note that we have always nonzero X rotation imparted by the significant ZX term,
        the error along Y and Z axis are skewed by the nonzero commutator term.
        This yields slight mismatch in the estimated coefficients with the generator Hamiltonian,
        however this matters less when the expected magnitude of the error is small.
        On the other hand, the error in the X axis is straightforward
        because this is commute with the ZX term of the generator.

        .. parsed-literal::
                             (xN)
                             ░ ┌───────┐                            ░
            q_0: ────────────░─┤0      ├────────────────────────────░─────────
                 ┌─────────┐ ░ │  heat │┌────────────────┐┌───────┐ ░ ┌───┐┌─┐
            q_1: ┤ Rα(π/2) ├─░─┤1      ├┤ Rx(-1.0*angle) ├┤ Rβ(π) ├─░─┤ γ ├┤M├
                 └─────────┘ ░ └───────┘└────────────────┘└───────┘ ░ └───┘└╥┘
            c: 1/═══════════════════════════════════════════════════════════╩═
                                                                            0

                             (xN)
                    ┌───┐    ░ ┌───────┐                       ░
            q_0: ───┤ X ├────░─┤0      ├───────────────────────░─────────
                 ┌──┴───┴──┐ ░ │  heat │┌───────────┐┌───────┐ ░ ┌───┐┌─┐
            q_1: ┤ Rα(π/2) ├─░─┤1      ├┤ Rx(angle) ├┤ Rβ(π) ├─░─┤ γ ├┤M├
                 └─────────┘ ░ └───────┘└───────────┘└───────┘ ░ └───┘└╥┘
            c: 1/══════════════════════════════════════════════════════╩═
                                                                       0

        ZX-HEAT experiments are performed with combination of two
        error amplification experiments shown above, where :math:`\\alpha, \\beta, \\gamma`
        depend on the interrogated error axis, namely,
        (``X``, ``X``, ``I``), (``Y``, ``Y``, ``I``), (``Y``, ``Z``, ``Rx(-pi/2)``)
        for amplifying X, Y, Z axis, respectively.
        The circuit in middle is repeated by ``N`` times for the error amplification.

    # section: note
        The ``heat`` gate is a special gate to represent the entangler pulse sequence.
        This gate is usually not provided by the backend, and thus user must provide
        the pulse definition to run this experiment.
        This pulse sequence should be pre-calibrated to roughly implement the
        ZX(angle) evolution otherwise selective amplification doesn't work properly.

    # section: see_also
        qiskit_experiments.library.hamiltonian.HeatElement

    # section: analysis_ref
        :py:class:`HeatAnalysis`

    # section: reference
        .. ref_arxiv:: 1 2007.02925
    """

    def __init__(
        self,
        qubits: Tuple[int, int],
        error_axis: str,
        backend: Optional[Backend] = None,
        angle: Optional[float] = np.pi / 2,
    ):
        """Create new HEAT experiment for the entangler of ZX generator.

        Args:
            qubits: Index of control and target qubit, respectively.
            error_axis: String representation of axis that amplifies the error,
                either one of "x", "y", "z".
            backend: Optional, the backend to run the experiment on.
            angle: Angle of controlled rotation, which defaults to pi/2.

        Raises:
            ValueError: When ``error_axis`` is not one of "x", "y", "z".
        """

        amplification_exps = []
        for control in (0, 1):
            prep = QuantumCircuit(2)
            echo = QuantumCircuit(2)
            meas = QuantumCircuit(2)

            if control:
                prep.x(0)
                echo.rx(angle, 1)
            else:
                echo.rx(-angle, 1)

            if error_axis == "x":
                prep.rx(np.pi / 2, 1)
                echo.rx(np.pi, 1)
            elif error_axis == "y":
                prep.ry(np.pi / 2, 1)
                echo.ry(np.pi, 1)
            elif error_axis == "z":
                prep.ry(np.pi / 2, 1)
                echo.rz(np.pi, 1)
                meas.rx(-np.pi / 2, 1)
            else:
                raise ValueError(f"Invalid error term {error_axis}.")

            exp = HeatElement(
                qubits=qubits,
                prep_circ=prep,
                echo_circ=echo,
                meas_circ=meas,
                backend=backend,
                parameter_name=f"d_heat_{error_axis}{control}",
            )
            amplification_exps.append(exp)

        analysis = HeatAnalysis(
            fit_params=[f"d_heat_{error_axis}0", f"d_heat_{error_axis}1"],
            out_params=[f"A_I{error_axis.upper()}", f"A_Z{error_axis.upper()}"],
        )

        super().__init__(
            heat_experiments=amplification_exps,
            heat_analysis=analysis,
            backend=backend,
        )


class ZX90HeatXError(ZXHeat):
    """HEAT experiment for X error amplification for ZX(pi/2) Hamiltonian.

    # section: see_also
        qiskit_experiments.library.hamiltonian.ZXHeat
    """

    def __init__(self, qubits: Tuple[int, int], backend: Optional[Backend] = None):
        """Create new experiment.

        qubits: Index of control and target qubit, respectively.
        backend: Optional, the backend to run the experiment on.
        """
        super().__init__(
            qubits=qubits,
            error_axis="x",
            backend=backend,
            angle=np.pi / 2,
        )


class ZX90HeatYError(ZXHeat):
    """HEAT experiment for Y error amplification for ZX(pi/2) Hamiltonian.

    # section: see_also
        qiskit_experiments.library.hamiltonian.ZXHeat
    """

    def __init__(self, qubits: Tuple[int, int], backend: Optional[Backend] = None):
        """Create new experiment.

        qubits: Index of control and target qubit, respectively.
        backend: Optional, the backend to run the experiment on.
        """
        super().__init__(
            qubits=qubits,
            error_axis="y",
            backend=backend,
            angle=np.pi / 2,
        )


class ZX90HeatZError(ZXHeat):
    """HEAT experiment for Z error amplification for ZX(pi/2) Hamiltonian.

    # section: see_also
        qiskit_experiments.library.hamiltonian.ZXHeat
    """

    def __init__(self, qubits: Tuple[int, int], backend: Optional[Backend] = None):
        """Create new experiment.

        qubits: Index of control and target qubit, respectively.
        backend: Optional, the backend to run the experiment on.
        """
        super().__init__(
            qubits=qubits,
            error_axis="z",
            backend=backend,
            angle=np.pi / 2,
        )
