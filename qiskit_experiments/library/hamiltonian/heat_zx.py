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
    r"""HEAT experiment for the ZX-type entangler.

    # section: overview
        This experiment is designed to amplify the error contained in the
        ZX-type generator, a typical Hamiltonian implemented
        by a cross-resonance drive, which is typically used to create a CNOT gate.

        The experimental circuits are prepared as follows for different
        interrogated error axis specified by the experiment parameter ``error_axis``.

        .. parsed-literal::

            　　　　 　prep        heat         　　 echo              　 meas

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
        error amplification experiments shown above, where :math:`\alpha, \beta, \gamma`
        depend on the interrogated error axis, namely,
        (``X``, ``X``, ``I``), (``Y``, ``Y``, ``I``), (``Y``, ``Z``, ``Rx(-pi/2)``)
        for amplifying X, Y, Z axis, respectively.
        The circuit in the middle is repeated by N times for the error amplification.

        For example, we amplify the X error in the simplified ``heat`` gate Hamiltonian

        .. math::

            Ht = \frac{\Omega_{ZX}(t) ZX + \Delta_{IX}(t) IX}{2}.

        From the BCH formula we can derive a unitary evolution of the Hamiltonian

        .. math::

            U = A_{II} II + A_{IX} IX + A_{ZX} ZX + A_{ZI} ZI.

        Since we have known control qubit state throughout the echo sequence,
        we can compute partial unitary on the target qubit, namely,
        :math:`U_{j} = A_{Ij} I + A_{Xj} X` for the control qubit state :math:`|j\rangle`.
        Here :math:`A_{Ij} =\cos \theta_j` and :math:`A_{Xj} =-i \sin \theta_j`.
        This form is exactly identical to the unitary of :math:`R_X(\theta_j)` gate,
        with :math:`\theta_0 =\Delta_{IX} + \Omega_{ZX}` and
        :math:`\theta_1 =\Delta_{IX} - \Omega_{ZX}`.
        Given we calibrated the gate to have :math:`\Omega_{ZX} = \phi + \Delta_{ZX}`
        so that :math:`\phi` corresponds to the experiment parameter ``angle``,
        or the angle of the controlled rotation we want,
        e.g. :math:`\phi = \pi/2` for the CNOT gate.
        The total evolution during the echo sequence will be expressed by
        :math:`R_X(\pi + \Delta_{ZX} \pm \Delta_{IX})` for the control qubit state
        0 and 1, respectively.

        In the echo circuit, the non-local ZX rotation by :math:`\phi` is undone by
        applying :math:`R_X(\mp \phi)` with sign depending on the control qubit state,
        thus only rotation error :math:`\Delta_{ZX}` from the target
        angle :math:`\phi` is selectively amplified.
        Repeating this sequence N times forms a typical ping-pong oscillation pattern
        in the measured target qubit population,
        which may be fit by :math:`P(N) = \cos(N (d\theta_j + \pi) + \phi_{\rm offset})`,
        where :math:`d\theta_j = \Delta_{ZX}\pm \Delta_{IX}`.
        By combining error amplification fit parameter :math:`d\theta_j` for
        different control qubit states, we can resolve local (IX) and non-local (ZX)
        dynamics of the Hamiltonian of interest.

        In this pulse sequence, the pi-pulse echo is applied to the target qubit
        around the same axis as the interrogated error.
        This cancels out the errors in other axes since the errors anti-commute with the echo,
        e.g. :math:`XYX = -Y`, while the error in the interrogated axis is accumulated.
        This is the trick of how the sequence selectively amplifies the error axis.

        However, strictly speaking, non-X error terms :math:`{\cal P}` also anti-commute
        with the primary :math:`ZX` term of the Hamiltonian, and
        they are skewed by the significant nonzero commutator :math:`[ZX, {\cal P}]`.
        Thus this sequence pattern might underestimate the coefficients in non-X axes.
        Usually this is less impactful if the errors of interest are sufficiently small,
        but you should keep this in mind.

    # section: example
        This experiment requires you to provide the pulse definition of the ``heat`` gate.
        This gate should implement the ZX Hamiltonian with rotation angle :math:`\phi`.
        This might be done in the following workflow.

        .. code-block:: python

            from qiskit import pulse
            from qiskit.test.mock import FakeJakarta
            from qiskit_experiments.library import ZXHeat

            backend = FakeJakarta()
            qubits = 0, 1

            # Write pulse schedule implementing ZX Hamiltonian
            heat_pulse = pulse.GaussianSquare(100, 1, 10, 5)

            with pulse.build(backend) as heat_sched:
                pulse.play(heat_pulse, pulse.control_channels(*qubits)[0])

            # Map schedule to the gate
            my_inst_map = backend.defaults().instruction_schedule_map
            my_inst_map.add("heat", qubits, heat_sched)

            # Set up experiment
            heat_exp = ZXHeat(qubits, error_axis="x", backend=backend)
            heat_exp.set_transpile_options(inst_map=my_inst_map)
            heat_exp.run()

    # section: note
        The ``heat`` gate represents the entangling pulse sequence.
        This gate is usually not provided by the backend, and users must thus provide
        the pulse schedule to run this experiment.
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
