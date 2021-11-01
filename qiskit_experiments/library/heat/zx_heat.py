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
HEAT experiments for ZX Hamiltonian.
"""

from typing import Tuple, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.providers import Backend

from qiskit_experiments.framework import fix_class_docs
from qiskit_experiments.curve_analysis import ParameterRepr
from .base_analysis import CompositeHeatAnalysis
from .base_experiment import BaseCompositeHeat, HeatElement


class HeatYAnalysis(CompositeHeatAnalysis):
    """"""
    __fit_params__ = ["d_heat_y0", "d_heat_y1"]
    __out_params__ = ["A_iy", "A_zy"]


@fix_class_docs
class ZXHeatYError(BaseCompositeHeat):
    r"""HEAT experiments for Y error amplification.

    # section: overview

        This experiment consists of following two error amplification experiments.

        .. parsed-literal::

                             ░ ┌───────┐      ░
            q_0: ────────────░─┤0      ├──────░────
                 ┌─────────┐ ░ │  heat │┌───┐ ░ ┌─┐
            q_1: ┤ Ry(π/2) ├─░─┤1      ├┤ Y ├─░─┤M├
                 └─────────┘ ░ └───────┘└───┘ ░ └╥┘
            c: 1/════════════════════════════════╩═
                                                 0

                    ┌───┐    ░ ┌───────┐      ░
            q_0: ───┤ X ├────░─┤0      ├──────░────
                 ┌──┴───┴──┐ ░ │  heat │┌───┐ ░ ┌─┐
            q_1: ┤ Ry(π/2) ├─░─┤1      ├┤ Y ├─░─┤M├
                 └─────────┘ ░ └───────┘└───┘ ░ └╥┘
            c: 1/════════════════════════════════╩═
                                                 0

        The circuit in the middle contains :math:`ZX(\pi/2)` gate represented by a ``heat`` gate,
        and this block is repeated N times specified by the experiment option.

        TODO more docs
    """
    __analysis_class__ = HeatYAnalysis

    def __init__(
        self,
        qubits: Tuple[int, int],
        backend: Optional[Backend] = None,
    ):

        meas = QuantumCircuit(2)

        echo = QuantumCircuit(2)
        echo.y(1)

        prep_0 = QuantumCircuit(2)
        prep_0.ry(np.pi/2, 1)

        prep_1 = QuantumCircuit(2)
        prep_1.x(0)
        prep_1.ry(np.pi/2, 1)

        heat_y0 = HeatElement(
            qubits=qubits,
            prep_circ=prep_0,
            echo_circ=echo,
            meas_circ=meas,
        )
        heat_y0.set_analysis_options(
            result_parameters=[ParameterRepr("d_theta", "d_heat_y0", "rad")]
        )

        heat_y1 = HeatElement(
            qubits=qubits,
            prep_circ=prep_1,
            echo_circ=echo,
            meas_circ=meas,
        )
        heat_y1.set_analysis_options(
            result_parameters=[ParameterRepr("d_theta", "d_heat_y1", "rad")]
        )

        super().__init__([heat_y0, heat_y1], backend=backend)


class HeatZAnalysis(CompositeHeatAnalysis):
    """"""
    __fit_params__ = ["d_heat_z0", "d_heat_z1"]
    __out_params__ = ["A_iz", "A_zz"]


@fix_class_docs
class ZXHeatZError(BaseCompositeHeat):
    r"""HEAT experiments for Z error amplification.

    # section: overview

        This experiment consists of following two error amplification experiments.

        .. parsed-literal::

                             ░ ┌───────┐      ░
            q_0: ────────────░─┤0      ├──────░───────────────
                 ┌─────────┐ ░ │  heat │┌───┐ ░ ┌─────────┐┌─┐
            q_1: ┤ Ry(π/2) ├─░─┤1      ├┤ Z ├─░─┤ Rx(π/2) ├┤M├
                 └─────────┘ ░ └───────┘└───┘ ░ └─────────┘└╥┘
            c: 1/═══════════════════════════════════════════╩═
                                                            0

                    ┌───┐    ░ ┌───────┐      ░
            q_0: ───┤ X ├────░─┤0      ├──────░───────────────
                 ┌──┴───┴──┐ ░ │  heat │┌───┐ ░ ┌─────────┐┌─┐
            q_1: ┤ Ry(π/2) ├─░─┤1      ├┤ Z ├─░─┤ Rx(π/2) ├┤M├
                 └─────────┘ ░ └───────┘└───┘ ░ └─────────┘└╥┘
            c: 1/═══════════════════════════════════════════╩═
                                                            0


        The circuit in the middle contains :math:`ZX(\pi/2)` gate represented by a ``heat`` gate,
        and this block is repeated N times specified by the experiment option.

        TODO more docs
    """
    __analysis_class__ = HeatZAnalysis

    def __init__(
        self,
        qubits: Tuple[int, int],
        backend: Optional[Backend] = None,
    ):

        meas = QuantumCircuit(2)
        meas.rx(np.pi/2, 1)

        echo = QuantumCircuit(2)
        echo.z(1)

        prep_0 = QuantumCircuit(2)
        prep_0.ry(np.pi/2, 1)

        prep_1 = QuantumCircuit(2)
        prep_1.x(0)
        prep_1.ry(np.pi/2, 1)

        heat_z0 = HeatElement(
            qubits=qubits,
            prep_circ=prep_0,
            echo_circ=echo,
            meas_circ=meas,
        )
        heat_z0.set_analysis_options(
            result_parameters=[ParameterRepr("d_theta", "d_heat_z0", "rad")]
        )

        heat_z1 = HeatElement(
            qubits=qubits,
            prep_circ=prep_1,
            echo_circ=echo,
            meas_circ=meas,
        )
        heat_z1.set_analysis_options(
            result_parameters=[ParameterRepr("d_theta", "d_heat_z1", "rad")]
        )

        super().__init__([heat_z0, heat_z1], backend=backend)
