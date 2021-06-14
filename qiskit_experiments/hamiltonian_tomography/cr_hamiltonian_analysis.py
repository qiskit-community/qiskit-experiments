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
Cross resonance Hamiltonian tomography analysis.
"""

from typing import List, Dict, Any, Union

import numpy as np

from qiskit_experiments.analysis import CurveAnalysis, SeriesDef, CurveAnalysisResult
from qiskit_experiments.analysis.data_processing import expectation_value
from qiskit_experiments.analysis.utils import get_opt_value, get_opt_error, frequency_guess
from qiskit_experiments.exceptions import AnalysisError


def oscillation_x(x: np.ndarray, px: float, py: float, pz: float, b: float):
    """Fit function for x basis oscillation."""
    omega = np.sqrt(px**2 + py**2 + pz**2)
    return (-pz * px + pz * px * np.cos(omega * x) + omega * py * np.sin(omega * x)) / omega**2 + b


def oscillation_y(x: np.ndarray, px: float, py: float, pz: float, b: float):
    """Fit function for y basis oscillation."""
    omega = np.sqrt(px**2 + py**2 + pz**2)
    return (pz * py - pz * py * np.cos(omega * x) - omega * px * np.sin(omega * x)) / omega**2 + b


def oscillation_z(x: np.ndarray, px: float, py: float, pz: float, b: float):
    """Fit function for z basis oscillation."""
    omega = np.sqrt(px**2 + py**2 + pz**2)
    return (pz**2 + (px**2 + py**2) * np.cos(omega * x)) / omega**2 + b


class CRHamiltonianAnalysis(CurveAnalysis):
    r"""A class to analyze spectroscopy experiment.

    Overview
        This analysis takes multiple data series. These are fit by
        composite trigonometric functions, which are derived from the dynamics of
        block-diagonalized cross resonance Hamiltonian.

    Fit Model
        Following equations are used to approximate the dynamics of target qubit Bloch vector.

        .. math::

            F_{x, c}(t) &= \frac{1}{\Omega_c^2} \left(
                - p_{z, c} p_{x, c} + p_{z, c} p_{x, c} \cos(\Omega_c t) +
                \Omega_c p_{y, c} \sin(\Omega_c t) \right) + b \ ... \ (1), \\
            F_{y, c}(t) &= \frac{1}{\Omega_c^2} \left(
                p_{z, c} p_{y, c} - p_{z, c} p_{y, c} \cos(\Omega_c t) -
                \Omega_c p_{x, c} \sin(\Omega_c t) \right) + b \ ... \ (2), \\
            F_{z, c}(t) &= \frac{1}{\Omega_c^2} \left(
             p_{z, c}^2 + (p_{x, c}^2 + p_{y, c}^2) \cos(\Omega_c t) \right) + b \ ... \ (3),

        where :math:`|\Omega_c = \sqrt{p_{x, c}^2+p_{y, c}^2+p_{z, c}^2}` and
        :math:`p_{x, c}, p_{y, c}, p_{z, c}, b` are the fit parameters.
        The subscript :math:`c` represents the state of control qubit :math:`c \in \{0, 1\}`.
        The fit functions :math:`F_{x, c}, F_{y, c}, F_{z, c}` approximate the Pauli expectation
        value of target qubit :math:`\langle \sigma_{x, c} (t) \rangle,
        \langle \sigma_{y, c} (t) \rangle, \langle \sigma_{z, c} (t) \rangle`, respectively.

    Fit Parameters
        - :math:`p_{x,0}`: Fit parameter for curve with control state :math:`c=0`.
        - :math:`p_{y,0}`: Fit parameter for curve with control state :math:`c=0`.
        - :math:`p_{z,0}`: Fit parameter for curve with control state :math:`c=0`.
        - :math:`p_{x,1}`: Fit parameter for curve with control state :math:`c=1`.
        - :math:`p_{y,1}`: Fit parameter for curve with control state :math:`c=1`.
        - :math:`p_{z,1}`: Fit parameter for curve with control state :math:`c=1`.
        - :math:`b`: Base line.

        The CR Hamiltonian coefficients can be written as

        .. math::

            ZX &= \frac{p_{x,0} - p_{x,1}}{2} \\
            ZY &= \frac{p_{y,0} - p_{y,1}}{2} \\
            ZZ &= \frac{p_{z,0} - p_{z,1}}{2} \\
            IX &= \frac{p_{x,0} + p_{x,1}}{2} \\
            IY &= \frac{p_{y,0} + p_{y,1}}{2} \\
            IZ &= \frac{p_{z,0} + p_{z,1}}{2}

    Initial Guesses
        Following protocol is used to obtain initial parameters, regardless of :math:`c`.

        From (1), :math:`F_{z, 0} = \frac{1}{\Omega^2} (p_z^2 + (p_x^2 + p_y^2))` when
        :math:`\Omega t = 0` and :math:`F_{z, \pi} = \frac{1}{\Omega^2} (p_z^2 - (p_x^2 + p_y^2))`
        when :math:`\Omega t = \pi`. They appear as the maximum and minimum value of the
        :math:`\langle \sigma_{z} (t) \rangle` oscillation.
        These points are estimated as 90 and 10 percentile of measured value to avoid outliers.
        With evaluated min and max values,
        :math:`p_z = \Omega \sqrt{\frac{F_{z, 0} + F_{z, \pi}}{2}}`.
        Here :math:`\Omega` is directly estimated from :math:`\langle \sigma_{z} (t) \rangle`
        by using the fast Fourier analysis.

        Next, we estimate `p_x` and `p_y` with following relationship.

        .. math::

            F_{x} + F_{y} &= \frac{p_x - p_y}{\Omega^2} \left(
                p_z (\cos(\Omega t) - 1) - \Omega \sin(\Omega t) \right), \\
            F_{x} - F_{y} &= \frac{p_x + p_y}{\Omega^2} \left(
                p_z (\cos(\Omega t) - 1) + \Omega \sin(\Omega t) \right).

        At :math:`\Omega t = \pi/2`,

        .. math::

            v_+ &= \frac{\Omega^2}{\Omega + p_z} (F_{x} + F_{y}) = p_y - p_x, \\
            v_- &= \frac{\Omega^2}{\Omega - p_z} (F_{x} - F_{y}) = p_y + p_x.

        This yields :math:`p_x = \frac{v_- - v_+}{2}` and :math:`p_y = \frac{v_- + v_+}{2}`.
        Here `p_z` is the estimated initial value.
        $b = 0$ is assumed regardless of measured data.

    Bounds
        - No boundary for :math:`p_{x, c}, p_{y, c}, p_{z, c}`.
        - :math:`b`: [-1, 1]

    """

    __series__ = [
        SeriesDef(
            name="x|c=0",
            fit_func=lambda x, px0, px1, py0, py1, pz0, pz1, b: oscillation_x(
                x, px=px0, py=py0, pz=pz0, b=b
            ),
            filter_kwargs={"control_state": 0, "meas_basis": "x"},
            plot_color="blue",
            plot_symbol="o",
            canvas=0,
        ),
        SeriesDef(
            name="y|c=0",
            fit_func=lambda x, px0, px1, py0, py1, pz0, pz1, b: oscillation_y(
                x, px=px0, py=py0, pz=pz0, b=b
            ),
            filter_kwargs={"control_state": 0, "meas_basis": "y"},
            plot_color="blue",
            plot_symbol="o",
            canvas=1,
        ),
        SeriesDef(
            name="z|c=0",
            fit_func=lambda x, px0, px1, py0, py1, pz0, pz1, b: oscillation_z(
                x, px=px0, py=py0, pz=pz0, b=b
            ),
            filter_kwargs={"control_state": 0, "meas_basis": "z"},
            plot_color="blue",
            plot_symbol="o",
            canvas=2,
        ),
        SeriesDef(
            name="x|c=1",
            fit_func=lambda x, px0, px1, py0, py1, pz0, pz1, b: oscillation_x(
                x, px=px1, py=py1, pz=pz1, b=b
            ),
            filter_kwargs={"control_state": 1, "meas_basis": "x"},
            plot_color="red",
            plot_symbol="^",
            canvas=0,
        ),
        SeriesDef(
            name="y|c=1",
            fit_func=lambda x, px0, px1, py0, py1, pz0, pz1, b: oscillation_y(
                x, px=px1, py=py1, pz=pz1, b=b
            ),
            filter_kwargs={"control_state": 1, "meas_basis": "y"},
            plot_color="red",
            plot_symbol="^",
            canvas=1,
        ),
        SeriesDef(
            name="z|c=1",
            fit_func=lambda x, px0, px1, py0, py1, pz0, pz1, b: oscillation_z(
                x, px=px1, py=py1, pz=pz1, b=b
            ),
            filter_kwargs={"control_state": 1, "meas_basis": "z"},
            plot_color="red",
            plot_symbol="^",
            canvas=2,
        ),
    ]

    @classmethod
    def _default_options(cls):
        """Return default data processing options.

        See :meth:`~qiskit_experiment.analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """
        default_options = super()._default_options()
        default_options.data_processor = expectation_value()
        default_options.fig_size = (8, 10)
        default_options.xlabel = "CR duration (sec)"
        default_options.ylabel = r"$\langle\sigma_{X, Y, Z}\rangle$"
        default_options.p0 = {
            "px0": None, "px1": None, "py0": None, "py1": None, "pz0": None, "pz1": None, "b": None
        }
        default_options.bounds = {
            "px0": None, "px1": None, "py0": None, "py1": None, "pz0": None, "pz1": None, "b": None
        }
        default_options.fit_reports = {
            "IX": r"$\omega_{IX}$",
            "IY": r"$\omega_{IY}$",
            "IZ": r"$\omega_{IZ}$",
            "ZX": r"$\omega_{ZX}$",
            "ZY": r"$\omega_{ZY}$",
            "ZZ": r"$\omega_{ZZ}$",
        }

        return default_options

    def _setup_fitting(self, **options) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Fitter options.

        See class docstring for

        Raises:
            AnalysisError
                - When time range is shorter than one cycle.
        """
        user_p0 = self._get_option("p0")
        user_bounds = self._get_option("bounds")

        init_guesses = []
        for tshift in np.arange(-3, 3):
            # slightly offset data points to find tOmega = pi/2 in discrete data.

            init_guess = dict()
            for control in (0, 1):
                # initial guess of pz
                ts, exp_z, _ = self._subset_data(
                    name=f"z|c={control}",
                    data_index=self._data_index,
                    x_values=self._x_values,
                    y_values=self._y_values,
                    y_sigmas=self._y_sigmas,
                )
                omega = 2 * np.pi * frequency_guess(ts, exp_z, method="FFT")

                if omega == 0:
                    raise AnalysisError(
                        "Gate time scan range seems to be shorter than one cycle. "
                        "Need more longer gate time to complete the analysis."
                    )

                zrange_mid = np.mean(np.percentile(exp_z, [10, 90]))
                # take percentile to remove outlier, rather than taking min max

                if zrange_mid < 0:
                    pz_guess = 0.
                else:
                    pz_guess = omega * np.sqrt(zrange_mid)

                # initial guess of px and py
                pi2_time = np.pi / 2 / omega
                pi2_ind = np.argmin(np.abs(ts - pi2_time)) + tshift

                if pi2_ind < 0:
                    continue

                _, exp_x, _ = self._subset_data(
                    name=f"x|c={control}",
                    data_index=self._data_index,
                    x_values=self._x_values,
                    y_values=self._y_values,
                    y_sigmas=self._y_sigmas,
                )
                _, exp_y, _ = self._subset_data(
                    name=f"y|c={control}",
                    data_index=self._data_index,
                    x_values=self._x_values,
                    y_values=self._y_values,
                    y_sigmas=self._y_sigmas,
                )
                v1 = omega ** 2 / (omega + pz_guess) * (exp_x[pi2_ind] + exp_y[pi2_ind])
                v2 = omega ** 2 / (omega - pz_guess) * (exp_x[pi2_ind] - exp_y[pi2_ind])

                px_guess = (v2 - v1) / 2
                py_guess = (v2 + v1) / 2

                init_guess[f"px{control}"] = user_p0[f"px{control}"] or px_guess
                init_guess[f"py{control}"] = user_p0[f"py{control}"] or py_guess
                init_guess[f"pz{control}"] = user_p0[f"pz{control}"] or pz_guess
            init_guess["b"] = user_p0["b"] or 0.
            init_guesses.append(init_guess)

        fit_options = []
        for init_guess in init_guesses:
            fit_option = {
                "p0": init_guess,
                "bounds": {
                    "px0": user_bounds["px0"] or (-np.inf, np.inf),
                    "py0": user_bounds["py0"] or (-np.inf, np.inf),
                    "pz0": user_bounds["pz0"] or (-np.inf, np.inf),
                    "px1": user_bounds["px1"] or (-np.inf, np.inf),
                    "py1": user_bounds["py1"] or (-np.inf, np.inf),
                    "pz1": user_bounds["pz1"] or (-np.inf, np.inf),
                    "b": user_bounds["b"] or (-1, 1),
                },
            }
            fit_option.update(options)
            fit_options.append(fit_option)

        return fit_options

    def _post_processing(self, analysis_result: CurveAnalysisResult) -> CurveAnalysisResult:
        """Calculate Hamiltonian coefficients."""

        for control in ("Z", "I"):
            for target in ("X", "Y", "Z"):
                p0_val = get_opt_value(analysis_result, f"p{target.lower()}0") / (2 * np.pi)
                p1_val = get_opt_value(analysis_result, f"p{target.lower()}1") / (2 * np.pi)
                p0_err = get_opt_error(analysis_result, f"p{target.lower()}0") / (2 * np.pi)
                p1_err = get_opt_error(analysis_result, f"p{target.lower()}1") / (2 * np.pi)
                if control == "Z":
                    coef = 0.5 * (p0_val - p1_val)
                else:
                    coef = 0.5 * (p0_val + p1_val)
                analysis_result[f"{control}{target}"] = coef
                analysis_result[f"{control}{target}_err"] = np.sqrt(p0_err**2 + p1_err**2)

        return analysis_result
