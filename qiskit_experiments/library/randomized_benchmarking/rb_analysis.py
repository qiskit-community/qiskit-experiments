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
Standard RB analysis class.
"""
import warnings
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple, Union, Optional, TYPE_CHECKING

import numpy as np
from qiskit.exceptions import QiskitError

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis.data_processing import multi_mean_xy_data, data_sort
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import AnalysisResultData, ExperimentData
from qiskit_experiments.database_service import DbAnalysisResultV1

if TYPE_CHECKING:
    from uncertainties import UFloat

# A dictionary key of qubit aware quantum instruciton; type alias for better readability
QubitGateTuple = Tuple[Tuple[int, ...], str]


def _lookup_epg_ratio(gate: str, n_qubits: int) -> Union[None, int]:
    """A helper method to look-up preset gate error ratio for given basis gate name.

    In the table the error ratio is defined based on the count of
    typical assembly gate in the gate decomposition.
    For example, "u3" gate can be decomposed into two "sx" gates.
    In this case, the ratio of "u3" gate error becomes 2.

    .. note::

        This table is not aware of the actual waveform played on the hardware,
        and the returned error ratio is just a guess.
        To be precise, user can always set "gate_error_ratio" option of the experiment.

    Args:
        gate: Name of the gate.
        n_qubits: Number of qubits measured in the RB experiments.

    Returns:
        Corresponding error ratio.

    Raises:
        QiskitError: When number of qubit is more than three.
    """

    # Gate count in (X, SX)-based decomposition. VZ gate contribution is ignored.
    # Amplitude or duration modulated pulse implementation is not considered.
    standard_1q_ratio = {
        "u1": 0.0,
        "u2": 1.0,
        "u3": 2.0,
        "u": 2.0,
        "p": 0.0,
        "x": 1.0,
        "y": 1.0,
        "z": 0.0,
        "t": 0.0,
        "tdg": 0.0,
        "s": 0.0,
        "sdg": 0.0,
        "sx": 1.0,
        "sxdg": 1.0,
        "rx": 2.0,
        "ry": 2.0,
        "rz": 0.0,
        "id": 0.0,
        "h": 1.0,
    }

    # Gate count in (CX, CSX)-based decomposition, 1q gate contribution is ignored.
    # Amplitude or duration modulated pulse implementation is not considered.
    standard_2q_ratio = {
        "swap": 3.0,
        "rxx": 2.0,
        "rzz": 2.0,
        "cx": 1.0,
        "cy": 1.0,
        "cz": 1.0,
        "ch": 1.0,
        "crx": 2.0,
        "cry": 2.0,
        "crz": 2.0,
        "csx": 1.0,
        "cu1": 2.0,
        "cp": 2.0,
        "cu": 2.0,
        "cu3": 2.0,
    }

    if n_qubits == 1:
        return standard_1q_ratio.get(gate, None)

    if n_qubits == 2:
        return standard_2q_ratio.get(gate, None)

    raise QiskitError(
        f"Standard gate error ratio for {n_qubits} qubit RB is not provided. "
        "Please explicitly set 'gate_error_ratio' option of the experiment."
    )


def _calculate_epg(
    epc: Union[float, "UFloat"],
    qubits: Sequence[int],
    gate_error_ratio: Dict[str, float],
    gate_counts_per_clifford: Dict[QubitGateTuple, float],
) -> Dict[str, Union[float, "UFloat"]]:
    r"""A helper mehtod to compute EPGs of basis gates from fit EPC value.

    Args:
        epc: Error per Clifford.
        qubits: List of qubits used in the experiment.
        gate_error_ratio: A dictionary of assumed ratio of errors among basis gates.
        gate_counts_per_clifford: Basis gate counts per Clifford gate.

    Returns:
        A dictionary of gate errors keyed on the gate name.
    """
    norm = 0
    for gate, r_epg in gate_error_ratio.items():
        formatted_key = tuple(sorted(qubits)), gate
        norm += r_epg * gate_counts_per_clifford.get(formatted_key, 0.0)

    epgs = {}
    for gate, r_epg in gate_error_ratio.items():
        epgs[gate] = r_epg * epc / norm
    return epgs


def _exclude_1q_error(
    epc: Union[float, "UFloat"],
    qubits: Tuple[int, int],
    gate_counts_per_clifford: Dict[QubitGateTuple, float],
    extra_analyses: Optional[List[DbAnalysisResultV1]],
) -> Union[float, "UFloat"]:
    r"""A helper method to exclude contribution of single qubit gates from 2Q EPC.

    Args:
        epc: EPC from 2Q RB experiment.
        qubits: Index of two qubits used for 2Q RB experiment.
        gate_counts_per_clifford: Basis gate counts per 2Q Clifford gate.
        extra_analyses: Analysis results containing depolarizing parameters of 1Q RB experiments.

    Returns:
        Corrected 2Q EPC.
    """
    # Extract EPC of non-measured qubits from previous experiments
    epg_1qs = {}
    for analyis_data in extra_analyses:
        if (
            not analyis_data.name.startswith("EPG_")
            or len(analyis_data.device_components) > 1
            or not str(analyis_data.device_components[0]).startswith("Q")
        ):
            continue
        qind = analyis_data.device_components[0]._index
        gate = analyis_data.name[4:]
        formatted_key = (qind,), gate
        epg_1qs[formatted_key] = analyis_data.value

    if not epg_1qs:
        return epc

    # Convert 2Q EPC into depolarizing parameter alpha
    alpha_c_2q = 1 - 4 / 3 * epc

    # Estimate composite alpha of 1Q channels
    alpha_i = [1.0, 1.0]
    for q_gate_tup, epg in epg_1qs.items():
        n_gate = gate_counts_per_clifford.get(q_gate_tup, 0.0)
        aind = qubits.index(q_gate_tup[0][0])
        alpha_i[aind] *= (1 - 2 * epg) ** n_gate
    alpha_c_1q = 1 / 5 * (alpha_i[0] + alpha_i[1] + 3 * alpha_i[0] * alpha_i[1])

    # Corrected 2Q channel EPC
    return 3 / 4 * (1 - (alpha_c_2q / alpha_c_1q))


class RBAnalysis(curve.CurveAnalysis):
    r"""A class to analyze randomized benchmarking experiments.

    # section: overview
        This analysis takes only single series.
        This series is fit by the exponential decay function.
        From the fit :math:`\alpha` value this analysis estimates the error per Clifford (EPC).

        When analysis option ``gate_error_ratio`` is provided, this analysis also estimates
        errors of individual gates assembling a Clifford gate.
        In computation of two-qubit EPC, this analysis can also decompose
        the contribution from the underlying single qubit depolarizing channels when
        ``epg_1_qubit`` analysis option is provided [1].

    # section: fit_model
        .. math::

            F(x) = a \alpha^x + b

    # section: fit_parameters
        defpar a:
            desc: Height of decay curve.
            init_guess: Determined by :math:`(y - b) / \alpha^x`.
            bounds: [0, 1]
        defpar b:
            desc: Base line.
            init_guess: Determined by the average :math:`b` of the standard and interleaved RB.
                Usually equivalent to :math:`(1/2)^n` where :math:`n` is number of qubit.
            bounds: [0, 1]
        defpar \alpha:
            desc: Depolarizing parameter.
            init_guess: Determined by the slope of :math:`(y - b)^{-x}` of the first and the
                second data point.
            bounds: [0, 1]

    # section: reference
        .. ref_arxiv:: 1 1712.06550

    """

    __series__ = [
        curve.SeriesDef(
            fit_func=lambda x, a, alpha, b: curve.fit_function.exponential_decay(
                x, amp=a, lamb=-1.0, base=alpha, baseline=b
            ),
            plot_color="blue",
            model_description=r"a \alpha^x + b",
        )
    ]

    @classmethod
    def _default_options(cls):
        """Default analysis options.

        Analysis Options:
            gate_error_ratio (Optional[Dict[str, float]]): A dictionary with gate name keys
                and error ratio values used when calculating EPG from the estimated EPC.
                The default value will use standard gate error ratios.
                If set to ``False`` EPG will not be calculated.
            gate_counts_per_clifford (Union[bool, Dict[str, float]]): A dictionary
                of gate numbers constituting a single averaged Clifford operation
                on particular physical qubit. Usually this value is automatically
                computed based on the circuit metadata.
            epg_1_qubit (List[DbAnalysisResultV1]): Analysis results from previous RB experiments
                for individual single qubit gates. If this is provided, EPC of
                2Q RB is corected to exclude the deporalization of underlying 1Q channels.
        """
        default_options = super()._default_options()
        default_options.curve_plotter.set_options(
            xlabel="Clifford Length",
            ylabel="P(0)",
        )
        default_options.plot_raw_data = True
        default_options.result_parameters = ["alpha"]
        default_options.gate_error_ratio = None
        default_options.gate_counts_per_clifford = None
        default_options.epg_1_qubit = None

        return default_options

    def set_options(self, **fields):
        if "error_dict" in fields:
            warnings.warn(
                "Option 'error_dict' has been removed and merged into 'gate_error_ratio'.",
                DeprecationWarning,
            )
            fields["gate_error_ratio"] = fields.pop("error_dict")
        super().set_options(**fields)

    def _generate_fit_guesses(
        self, user_opt: curve.FitOptions
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Compute the initial guesses.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        curve_data = self._data()

        user_opt.bounds.set_if_empty(
            a=(0, 1),
            alpha=(0, 1),
            b=(0, 1),
        )

        return self._initial_guess(user_opt, curve_data.x, curve_data.y, self._num_qubits)

    @staticmethod
    def _initial_guess(
        opt: curve.FitOptions, x_values: np.ndarray, y_values: np.ndarray, num_qubits: int
    ) -> curve.FitOptions:
        """Create initial guess with experiment data."""
        opt.p0.set_if_empty(b=1 / 2**num_qubits)

        # Use the first two points to guess the decay param
        dcliff = x_values[1] - x_values[0]
        dy = (y_values[1] - opt.p0["b"]) / (y_values[0] - opt.p0["b"])
        alpha_guess = dy ** (1 / dcliff)

        opt.p0.set_if_empty(alpha=alpha_guess if alpha_guess < 1.0 else 0.99)

        if y_values[0] > opt.p0["b"]:
            opt.p0.set_if_empty(a=(y_values[0] - opt.p0["b"]) / (opt.p0["alpha"] ** x_values[0]))
        else:
            opt.p0.set_if_empty(a=0.95)

        return opt

    def _format_data(self, data: curve.CurveData) -> curve.CurveData:
        """Data format with averaging with sampling strategy."""

        # take average over the same x value by regenerating sigma from variance of y values
        series, xdata, ydata, sigma, shots = multi_mean_xy_data(
            series=data.data_index,
            xdata=data.x,
            ydata=data.y,
            sigma=data.y_err,
            shots=data.shots,
            method="sample",
        )

        # sort by x value in ascending order
        series, xdata, ydata, sigma, shots = data_sort(
            series=series,
            xdata=xdata,
            ydata=ydata,
            sigma=sigma,
            shots=shots,
        )

        return curve.CurveData(
            label="fit_ready",
            x=xdata,
            y=ydata,
            y_err=sigma,
            shots=shots,
            data_index=series,
        )

    def _extra_database_entry(self, fit_data: curve.FitData) -> List[AnalysisResultData]:
        """Calculate EPC."""
        extra_entries = []
        # pylint: disable=assignment-from-none
        quality = self._evaluate_quality(fit_data)

        # Calculate EPC
        alpha = fit_data.fitval("alpha")
        scale = (2**self._num_qubits - 1) / (2**self._num_qubits)
        epc = scale * (1 - alpha)

        extra_entries.append(
            AnalysisResultData(
                name="EPC",
                value=epc,
                chisq=fit_data.reduced_chisq,
                quality=quality,
            )
        )

        # Correction for 1Q depolarizing channel if EPGs are provided
        if self.options.epg_1_qubit and self._num_qubits == 2:
            epc = _exclude_1q_error(
                epc=epc,
                qubits=self._physical_qubits,
                gate_counts_per_clifford=self.options.gate_counts_per_clifford,
                extra_analyses=self.options.epg_1_qubit,
            )
            extra_entries.append(
                AnalysisResultData(
                    name="EPC_corrected",
                    value=epc,
                    chisq=fit_data.reduced_chisq,
                    quality=quality,
                )
            )

        # Calculate EPG
        if self.options.gate_counts_per_clifford is not None and self.options.gate_error_ratio:
            epg_dict = _calculate_epg(
                epc=epc,
                qubits=self._physical_qubits,
                gate_error_ratio=self.options.gate_error_ratio,
                gate_counts_per_clifford=self.options.gate_counts_per_clifford,
            )
            if epg_dict:
                for gate, epg_val in epg_dict.items():
                    extra_entries.append(
                        AnalysisResultData(
                            name=f"EPG_{gate}",
                            value=epg_val,
                            chisq=fit_data.reduced_chisq,
                            quality=quality,
                        )
                    )

        return extra_entries

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List["pyplot.Figure"]]:

        if self.options.gate_error_ratio is not False:
            # If gate error ratio is not False, EPG analysis is enabled.
            # Here analysis prepares gate error ratio and gate counts for EPC to EPG conversion.

            if self.options.gate_counts_per_clifford is None:
                # If gate count dictionary is not set it will compute counts from circuit metadata.
                avg_gpc = defaultdict(float)
                n_circs = len(experiment_data.data())
                for circ_result in experiment_data.data():
                    try:
                        count_ops = circ_result["metadata"]["count_ops"]
                    except KeyError as ex:
                        raise AnalysisError(
                            "'count_ops' key is not found in the circuit metadata. "
                            "This analysis cannot compute error per gates. "
                            "Please disable this with 'gate_error_ratio=False'."
                        ) from ex
                    nclif = circ_result["metadata"]["xval"]
                    for (qinds, gate), count in count_ops:
                        formatted_key = tuple(sorted(qinds)), gate
                        avg_gpc[formatted_key] += count / nclif / n_circs
                self.set_options(gate_counts_per_clifford=dict(avg_gpc))

            if self.options.gate_error_ratio is None:
                # Gate error dict is computed for gates appearing in counts dictionary
                # Error ratio among gates is determined based on the predefined lookup table.
                # This is not always accurate for every quantum backends.
                gate_error_ratio = {}
                for qinds, gate in self.options.gate_counts_per_clifford.keys():
                    if set(qinds) != set(experiment_data.metadata["physical_qubits"]):
                        continue
                    gate_error_ratio[gate] = _lookup_epg_ratio(gate, len(qinds))
                self.set_options(gate_error_ratio=gate_error_ratio)

        return super()._run_analysis(experiment_data)
