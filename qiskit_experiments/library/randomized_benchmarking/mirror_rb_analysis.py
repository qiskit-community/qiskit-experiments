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
Mirror RB analysis class.
"""
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple, Union, Optional, TYPE_CHECKING

import lmfit
from qiskit.exceptions import QiskitError

import numpy as np
from scipy.spatial.distance import hamming

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import AnalysisResultData, ExperimentData
from qiskit_experiments.database_service import DbAnalysisResultV1
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from uncertainties import unumpy as unp  # pylint: disable=wrong-import-order

if TYPE_CHECKING:
    from uncertainties import UFloat

# A dictionary key of qubit aware quantum instruction; type alias for better readability
QubitGateTuple = Tuple[Tuple[int, ...], str]


class MirrorRBAnalysis(curve.CurveAnalysis):
    r"""A class to analyze mirror randomized benchmarking experiment.

    # section: overview
        This analysis takes a series for Mirror RB curve fitting.
        From the fit :math:`\alpha` value this analysis estimates the mean entanglement infidelity (EI)
        and the error per Clifford (EPC), also known as the average gate infidelity (AGI).

        The EPC (AGI) estimate is obtained using the equation

        .. math::

            EPC = \frac{2^n - 1}{2^n}\left(1 - \alpha\right)

        where :math:`n` is the number of qubits (width of the circuit).

        The EI is obtained using the equation

        .. math::

            EI = \frac{4^n - 1}{4^n}\left(1 - \alpha\right)

        The fit :math:`\alpha` parameter can be fit using one of the following three quantities
        plotted on the y-axis:

        Success Probabilities (:math:`p`): The proportion of shots that return the correct bitstring

        Adjusted Success Probabilities (:math:`p_0`):

        .. math::

            p_0 = \sum_{k = 0}^n (-\frac{1}{2})^k h_k

        where :math:`h_k` is the probability of observing a bitstring of Hamming distance of k from the
        correct bitstring

        Effective Polarizations (:math:`S`):

        .. math::

            S = \frac{4^n}{4^n - 1}\sum_{k = 0}^n (-\frac{1}{2})^k h_k - \frac{1}{4^n - 1}

    # section: fit_model
        The fit is based on the following decay functions:

        Fit model for mirror RB

        .. math::

            F(x) = a \alpha^{x} + b

    # section: fit_parameters
        defpar a:
            desc: Height of decay curve.
            init_guess: Determined by :math:`1 - b`.
            bounds: [0, 1]
        defpar b:
            desc: Base line.
            init_guess: Determined by :math:`(1/2)^n` (for success probability) or :math:`(1/4)^n`
            (for adjusted success probability and effective polarization).
            bounds: [0, 1]
        defpar \alpha:
            desc: Depolarizing parameter.
            init_guess: Determined by :func:`~rb_decay` with standard RB curve.
            bounds: [0, 1]

    # section: reference
        .. ref_arxiv:: 1 2112.09853

    """

    def __init__(self):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="a * alpha ** x + b", name="mirror", data_sort_key={"mirror": True}
                )
            ]
        )
        self._gate_counts_per_clifford = None
        self._physical_qubits = None
        self._num_qubits = None

    # __series__ = [
    #     curve.SeriesDef(
    #         name="Mirror",
    #         fit_func=lambda x, a, alpha, b: curve.fit_function.exponential_decay(
    #             x, amp=a, lamb=-1.0, base=alpha, baseline=b
    #         ),
    #         filter_kwargs={"mirror": True},
    #         plot_color="blue",
    #         plot_symbol="^",
    #         model_description=r"a \alpha^{x} + b",
    #     )
    # ]

    @classmethod
    def _default_options(cls):
        """Default analysis options."""
        default_options = super()._default_options()

        # Set labels of axes
        default_options.curve_drawer.set_options(
            xlabel="Clifford Length",
            ylabel="Effective Polarization",
        )

        # Plot all (adjusted) success probabilities
        default_options.plot_raw_data = True

        # Exponential decay parameter
        default_options.result_parameters = ["alpha"]

        # Default gate error ratio for calculating EPG
        default_options.gate_error_ratio = "default"

        # By default, EPG for single qubits aren't set
        default_options.epg_1_qubit = None

        # By default, effective polarization is plotted (see arXiv:2112.09853). We can
        # also plot success probability or adjusted success probability (see PyGSTi).
        # Do this by setting options to "Success Probability" or "Adjusted Success Probability"
        default_options.y_axis = "Effective Polarization"

        return default_options

    def set_options(self, **fields):
        if "y_axis" in fields:
            if fields["y_axis"] not in [
                "Success Probability",
                "Adjusted Success Probability",
                "Effective Polarization",
            ]:
                raise QiskitError(
                    'y_axis must be one of "Success Probability", "Adjusted Success Probability", '
                    'or "Effective Polarization"'
                )
        super().set_options(**fields)

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Create algorithmic guess with analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        user_opt.bounds.set_if_empty(a=(0, 1), alpha=(0, 1), b=(0, 1))

        # Initialize guess for baseline and amplitude based on infidelity type
        b_guess = 1 / 4**self._num_qubits
        if self.options.y_axis == "Success Probability":
            b_guess = 1 / 2**self._num_qubits

        mirror_curve = curve_data.get_subset_of("mirror")
        alpha_mirror = curve.guess.rb_decay(mirror_curve.x, mirror_curve.y, b=b_guess)
        a_guess = (curve_data.y[0] - b_guess) / (alpha_mirror ** curve_data.x[0])

        user_opt.p0.set_if_empty(b=b_guess, a=a_guess, alpha=alpha_mirror)

        return user_opt

    def _format_data(
        self,
        curve_data: curve.CurveData,
    ) -> curve.CurveData:
        """Postprocessing for the processed dataset.

        Args:
            curve_data: Processed dataset created from experiment results.

        Returns:
            Formatted data.
        """
        # TODO Eventually move this to data processor, then create RB data processor.

        # take average over the same x value by keeping sigma
        data_allocation, xdata, ydata, sigma, shots = curve.data_processing.multi_mean_xy_data(
            series=curve_data.data_allocation,
            xdata=curve_data.x,
            ydata=curve_data.y,
            sigma=curve_data.y_err,
            shots=curve_data.shots,
            method="sample",
        )

        # sort by x value in ascending order
        data_allocation, xdata, ydata, sigma, shots = curve.data_processing.data_sort(
            series=data_allocation,
            xdata=xdata,
            ydata=ydata,
            sigma=sigma,
            shots=shots,
        )

        return curve.CurveData(
            x=xdata,
            y=ydata,
            y_err=sigma,
            shots=shots,
            data_allocation=data_allocation,
            labels=curve_data.labels,
        )

    def _create_analysis_results(
        self,
        fit_data: curve.FitData,
        quality: str,
        **metadata,
    ) -> List[AnalysisResultData]:
        """Create analysis results for important fit parameters.

        Args:
            fit_data: Fit outcome.
            quality: Quality of fit outcome.

        Returns:
            List of analysis result data.
        """

        outcomes = super()._create_analysis_results(fit_data, quality, **metadata)
        num_qubits = len(self._physical_qubits)

        # nrb is calculated for both EPC and EI per the equations in the docstring
        ei_nrb = 4**self._num_qubits
        ei_scale = (ei_nrb - 1) / ei_nrb
        epc_nrb = 2**self._num_qubits
        epc_scale = (epc_nrb - 1) / epc_nrb

        alpha = fit_data.ufloat_params["alpha"]

        # Calculate EPC and EI per the equations in the docstring
        epc = epc_scale * (1 - alpha)
        ei = ei_scale * (1 - alpha)

        outcomes.append(
            AnalysisResultData(
                name="EPC", value=epc, chisq=fit_data.reduced_chisq, quality=quality, extra=metadata
            )
        )
        outcomes.append(
            AnalysisResultData(
                name="EI", value=ei, chisq=fit_data.reduced_chisq, quality=quality, extra=metadata
            )
        )

        # Correction for 1Q depolarizing channel if EPGs are provided
        if self.options.epg_1_qubit and num_qubits == 2:
            epc = _exclude_1q_error(
                epc=epc,
                qubits=self._physical_qubits,
                gate_counts_per_clifford=self._gate_counts_per_clifford,
                extra_analyses=self.options.epg_1_qubit,
            )
            outcomes.append(
                AnalysisResultData(
                    name="EPC_corrected",
                    value=epc,
                    chisq=fit_data.reduced_chisq,
                    quality=quality,
                    extra=metadata,
                )
            )

        # Calculate EPG
        if self._gate_counts_per_clifford is not None and self.options.gate_error_ratio:
            epg_dict = _calculate_epg(
                epc=epc,
                qubits=self._physical_qubits,
                gate_error_ratio=self.options.gate_error_ratio,
                gate_counts_per_clifford=self._gate_counts_per_clifford,
            )
            if epg_dict:
                for gate, epg_val in epg_dict.items():
                    outcomes.append(
                        AnalysisResultData(
                            name=f"EPG_{gate}",
                            value=epg_val,
                            chisq=fit_data.reduced_chisq,
                            quality=quality,
                            extra=metadata,
                        )
                    )

        return outcomes

    def _run_data_processing(
        self, raw_data: List[Dict], models: List[lmfit.Model]
    ) -> curve.CurveData:
        """Manual data processing

        Args:
            raw_data: Payload in the experiment data.
            models: A list of LMFIT models that provide the model name and
                optionally data sorting keys.

        Returns:
            Processed data that will be sent to the formatter method.

        Raises:
            DataProcessorError: When model is multi-objective function but
                data sorting option is not provided.
            DataProcessorError: When key for x values is not found in the metadata.
        """
        x_key = self.options.x_key

        try:
            xdata = np.asarray([datum["metadata"][x_key] for datum in raw_data], dtype=float)
        except KeyError as ex:
            raise DataProcessorError(
                f"X value key {x_key} is not defined in circuit metadata."
            ) from ex

        ydata = self._compute_polarizations_and_probabilities(raw_data)
        shots = np.asarray([datum.get("shots", np.nan) for datum in raw_data])

        def _matched(metadata, **filters):
            try:
                return all(metadata[key] == val for key, val in filters.items())
            except KeyError:
                return False

        if len(models) == 1:
            # all data belongs to the single model
            data_allocation = np.full(xdata.size, 0, dtype=int)
        else:
            data_allocation = np.full(xdata.size, -1, dtype=int)
            for idx, sub_model in enumerate(models):
                try:
                    tags = sub_model.opts["data_sort_key"]
                except KeyError as ex:
                    raise DataProcessorError(
                        f"Data sort options for model {sub_model.name} is not defined."
                    ) from ex
                if tags is None:
                    continue
                matched_inds = np.asarray(
                    [_matched(d["metadata"], **tags) for d in raw_data], dtype=bool
                )
                data_allocation[matched_inds] = idx

        return curve.CurveData(
            x=xdata,
            y=unp.nominal_values(ydata),
            y_err=unp.std_devs(ydata),
            shots=shots,
            data_allocation=data_allocation,
            labels=[sub_model._name for sub_model in models],
        )

    def _compute_polarizations_and_probabilities(self, raw_data: List[Dict]) -> unp.uarray:
        """Compute success probabilities, adjusted success probabilities, and
        polarizations from raw results

        Args:
            raw_data: List of raw results for each circuit

        Returns:
            Unp array of either success probabiltiies, adjusted success probabilities,
            or polarizations as specified by the user.
        """

        # Arrays to store the y-axis data and uncertainties
        y_data = []
        y_data_unc = []
        target_bs = "0" * self._num_qubits
        for circ_result in raw_data:

            # If there is no inverting Pauli layer at the end of the circuit, get the target bitstring
            if not circ_result["metadata"]["inverting_pauli_layer"]:
                target_bs = circ_result["metadata"]["target"]

            # h[k] = proportion of shots that are Hamming distance k away from target bitstring
            hamming_dists = np.zeros(self._num_qubits + 1)
            for bitstring, count in circ_result["counts"].items():
                # Compute success probability
                success_prob = 0.0
                if bitstring == target_bs:
                    success_prob = count / circ_result.get(
                        "shots", sum(circ_result["counts"].values())
                    )
                    success_prob_unc = np.sqrt(success_prob * (1 - success_prob))
                    if self.options.y_axis == "Success Probability":
                        y_data.append(success_prob)
                        y_data_unc.append(success_prob_unc)
                    circ_result["metadata"]["success_probability"] = success_prob
                    circ_result["metadata"]["success_probability_stddev"] = success_prob_unc

                # Compute hamming distance proportions
                target_bs_to_list = [int(char) for char in target_bs]
                actual_bs_to_list = [int(char) for char in bitstring]
                k = round(hamming(target_bs_to_list, actual_bs_to_list) * self._num_qubits)
                hamming_dists[k] += count / circ_result.get(
                    "shots", sum(circ_result["counts"].values())
                )

            # Compute hamming distance uncertainties
            hamming_dist_unc = np.sqrt(hamming_dists * (1 - hamming_dists))

            # Compute adjusted success probability and standard deviation
            adjusted_success_prob = 0.0
            adjusted_success_prob_unc = 0.0
            for k in range(self._num_qubits + 1):
                adjusted_success_prob += (-0.5) ** k * hamming_dists[k]
                adjusted_success_prob_unc += (0.5) ** k * hamming_dist_unc[k] ** 2
            adjusted_success_prob_unc = np.sqrt(adjusted_success_prob_unc)
            circ_result["metadata"]["adjusted_success_probability"] = adjusted_success_prob
            circ_result["metadata"][
                "adjusted_success_probability_stddev"
            ] = adjusted_success_prob_unc
            if self.options.y_axis == "Adjusted Success Probability":
                y_data.append(adjusted_success_prob)
                y_data_unc.append(adjusted_success_prob_unc)

            # Compute effective polarization and standard deviation (arXiv:2112.09853v1)
            pol_factor = 4**self._num_qubits
            pol = pol_factor / (pol_factor - 1) * adjusted_success_prob - 1 / (pol_factor - 1)
            pol_unc = np.sqrt(pol_factor / (pol_factor - 1)) * adjusted_success_prob_unc
            circ_result["metadata"]["polarization"] = pol
            circ_result["metadata"]["polarization_uncertainty"] = pol_unc
            if self.options.y_axis == "Effective Polarization":
                y_data.append(pol)
                y_data_unc.append(pol_unc)

        return unp.uarray(y_data, y_data_unc)

    def _initialize(
        self,
        experiment_data: ExperimentData,
    ):
        """Initialize curve analysis with experiment data.

        This method is called ahead of other processing.

        Args:
            experiment_data: Experiment data to analyze.

        Raises:
            AnalysisError: When circuit metadata for ops count is missing.
        """
        super()._initialize(experiment_data)

        if self.options.gate_error_ratio is not None:
            # If gate error ratio is not False, EPG analysis is enabled.
            # Here analysis prepares gate error ratio and gate counts for EPC to EPG conversion.

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
            self._gate_counts_per_clifford = dict(avg_gpc)

            if self.options.gate_error_ratio == "default":
                # Gate error dict is computed for gates appearing in counts dictionary
                # Error ratio among gates is determined based on the predefined lookup table.
                # This is not always accurate for every quantum backends.
                gate_error_ratio = {}
                for qinds, gate in self._gate_counts_per_clifford.keys():
                    if set(qinds) != set(experiment_data.metadata["physical_qubits"]):
                        continue
                    gate_error_ratio[gate] = _lookup_epg_ratio(gate, len(qinds))
                self.set_options(gate_error_ratio=gate_error_ratio)

        # Get qubit number
        self._physical_qubits = experiment_data.metadata["physical_qubits"]
        self._num_qubits = len(experiment_data.metadata["physical_qubits"])


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
    """A helper mehtod to compute EPGs of basis gates from fit EPC value.

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
    """A helper method to exclude contribution of single qubit gates from 2Q EPC.

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
