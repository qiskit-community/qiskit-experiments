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
from typing import Dict, List, Tuple, Union, TYPE_CHECKING

import lmfit
from qiskit.exceptions import QiskitError

import numpy as np
from scipy.spatial.distance import hamming

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.framework import AnalysisResultData
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from uncertainties import unumpy as unp  # pylint: disable=wrong-import-order

from .rb_analysis import RBAnalysis

if TYPE_CHECKING:
    from uncertainties import UFloat

# A dictionary key of qubit aware quantum instruction; type alias for better readability
QubitGateTuple = Tuple[Tuple[int, ...], str]


class MirrorRBAnalysis(RBAnalysis):
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

            p_0 = \sum_{k = 0}^n \left(-\frac{1}{2}\right)^k h_k

        where :math:`h_k` is the probability of observing a bitstring of Hamming distance of k from the
        correct bitstring

        Effective Polarizations (:math:`S`):

        .. math::

            S = \frac{4^n}{4^n-1}\left(\sum_{k=0}^n\left(-\frac{1}{2}\right)^k h_k\right)-\frac{1}{4^n-1}

    # section: fit_model
        The fit is based on the following decay functions:

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

    @classmethod
    def _default_options(cls):
        """Default analysis options.

        Analysis Options:
            y_axis (str): Set the metric to plot on the y-axis. Must be one of
                "Effective Polarization" (default), "Success Probability", or "Adjusted
                Success Probability".
            gate_error_ratio (Optional[Dict[str, float]]): A dictionary with gate name keys
                and error ratio values used when calculating EPG from the estimated EPC.
                The default value will use standard gate error ratios.
                If you don't know accurate error ratio between your basis gates,
                you can skip analysis of EPGs by setting this options to ``None``.
            epg_1_qubit (List[AnalysisResult]): Analysis results from previous RB experiments
                for individual single qubit gates. If this is provided, EPC of
                2Q RB is corrected to exclude the depolarization of underlying 1Q channels.
        """
        default_options = super()._default_options()

        # Set labels of axes
        default_options.plotter.set_figure_options(
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
        num_qubits = len(self._physical_qubits)

        # Initialize guess for baseline and amplitude based on infidelity type
        b_guess = 1 / 4**num_qubits
        if self.options.y_axis == "Success Probability":
            b_guess = 1 / 2**num_qubits

        mirror_curve = curve_data.get_subset_of("rb_decay")
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
        data_allocation, xdata, ydata, sigma, shots = curve.utils.multi_mean_xy_data(
            series=curve_data.data_allocation,
            xdata=curve_data.x,
            ydata=curve_data.y,
            sigma=curve_data.y_err,
            shots=curve_data.shots,
            method="sample",
        )

        # sort by x value in ascending order
        data_allocation, xdata, ydata, sigma, shots = curve.utils.data_sort(
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
        """Create analysis results for important fit parameters. Besides the
        default standard RB parameters, Entanglement Infidelity (EI) is also calculated.

        Args:
            fit_data: Fit outcome.
            quality: Quality of fit outcome.

        Returns:
            List of analysis result data.
        """

        outcomes = super()._create_analysis_results(fit_data, quality, **metadata)
        num_qubits = len(self._physical_qubits)

        # nrb is calculated for both EPC and EI per the equations in the docstring
        ei_nrb = 4**num_qubits
        ei_scale = (ei_nrb - 1) / ei_nrb
        ei = ei_scale * (1 - fit_data.ufloat_params["alpha"])

        outcomes.append(
            AnalysisResultData(
                name="EI", value=ei, chisq=fit_data.reduced_chisq, quality=quality, extra=metadata
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
        num_qubits = len(self._physical_qubits)
        target_bs = "0" * num_qubits
        for circ_result in raw_data:

            # If there is no inverting Pauli layer at the end of the circuit, get the target bitstring
            if not circ_result["metadata"]["inverting_pauli_layer"]:
                target_bs = circ_result["metadata"]["target"]

            # h[k] = proportion of shots that are Hamming distance k away from target bitstring
            hamming_dists = np.zeros(num_qubits + 1)
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
                k = int(round(hamming(target_bs_to_list, actual_bs_to_list) * num_qubits))
                hamming_dists[k] += count / circ_result.get(
                    "shots", sum(circ_result["counts"].values())
                )

            # Compute hamming distance uncertainties
            hamming_dist_unc = np.sqrt(hamming_dists * (1 - hamming_dists))

            # Compute adjusted success probability and standard deviation
            adjusted_success_prob = 0.0
            adjusted_success_prob_unc = 0.0
            for k in range(num_qubits + 1):
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
            pol_factor = 4**num_qubits
            pol = pol_factor / (pol_factor - 1) * adjusted_success_prob - 1 / (pol_factor - 1)
            pol_unc = np.sqrt(pol_factor / (pol_factor - 1)) * adjusted_success_prob_unc
            circ_result["metadata"]["polarization"] = pol
            circ_result["metadata"]["polarization_uncertainty"] = pol_unc
            if self.options.y_axis == "Effective Polarization":
                y_data.append(pol)
                y_data_unc.append(pol_unc)

        return unp.uarray(y_data, y_data_unc)
