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
Standard GST analysis class.
"""
import numpy as np
import scipy.linalg as la
from typing import List, Union, Dict, Tuple, Callable
from types import FunctionType
import time
import ast
from qiskit.quantum_info import DensityMatrix, Choi, Operator, PTM, average_gate_fidelity, state_fidelity, process_fidelity
from qiskit_experiments.exceptions import AnalysisError, QiskitError
from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData, Options
from qiskit.result import marginal_counts, Counts
from linear_inversion_gst import linear_inversion_gst
from gauge_optimizer import GaugeOptimizer, ideal_gateset_gen, Pauli_strings
from mle_gst import GSTOptimize, convert_from_ptm
from qiskit_experiments.library.tomography.fitters.fitter_utils import make_positive_semidefinite
from qiskit.circuit import Gate
from gatesetbasis import gate_matrix


class GSTAnalysis(BaseAnalysis):
    r"""A class to analyze gate set tomography experiment.
    
    Analysis Options
        - **fitter** (``str``): The fitter function to use for reconstruction
          of GST gate set.
          This can  be a string to select one of the built-in fitters. Built-in
          fitter functions may be selected using the following string
          labels:
          * ``"linear_inversion_gst"``:
          :func:`~qiskit_experiments.library.gateset_tomography.linear_inversion_gst` 
          * ``"scipy_optimizer_MLE_gst"``:
          :class:`~qiskit_experiments.library.gateset_tomography.mle_gst` (Default)
        - ** initial_fitter_guess** (``str or Dict``): The MLE fitter initial guess if
          MLE fitter is chosen (Not relevant for the linear inversion fitter)
          to reconstruct the data. If None, the optimization will be performed without 
          initial point. It takes as a string either "default" or "linear_inversion"
          and in both cases it uses linear inversion as a starting solution. The user
          can provide an arbitrary initial guess for the MLE fitter which is of the 
          form of a Dict[str, PTM] for the gate set guess.
        - **target_set** (``Dict``)
          Set a custom target quantum channels for computing the
          :func:~qiskit.quantum_info.gate_average_fidelity` of the fitted gate set against. If
          ``"default"`` or None, the ideal gate set will be used.
        - **rescale_CP** (``bool``): If True rescale the Choi representation
          (:class:`~qiskit.quantum_info.Choi`) of the gateset returned by the fitter
          to be positive-semidefinite which corresponds to CP gate set results.
          As GST results obtained by MLE fitter are always CP, this is only
          relevant for linear inversion and thus, its default is False (Default: False).
        - **rescale_TP** (``bool``):  If True rescale the Choi representation
          (:class:`~qiskit.quantum_info.Choi`) of the gateset returned by the fitter
          to be trace dim (2**num_qubits) which corresponds to TP gate set results.
          As GST results obtained by MLE fitter are always TP, this is only
          relevant for linear inversion and thus, its default is False (Default: False).

    """
    _builtin_fitters = {
        "linear_inversion_gst": linear_inversion_gst,
        "scipy_optimizer_MLE_gst": GSTOptimize,
    }

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options
        """
        options = super()._default_options()
        options.fitter = "scipy_optimizer_MLE_gst"
        # This is only in case of GST, if linear inversion, or no initial guess is needed, then the value is None.
        options.fitter_initial_guess = "linear_inversion"
        options.target_set = "ideal"  # the default corresponds to the ideal gate set.
        options.rescale_CP = False  # relevant only for the linear_inversion_results
        options.rescale_TP = False
        return options

    @classmethod
    def _get_target_set(cls, target_set, num_qubits):
        """ Return target set, it can be either 'ideal' if None or 'default', or a Dict with values that
         are either of instance Gate, Operator or Choi (If the user provides target set with values of type
         function  or PTM, they will be converted to Choi)."""
        if target_set is None or target_set == 'default' or target_set == 'ideal':
            return 'ideal'
        if isinstance(target_set, Dict):
            # E and rho if provided by user, should be first and second elements of target_set and E should
            # be either BaseOperator or PTM, rho takes only PTM or DensityMatrix
            """
            if 'rho' in target_set.keys():
                if not isinstance(target_set['rho'], (PTM, DensityMatrix)):
                    raise AnalysisError(f"Unrecognized target preparation state {rho}")
            if 'E' in target_set.keys():
                if not isinstance(target_set['E'], (PTM, Operator)):
                    raise AnalysisError(f"Unrecognized target measurement state {E}")
`           """
            if set(type(k) for k in target_set.keys()) == {str} and isinstance(list(target_set.values())[2],
                                                                               (Gate, PTM, FunctionType, Choi,
                                                                                Operator)):
                # if PTM or FunctionType convert them to Choi (operator or Gate or Choi,
                # can be taken as inputs to average_gate_fidelity)
                if isinstance(list(target_set.values())[2], PTM):
                    target_set_choi = {}
                    for key in target_set:
                        target_set_choi[key] = Choi(PTM(target_set[key]))
                elif isinstance(list(target_set.values())[2], FunctionType):  # if the target provided as a function:
                    target_set_choi = {}
                    for key in target_set:
                        target_set_choi[key] = Choi(PTM(np.real(gate_matrix(num_qubits, target_set[key]))))
                    return target_set_choi
                return target_set
        raise AnalysisError(f"Unrecognized target set {target_set}")

    @classmethod
    def _get_fitter(cls, fitter):
        """Return fitter function for named builtin fitters"""
        if fitter is None or fitter == 'default':
            return cls._builtin_fitters["scipy_optimizer_MLE_gst"], "scipy_optimizer_MLE_gst"
        if fitter in cls._builtin_fitters:
            fitter_name = fitter
            return cls._builtin_fitters[fitter], fitter_name
        raise AnalysisError(f"Unrecognized tomography fitter {fitter}")

    @classmethod
    def _get_fitter_initial_guess(cls, fitter_initial_guess, num_qubits):
        """Return fitter function for named builtin fitters"""
        # if fitter_initial_guess is provided by the user it should be of the form Dict[str, (Gate, PTM, FunctionType, Choi, Operator)]
        # or a str='linv' or 'default'.
        if fitter_initial_guess is None:
            return None
        else:
            if isinstance(fitter_initial_guess, Dict):
                if set(type(k) for k in fitter_initial_guess.keys()) == {str} and isinstance(
                        list(fitter_initial_guess.values())[2], (Gate, PTM, FunctionType, Choi, Operator)):
                    if isinstance(fitter_initial_guess['E'], np.ndarray) == False or isinstance(
                            fitter_initial_guess['rho'], (np.ndarray, DensityMatrix)) == False:
                        # PTM type does not work with a vector like E and rho..
                        raise AnalysisError(f"Unrecognized fitter initial guess {fitter_initial_guess},"
                                            f" the initial guess of E should be PTM (array) or Operator,"
                                            f" and the initial guess of rho should be PTM(array) or DensityMatrix")
                    else:
                        if isinstance(fitter_initial_guess['rho'], DensityMatrix):
                            fitter_initial_guess['rho'] = density_matrix_to_ptm(fitter_initial_guess['rho'], num_qubits)
                        else:
                            if np.shape(fitter_initial_guess['rho']) is not ((2 ** (2 * num_qubits), 1)):
                                fitter_initial_guess['rho'] = np.reshape(fitter_initial_guess['rho'],
                                                                         ((2 ** (2 * num_qubits), 1)))

                        if isinstance(list(fitter_initial_guess.values())[2], Choi):
                            fitter_initial_guess_ptm = {}
                            fitter_initial_guess_ptm['E'] = fitter_initial_guess['E']
                            fitter_initial_guess_ptm['rho'] = fitter_initial_guess['rho']

                            for key in fitter_initial_guess:
                                if key not in ['E', 'rho']:
                                    fitter_initial_guess_ptm[key] = PTM(Choi(fitter_initial_guess[key]))
                            return fitter_initial_guess_ptm
                        elif isinstance(list(fitter_initial_guess.values())[2], (Gate, Operator)):
                            fitter_initial_guess_ptm = {}
                            fitter_initial_guess_ptm['E'] = fitter_initial_guess['E']
                            fitter_initial_guess_ptm['rho'] = fitter_initial_guess['rho']
                            for key in fitter_initial_guess:
                                if key not in ['E', 'rho']:
                                    fitter_initial_guess_ptm[key] = PTM(fitter_initial_guess[key])
                            return fitter_initial_guess_ptm
                        elif isinstance(list(fitter_initial_guess.values())[2], FunctionType):
                            fitter_initial_guess_ptm = {}
                            fitter_initial_guess_ptm['E'] = fitter_initial_guess['E']
                            fitter_initial_guess_ptm['rho'] = fitter_initial_guess['rho']
                            for key in fitter_initial_guess:
                                if key not in ['E', 'rho']:
                                    fitter_initial_guess_ptm[key] = PTM(
                                        np.real(gate_matrix(num_qubits, fitter_initial_guess[key])))
                            return fitter_initial_guess_ptm
                        # if PTM
                        return fitter_initial_guess

            elif fitter_initial_guess in ['linear_inversion', 'default']:
                return 'linear_inversion'
        raise AnalysisError(f"Unrecognized fitter initial guess {fitter_initial_guess},"
                            f" the fitter initial guess should be  "
                            f"of type Dict[str, (Gate, PTM, FunctionType, Choi, Operator)] "
                            f"or a string: 'default' or 'linear_inversion'.")

    def _run_analysis(self, experiment_data, **options):
        # Extract tomography measurement data

        num_qubits = experiment_data.metadata["num_qubits"]
        gateset_basis = experiment_data.metadata["gateset_basis"]
        ideal_gateset_choi = ideal_gateset_gen(gateset_basis, num_qubits, 'Choi')
        outcome_data = self._fitter_data(
            experiment_data.data(), num_qubits
        )

        # Get tomography fitter function
        fitter, fitter_name = self._get_fitter(options.pop("fitter", None))
        # Get tomography fitter initial guess
        fitter_initial_guess = self._get_fitter_initial_guess(options.pop("fitter_initial_guess", None), num_qubits)
        # Get target set to compare the results with
        target_set = self._get_target_set(options.pop("target_set", None), num_qubits)
        if target_set == 'ideal':
            target_set = ideal_gateset_choi
        t_fitter_start_linv = time.time()
        linear_inversion_gateset = linear_inversion_gst(outcome_data, gateset_basis)

        # Gauge optimization of linear inversion gst results
        post_gauge_linv_gateset = GaugeOptimizer(linear_inversion_gateset, gateset_basis, num_qubits)

        t_fitter_stop_linv = time.time()
        fitter_time_linv = t_fitter_stop_linv - t_fitter_start_linv
        if fitter_name == 'linear_inversion_gst':
            rescale_CP = options.pop("rescale_CP")
            rescale_TP = options.pop("rescale_TP")

            fitter_time = fitter_time_linv
            gateset_result = post_gauge_linv_gateset
            gateset_result['E'] = Operator(convert_from_ptm(gateset_result['E'], num_qubits))
            gateset_result['rho'] = DensityMatrix(convert_from_ptm(gateset_result['rho'], num_qubits))
            if rescale_CP:
                gateset_result = self._rescale_CP(gateset_result)
            if rescale_TP:
                gateset_result = self._rescale_TP(gateset_result, num_qubits)
        else:
            if fitter_name == 'scipy_optimizer_MLE_gst':  # should be the only option
                if fitter_initial_guess == 'linear_inversion':
                    initial_gateset = post_gauge_linv_gateset
                elif fitter_initial_guess is None:
                    initial_gateset = None
                else:
                    # if it is given by the user as Dict[str, PTM].
                    initial_gateset = fitter_initial_guess
                gst_optimizer = GSTOptimize(gateset_basis.gate_labels,
                                            gateset_basis.spam_labels,
                                            gateset_basis.spam_spec,
                                            outcome_data,
                                            initial_gateset,
                                            num_qubits)
                gateset_result = gst_optimizer.optimize()
                t_fitter_stop = time.time()
                fitter_time = t_fitter_stop - t_fitter_start_linv
        fitter_metadata = {}
        fitter_metadata["fitter"] = fitter_name
        fitter_metadata["fitter_time"] = fitter_time
        if fitter_name == "linear_inversion_gst":
            fitter_metadata["rescale_CP"] = rescale_CP
            fitter_metadata["rescale_TP"] = rescale_TP
        if fitter_name == "scipy_optimizer_MLE_gst":
            fitter_metadata["fitter_initial_guess"] = fitter_initial_guess
        analysis_results = self._postprocess_fit(
            gateset_result,
            metadata=fitter_metadata,
            target_gateset=target_set,
            gates_labels=gateset_basis.gate_labels,
            spam_gates_labels=gateset_basis.spam_spec,
        )

        return analysis_results, []

    def _fitter_data(self,
                     data: List[Dict[str, any]], num_qubits
                     ) -> Dict[Tuple[str], float]:
        """Return list a tuple of basis, frequency, shot data"""
        outcome_dict = {}

        for datum in data:
            # Get basis data
            metadata = datum["metadata"]
            circuit_name = ast.literal_eval(metadata["circuit_name"])
            # Add outcomes
            # counts = Counts(marginal_counts(datum["counts"])).int_outcomes()
            counts = datum["counts"]
            outcome_dict[circuit_name] = counts
        return self._counts_to_probabilities_converter(outcome_dict, num_qubits)

    @staticmethod
    def _counts_to_probabilities_converter(data: Dict[Tuple[str], Dict], num_qubits
                                           ) -> Dict[Tuple[str], float]:
        probs = {}
        for key, vals in data.items():
            # We assume the POVM is always |E>=|0000..><0000..|
            probs[key] = vals.get('0' * num_qubits, 0) / sum(vals.values())
        return probs

    @classmethod
    # I should add also target gateset noisy if found.
    def _postprocess_fit(
            cls,
            gateset_result,
            metadata,
            target_gateset,
            gates_labels,
            spam_gates_labels,
    ):
        """Post-process fitter data"""
        # Results list
        # add spam gates labels to extra
        metadata["GST gates"] = gates_labels
        metadata["GST SPAM gates"] = spam_gates_labels
        # convert all results to choi representation
        gateset_result_choi = {}
        gateset_result_choi['E'] = gateset_result['E']
        gateset_result_choi['rho'] = gateset_result['rho']

        for key in gateset_result:
            if key not in ['E', 'rho']:
                gateset_result_choi[key] = Choi(PTM(gateset_result[key]))
        analysis_results = []
        analysis_results.append(AnalysisResultData("GST Experiment properties", metadata))
        """
        Is this part needed? How about rescaling rho to be positive and with trace 1?
        metadata_rho = {}
        if not isinstance(target_gateset['rho'], DensityMatrix):
            target_gateset['rho'] = DensityMatrix(convert_from_ptm(target_gateset['rho'], num_qubits))
        metadata_rho["Preparation state fidelity"] = state_fidelity(gateset_result_choi['rho'], target_gateset['rho'])
        analysis_results.append(AnalysisResultData("gst estimation of {}".format('rho'), gateset_result_choi['rho'], extra = metadata_rho))
        """
        for key in gateset_result_choi:
            metadata_key = {}
            if key not in ['rho', 'E']:
                try:
                    metadata_key["Average gate fidelity"] = gate_average_fidelity(gateset_result_choi[key],
                                                                              target_gateset[key])
                except(QiskitError):
                    metadata_key["Process fidelity"] = process_fidelity(gateset_result_choi[key],
                                                                              target_gateset[key])

            analysis_results.append(
                AnalysisResultData("gst estimation of {}".format(key), gateset_result_choi[key], extra=metadata_key))

            # cls._hs_distance_result(gateset_result, target_gateset_ideal)
            # cls._froebenius_distance(gateset_result, target_gateset_ideal)

        return analysis_results

    @staticmethod
    def _rescale_CP(gateset_result):
        # This method is only relevant for the linear inversion fitter as MLE results are always CP
        gateset_result_rescaled_CP = {}
        gateset_result_rescaled_CP['E'] = gateset_result['E']
        gateset_result_rescaled_CP['rho'] = gateset_result['rho']
        for key in gateset_result.keys():
            if key not in ['E', 'rho']:
                gateset_result_rescaled_CP[key] = PTM(
                    Choi(make_positive_semidefinite(Choi(PTM(gateset_result[key])).data)))
        return gateset_result_rescaled_CP

    @staticmethod
    def _rescale_TP(gateset_result, num_qubits):
        # This method is only relevant for the linear inversion fitter as MLE results are always CP
        gateset_result_rescaled_TP = {}
        gateset_result_rescaled_TP['E'] = gateset_result['E']
        gateset_result_rescaled_TP['rho'] = gateset_result['rho']
        for key in gateset_result.keys():
            if key not in ['E', 'rho']:
                gate = Choi(PTM(gateset_result[key])).data
                gateset_result_rescaled_TP[key] = PTM(Choi((2 ** num_qubits) * gate / np.trace(gate)))
        return gateset_result_rescaled_TP


def gate_average_fidelity(A, B):
    # A and B are both PTM
    return average_gate_fidelity(A, B)


def density_matrix_to_ptm(density_matrix, num_qubits):
    """Returns the PTM representation of an arbitrary density matrix"""
    d = np.power(2, num_qubits)
    if isinstance(density_matrix, DensityMatrix):
        density_matrix = density_matrix.data

    matrix_pauli = [np.trace(np.dot(density_matrix.data, Pauli_strings(num_qubits)[i])) for i in range(np.power(d, 2))]
    # decompoition into Pauli strings basis (PTM representation)
    return np.reshape(matrix_pauli, (np.power(d, 2), 1))
