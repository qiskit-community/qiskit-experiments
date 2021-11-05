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
import numpy as np
import scipy.linalg as la
from typing import List, Union, Dict, Tuple
import time
import ast
from qiskit.quantum_info import DensityMatrix, Choi, Operator, PTM, average_gate_fidelity
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData, Options
from qiskit.result import marginal_counts, Counts
from linear_inversion_gst import linear_inversion_gst
from gauge_optimizer import GaugeOptimizer, ideal_gateset_gen
from mle_gst import GST_Optimize


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

        - **target** (``Dict[str, PTM]`` or ``Dict[str, Operator] or Dict[str, QuantumChannel]``)
          Set a custom target quantum channels for computing the
          :func:~qiskit.quantum_info.gate_average_fidelity` of the fitted gate set against. If
          ``"default"`` or None, the ideal gate set will be used.

    """
    _builtin_fitters = {
        "linear_inversion_gst": linear_inversion_gst,
        "scipy_optimizer_MLE_gst": GST_Optimize,
    }

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options
        Analysis Options:
            fitter (str): The fitter function to use for reconstruction of GST gates.
                This can  be a string to select one of the built-in fitters, or a callable to
                supply a custom fitter function. See the `Fitter Functions` section for
                additional information.
            fitter_initial_guess (str or Dict): "linear_inversion".
        """
        options = super()._default_options()

        options.fitter = "scipy_optimizer_MLE_gst"

        #This is only in case of GST, if linear inversion, or no initial guess is needed, then the value is None.
        options.fitter_initial_guess = "linear_inversion"

        options.target_set = "default"
        #rescale_positive = False, #relevant only for the linear_inversion_results
        #rescale_trace = False,
        return options

    @classmethod
    def _get_target_set(cls, target_set):
        """ Return target set, it can be either 'ideal' or a Dict with values that are either of instance
        Gate, Operator or Choi"""
        if target_set is None or target_set == 'default':
            return 'ideal'
        if isinstance(target_set, Dict):
            if set(type(k) for k in target_set.keys()) == {str} and isinstance(list(target_set.values())[0],
                                                                               (Gate, PTM, Callable, Choi, Operator)):
                # if PTM or Callable convert them to Choi (operator or Gate or Choi, can be inputs to average_gate_fidelity)
                if isinstance(list(target_set.values())[0],PTM):
                    target_set_choi = {}
                    for key in target_set:
                        target_set_choi[key] = qi.Choi(PTM(target_set[key]))
                elif isnstance(list(target_set.values())[0],PTM):
                    # if the target provided as a function:
                    target_set_choi = {}
                    for key in target_set:
                        target_set_choi[key] = qi.Choi(PTM(np.real(gate_matrix(self.num_qubits, target_set[key]))))
                    return target_set_choi
                return target_set
        raise AnalysisError(f"Unrecognized target set {target_set}")


    @classmethod
    def _get_fitter(cls, fitter):
        """Return fitter function for named builtin fitters"""
        if fitter is None:
            raise AnalysisError("No tomography fitter given")
        if fitter in cls._builtin_fitters:
            fitter_name = fitter
            return cls._builtin_fitters[fitter], fitter_name
        raise AnalysisError(f"Unrecognized tomography fitter {fitter}")

    @classmethod
    def _get_fitter_initial_guess(cls, fitter_initial_guess):
        """Return fitter function for named builtin fitters"""
        # if fitter_initial_guess is provided by the user it should be of the form Dict[str, PTM]
        # or a str='linv'.
        if fitter_initial_guess is None:
            return None
        else:
            if isinstance(fitter_initial_guess, Dict):
                if set(type(k) for k in fitter_initial_guess.keys()) == {str} and set(type(k) for k in fitter_initial_guess.values()) == {PTM}:
                    return fitter_initial_guess
            elif fitter_initial_guess in ['linear_inversion', 'default']:
                return 'linear_inversion'
        raise AnalysisError(f"Unrecognized fitter initial guess {fitter_initial_guess}, the fitter initial guess should be  "
                                    f"of type Dict[str, PTM] or a string: 'linv'.")


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
        fitter_initial_guess = self._get_fitter_initial_guess(options.pop("fitter_initial_guess", None))

        # Get target set to compare the results with
        target_set = self._get_target_set(options.pop("target_set", None))
        if target_set == 'ideal':
            target_set = ideal_gateset_choi
        t_fitter_start_linv = time.time()
        linear_inversion_gateset = linear_inversion_gst(outcome_data, gateset_basis, num_qubits)

        # Gauge optimization of linear inversion gst results
        post_gauge_linv_gateset = GaugeOptimizer(linear_inversion_gateset, gateset_basis, num_qubits)

        t_fitter_stop_linv = time.time()
        fitter_time_linv=t_fitter_stop_linv-t_fitter_start_linv
        if (fitter_name == 'linear_inversion_gst'):

            #rescale_positive = options.pop("rescale_positive")
            #rescale_trace = options.pop("rescale_trace")

            fitter_time = fitter_time_linv
            # I should add the data post analysis here
            gateset_result = post_gauge_linv_gateset
            #if rescale_positive:
            #    gateset_result = self._rescale_PSD(gateset_result)
            #    if rescale_trace:
            #        gateset_result = self._rescale_PSD(gateset_result)

        else:
            if (fitter_name == 'scipy_optimizer_MLE_gst'):#should be the only option
                if (fitter_initial_guess =='linear_inversion'):
                    initial_gateset = post_gauge_linv_gateset
                elif (fitter_initial_guess == None):
                    initial_gateset = None
                else:
                    # if it is given by the user as Dict[str, PTM].
                    initial_gateset = fitter_initial_guess
                gst_optimizer = GST_Optimize(gateset_basis.gate_labels,
                                             gateset_basis.spam_labels,
                                             gateset_basis.spam_spec,
                                             outcome_data,
                                             initial_gateset,
                                             num_qubits)
                #print(gst_optimizer.optimize(), 'gst_optimizer.optimize()')
                gateset_result = gst_optimizer.optimize()
                t_fitter_stop=time.time()
                fitter_time=t_fitter_stop-t_fitter_start_linv


        fitter_metadata = {}
        fitter_metadata["fitter"] = fitter_name
        fitter_metadata["fitter_time"] =fitter_time
        #if (fitter_name == "linear_inversion_gst"):
        #    fitter_metadata["rescale_psd"] = rescale_positive
        #    fitter_metadata["rescale_TP"] = rescale_TP

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
                     ) -> Dict[Tuple[str], Dict]:
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
                                           ) -> Dict[Tuple[str], Dict]:
        probs = {}
        for key, vals in data.items():
            # We assume the POVM is always |E>=|0000..><0000..|
            probs[key] = vals.get('0' * num_qubits, 0) / sum(vals.values())
        #print(probs)
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
        metadata["GST SPAM gates"]=spam_gates_labels
        # convert all results to choi representation
        gateset_result_choi={}
        for key in gateset_result:
            gateset_result_choi[key] = Choi(PTM(gateset_result[key]))

        analysis_results=[]
        analysis_results.append(AnalysisResultData("GST Experiment properties", metadata))
        for key in gateset_result_choi:
            metadata_key={}
            if key not in ['rho', 'E']:
                metadata_key["Average gate fidelity"] = gate_average_fidelity(gateset_result_choi[key], target_gateset[key])
                #metadata["Trace Preservation"] =
                #metadate["Complete Positivity"]=
            analysis_results.append(AnalysisResultData("gst estimation of {}".format(key), gateset_result_choi[key], extra=metadata_key))

            #cls._hs_distance_result(gateset_result, target_gateset_ideal)
            #cls._froebenius_distance(gateset_result, target_gateset_ideal)

        return analysis_results



    """
    #I should add something for the noisy results fidelity...target_noisy or something like this
    def _fidelity_result(result, target):
        name = "gst fidelity with ideal target"
        fidelity={}
        for key in result.keys():
            if key not in ['E', 'rho']:
                fidelity[key] = gate_average_fidelity(result[key], target[key])
        return AnalysisResultData(name, fidelity)
    """

    @staticmethod
    def _hs_distance_result(result, target):
        name="Hilbert-Schmidtt norm for the target and gst results"
        hs_distance_results={}
        for key in result.keys():
            if key not in ['E', 'rho']:
                hs_distance_results[key] = hs_distance(result[key].data, target[key].data)
        return AnalysisResultData(name, hs_distance_results)


    @staticmethod
    def _froebenius_distance(result, target):
        name="froebenius distance between target and gst results"
        froeb_distance={}
        for key in result.keys():
            if key not in ['E', 'rho']:
                froeb_distance[key]= froeb_dist(result[key].data, target[key].data)
        return AnalysisResultData(name, froeb_distance)


def gate_average_fidelity(A, B):
    #A and B are both PTM
    return average_gate_fidelity(A, B)

def hs_distance(A, B):
    """
    Computes the Hilbert-Schmidt distance between two matrices A and B
    """
    return sum([np.abs(x) ** 2 for x in np.nditer(A - B)])

def froeb_dist(A, B):

    """
    Computes the Frobenius distance between two matrices A and B.
    """
    return np.sqrt(np.trace(np.dot(np.subtract(A, B), (np.subtract(A, B).T))))