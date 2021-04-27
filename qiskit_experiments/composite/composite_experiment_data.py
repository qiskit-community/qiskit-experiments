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
Composite Experiment data class.
"""

from qiskit.result import marginal_counts
from qiskit.providers.experiment import ExperimentData


class CompositeExperimentData(ExperimentData):
    """Composite experiment data class"""

    def __init__(self, experiment):
        """Initialize the experiment data.

        Args:
            experiment (CompositeExperiment): experiment object that
                                              generated the data.
        """
        super().__init__(experiment)

        # Initialize sub experiments
        self._composite_expdata = [
            expr.__experiment_data__(expr) for expr in self._experiment._experiments
        ]

    def __repr__(self):
        line = 51 * "-"
        n_res = len(self._analysis_results)
        ret = line
        ret += f"\nExperiment: {self._experiment._type}"
        ret += f"\nExperiment ID: {self.experiment_id}"
        ret += "\nStatus: COMPLETE"
        ret += f"\nComponent Experiments: {len(self._composite_expdata)}"
        ret += f"\nCircuits: {len(self._data)}"
        ret += f"\nAnalysis Results: {n_res}"
        ret += "\n" + line
        if n_res:
            ret += "\nLast Analysis Result"
            for key, value in self._analysis_results[-1].items():
                ret += f"\n- {key}: {value}"
        return ret

    def component_experiment_data(self, index):
        """Return component experiment data"""
        return self._composite_expdata[index]

    def _add_single_data(self, data):
        """Add data to the experiment"""
        # TODO: Handle optional marginalizing IQ data
        metadata = data.get("metadata", {})
        if metadata.get("experiment_type") == self._experiment._type:

            # Add parallel data
            self._data.append(data)

            # Add marginalized data to sub experiments
            if "composite_clbits" in metadata:
                composite_clbits = metadata["composite_clbits"]
            else:
                composite_clbits = None
            for i, index in enumerate(metadata["composite_index"]):
                sub_data = {"metadata": metadata["composite_metadata"][i]}
                if "counts" in data:
                    if composite_clbits is not None:
                        sub_data["counts"] = marginal_counts(data["counts"], composite_clbits[i])
                    else:
                        sub_data["counts"] = data["counts"]
                self._composite_expdata[index].add_data(sub_data)
