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
Curve analysis result data class.
"""
from qiskit_experiments.experiment_data import AnalysisResultData


class CurveAnalysisResultData(AnalysisResultData):
    """Analysis data container for curve fit analysis.

    Class Attributes:
        __keys_not_shown__: Data keys of analysis result which are not directly shown
            in `__str__` method. By default, `pcov` (covariance matrix),
            `raw_data` (raw x, y, sigma data points), `popt`, `popt_keys`, and `popt_err`
            are not displayed. Fit parameters (popt) are formatted to

            .. code-block::

                p0 = 1.2 ± 0.34
                p1 = 5.6 ± 0.78

            rather showing raw key-value pairs

            .. code-block::

                popt_keys = ["p0", "p1"]
                popt = [1.2, 5.6]
                popt_err = [0.34, 0.78]

            The covariance matrix and raw data points are not shown because they output
            very long string usually doesn't fit in with the summary of the analysis object,
            i.e. user wants to quickly get the over view of fit values and goodness of fit,
            such as the chi-squared value and computer evaluated quality.

            However these non-displayed values are still kept and user can access to
            these values with `result["raw_data"]` and `result["pcov"]` if necessary.
    """

    __keys_not_shown__ = "pcov", "raw_data", "popt", "popt_keys", "popt_err"

    def __str__(self):
        out = ""

        if self.get("success"):
            popt_keys = self.get("popt_keys")
            popt = self.get("popt")
            popt_err = self.get("popt_err")

            for key, value, error in zip(popt_keys, popt, popt_err):
                out += f"\n  - {key}: {value} \u00B1 {error}"
        out = str(super()) + out

        return out

    # def get_opt_value(self, param_name: str) -> float:
    #     """A helper function to get parameter value from a result dictionary.

    #     Args:
    #         param_name: Name of parameter to extract.

    #     Returns:
    #         Parameter value.

    #     Raises:
    #         KeyError:
    #             - When the result does not contain parameter information.
    #         ValueError:
    #             - When specified parameter is not defined.
    #     """
    #     try:
    #         index = self["popt_keys"].index(param_name)
    #         return self["popt"][index]
    #     except KeyError as ex:
    #         raise KeyError(
    #             "Input result has not fit parameter information. "
    #             "Please confirm if the fit is successfully completed."
    #         ) from ex
    #     except ValueError as ex:
    #         raise ValueError(f"Parameter {param_name} is not defined.") from ex

    # def get_opt_error(self, param_name: str) -> float:
    #     """A helper function to get error value from analysis result.

    #     Args:
    #         param_name: Name of parameter to extract.

    #     Returns:
    #         Parameter error value.

    #     Raises:
    #         KeyError:
    #             - When the result does not contain parameter information.
    #         ValueError:
    #             - When specified parameter is not defined.
    #     """
    #     try:
    #         index = self["popt_keys"].index(param_name)
    #         return self["popt_err"][index]
    #     except KeyError as ex:
    #         raise KeyError(
    #             "Input result has not fit parameter information. "
    #             "Please confirm if the fit is successfully completed."
    #         ) from ex
    #     except ValueError as ex:
    #         raise ValueError(f"Parameter {param_name} is not defined.") from ex
