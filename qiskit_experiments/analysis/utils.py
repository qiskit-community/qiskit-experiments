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

"""Analysis utility functions."""


from qiskit_experiments.experiment_data import ResultDict


def get_opt_value(result_data: ResultDict, param_name: str) -> float:
    """A helper function to get parameter value from a result dictionary.

    Args:
        result_data: Result object.
        param_name: Name of parameter to extract.

    Returns:
        Parameter value.

    Raises:
        KeyError:
            - When the result does not contain parameter information.
        ValueError:
            - When specified parameter is not defined.
    """
    try:
        index = result_data["popt_keys"].index(param_name)
        return result_data["popt"][index]
    except KeyError as ex:
        raise KeyError(
            "Input result has not fit parameter information. "
            "Please confirm if the fit is successfully completed."
        ) from ex
    except ValueError as ex:
        raise ValueError(f"Parameter {param_name} is not defined.") from ex


def get_opt_error(result_data: ResultDict, param_name: str) -> float:
    """A helper function to get error value from analysis result.

    Args:
        result_data: Result object.
        param_name: Name of parameter to extract.

    Returns:
        Parameter error value.

    Raises:
        KeyError:
            - When the result does not contain parameter information.
        ValueError:
            - When specified parameter is not defined.
    """
    try:
        index = result_data["popt_keys"].index(param_name)
        return result_data["popt_err"][index]
    except KeyError as ex:
        raise KeyError(
            "Input result has not fit parameter information. "
            "Please confirm if the fit is successfully completed."
        ) from ex
    except ValueError as ex:
        raise ValueError(f"Parameter {param_name} is not defined.") from ex
