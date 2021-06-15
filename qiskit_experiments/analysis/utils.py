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


import numpy as np
from scipy import signal
from qiskit_experiments.experiment_data import AnalysisResult
from qiskit_experiments.exceptions import AnalysisError


def get_opt_value(analysis_result: AnalysisResult, param_name: str) -> float:
    """A helper function to get parameter value from analysis result.

    Args:
        analysis_result: Analysis result object.
        param_name: Name of parameter to extract.

    Returns:
        Parameter value.

    Raises:
        KeyError:
            - When analysis result does not contain parameter information.
        ValueError:
            - When specified parameter is not defined.
    """
    try:
        index = analysis_result["popt_keys"].index(param_name)
        return analysis_result["popt"][index]
    except KeyError as ex:
        raise KeyError(
            "Input analysis result has no valid fit parameter information. "
            "Please confirm if the fit is successfully completed."
        ) from ex
    except ValueError as ex:
        raise ValueError(f"Parameter {param_name} is not defined.") from ex


def get_opt_error(analysis_result: AnalysisResult, param_name: str) -> float:
    """A helper function to get error value from analysis result.

    Args:
        analysis_result: Analysis result object.
        param_name: Name of parameter to extract.

    Returns:
        Parameter error value.

    Raises:
        KeyError:
            - When analysis result does not contain parameter information.
        ValueError:
            - When specified parameter is not defined.
    """
    try:
        index = analysis_result["popt_keys"].index(param_name)
        return analysis_result["popt_err"][index]
    except KeyError as ex:
        raise KeyError(
            "Input analysis result has no valid fit parameter information. "
            "Please confirm if the fit is successfully completed."
        ) from ex
    except ValueError as ex:
        raise ValueError(f"Parameter {param_name} is not defined.") from ex


def frequency_guess(
        x_values: np.ndarray,
        y_values: np.ndarray,
        method: str = "FFT",
) -> float:
    """Provide initial frequency guess.

    Args:
        x_values: Array of x values.
        y_values: Array of y values.
        method: A method to find signal frequency. See below for details.

    Methods
        - ``ACF``: Calculate autocorrelation function with numpy and run scipy peak search.
          Frequency is calculated based on x coordinate of the first peak.
        - ``FFT``: Use numpy fast Fourier transform to find signal frequency.

    """
    if method == "ACF":
        corr = np.correlate(y_values, y_values, mode="full")
        corr = corr[corr.size//2:]
        peak_inds, _ = signal.find_peaks(corr)
        if len(peak_inds) == 0:
            return 0
        return 1 / x_values[peak_inds[0]]

    if method == "FFT":
        y_values = y_values - np.mean(y_values)
        fft_data = np.fft.fft(y_values)
        fft_freqs = np.fft.fftfreq(len(x_values), float(np.mean(np.diff(x_values))))
        main_freq_arg = np.argmax(np.abs(fft_data))
        f_guess = np.abs(fft_freqs[main_freq_arg])
        return f_guess

    raise AnalysisError(
        f"The specified method {method} is not available in frequency guess function."
    )
