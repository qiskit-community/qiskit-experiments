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
Quantum state tomography analysis
"""
from qiskit_experiments.framework import Options
from .basis import PauliMeasurementBasis
from .tomography_analysis import TomographyAnalysis


class StateTomographyAnalysis(TomographyAnalysis):
    """State tomography experiment analysis.

    # section: overview
        Fitter Functions

        Built-in fitter functions may be selected using the following string
        labels, refer to the corresponding functions documentation for additional
        details on the fitters.

        * ``"linear_inversion"``:
          :func:`~qiskit_experiments.library.tomography.fitters.linear_inversion` (Default)
        * ``"scipy_linear_lstsq"``:
          :func:`~qiskit_experiments.library.tomography.fitters.scipy_linear_lstsq`
        * ``"cvxpy_linear_lstsq"``:
          :func:`~qiskit_experiments.library.tomography.fitters.cvxpy_linear_lstsq`
        * ``"scipy_gaussian_lstsq"``:
          :func:`~qiskit_experiments.library.tomography.fitters.scipy_gaussian_lstsq`
        * ``"cvxpy_gaussian_lstsq"``:
          :func:`~qiskit_experiments.library.tomography.fitters.cvxpy_gaussian_lstsq`

        PSD Rescaling

        For fitters that do not constrain the reconstructed state to be
        `positive-semidefinite` (PSD) we construct the maximum-likelihood
        nearest PSD state under the assumption of Gaussian measurement noise
        using the rescaling method in Reference [1]. For fitters that already
        support PSD constraints this option can be disabled by setting
        ``rescale_positive=False``.

    # section: warning
        The API for tomography fitters is still under development so may change
        in future releases.

    # section: note
        Fitters starting with ``"cvxpy_*"`` require the optional CVXPY Python
        package to be installed.

    # section: reference
        .. ref_arxiv:: 1 1106.5458

    # section: see_also
        qiskit_experiments.library.tomography.tomography_analysis.TomographyAnalysis

    """

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options

        Analysis Options:
            measurement_basis (:class:`~basis.BaseFitterMeasurementBasis`): A custom measurement
                basis for analysis. By default the :meth:`experiment_options` measurement basis
                will be used.
            fitter (str or Callable): The fitter function to use for reconstruction.
                rescale_psd (bool): If True rescale the fitted state to be positive-semidefinite
                (Default: True).
            fitter_options (Dict[str, Any]): Additional kwargs will be supplied to the
                fitter function.
            rescale_trace (bool): If True rescale the state returned by the fitter have either
                trace 1 (Default: True).
            target (Union[str, :class:`~qiskit.quantum_info.DensityMatrix`,
                :class:`~qiskit.quantum_info.Statevector`]): Optional, et a custom target
                quantum state for computing the :func:~qiskit.quantum_info.state_fidelity`
                of the fitted state against (Default: None).
        """
        options = super()._default_options()
        options.measurement_basis = PauliMeasurementBasis()
        return options
