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
Quantum process tomography analysis
"""
from qiskit_experiments.framework import Options
from .basis import PauliMeasurementBasis, PauliPreparationBasis
from .tomography_analysis import TomographyAnalysis


class ProcessTomographyAnalysis(TomographyAnalysis):
    """Quantum state and process tomography experiment analysis.

    # section overview:
    
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

    # section: note
        Fitters starting with ``"cvxpy_*"`` require the optional CVXPY Python
        package to be installed.
    
    # section: warning
        The API for tomography fitters is still under development so may change
        in future releases.
    
    # section: reference
        .. ref_arxiv:: 1 1106.5458

    """

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options

        Analysis Options:
            measurement_basis
                (:class:`~qiskit_experiments.library.tomography.basis.BaseFitterMeasurementBasis`):
                The measurement
                :class:`~qiskit_experiments.library.tomography.basis.BaseFitterMeasurementBasis`
                to use for tomographic reconstruction when running a
                :class:`~qiskit_experiments.library.tomography.StateTomography` or
                :class:`~qiskit_experiments.library.tomography.ProcessTomography`.
            preparation_basis
                (:class:`~qiskit_experiments.library.tomography.basis.BaseFitterPreparationBasis`):
                The preparation
                :class:`~qiskit_experiments.library.tomography.basis.BaseFitterPreparationBasis`
                to use for tomographic reconstruction for
                :class:`~qiskit_experiments.library.tomography.ProcessTomography`.
            fitter (str or Callable): The fitter function to use for reconstruction.
                This can  be a string to select one of the built-in fitters, or a callable to
                supply a custom fitter function. See the `Fitter Functions` section for
                additional information.
            rescale_positive (bool): If True rescale the state returned by the fitter
                to be positive-semidefinite. See the `PSD Rescaling` section for additional information
                (Default: True).
            rescale_trace (bool): If True rescale the state returned by the fitter have either trace 1
                for :class:`~qiskit.quantum_info.DensityMatrix`, or trace dim for
                :class:`~qiskit.quantum_info.Choi` matrices (Default: True).
            target (str or :class:`~qiskit.quantum_info..operators.channel.quantum_channel`
                or :class:`~qiskit.quantum_info.Operator`): Set a custom target quantum
                channel for computing the :func:~qiskit.quantum_info.process_fidelity` of the
                fitted process against. If ``"default"`` the ideal process corresponding for
                the input circuit will be used. If ``None`` no fidelity will be computed
                (Default: "default").
            kwargs: will be supplied to the fitter function, for documentation of available args
                refer to the fitter function documentation.

        """
        options = super()._default_options()

        options.measurement_basis = PauliMeasurementBasis()
        options.preparation_basis = PauliPreparationBasis()
        options.fitter = "linear_inversion"
        options.rescale_positive = True
        options.rescale_trace = True
        options.target = "default"

        return options
