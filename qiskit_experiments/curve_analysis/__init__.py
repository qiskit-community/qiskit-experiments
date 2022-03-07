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

r"""
=========================================================
Curve Analysis (:mod:`qiskit_experiments.curve_analysis`)
=========================================================

.. currentmodule:: qiskit_experiments.curve_analysis

Curve analysis provides the analysis base class for a variety of experiments with 1-D parameter scan.
Subclasses can override several class attributes to define the behavior of the
data formatting and fitting. Here we describe how code developers can create new curve fit
analysis inheriting from the base class.


Overview
========

The base class :class:`CurveAnalysis` supports multi-objective optimization on
different sets of experiment results, and you can also define multiple independent
optimization tasks in the same class. The analysis is implemented with the following data model.

- Group: This is top level component of the fitting. If an analysis defines
  multiple groups, it performs multiple independent optimizations
  and generates results for every optimization group.

- Series: This is a collection of curves to form a multi-objective optimization task.
  The fit entries in the same series share the fit parameters,
  and multiple experimental results are simultaneously fit to generate a single fit result.

- Curve: This is a single entry of analysis. Every curve may take unique filter keywords
  to extract corresponding (x, y) data from the whole experimental results,
  along with the callback function used for the curve fitting.

To manage this structure, curve analysis provides a special dataclass :class:`SeriesDef`
that represents an optimization configuration for a single curve data.
Based on this information, the analysis automatically constructs proper optimization logic.
Thus one can avoid writing boilerplate code in various curve analyses
and quickly write up the analysis code for a particular experiment.
This analysis generates a set of :class:`~qiskit_experiments.framework.AnalysisResultData`
entries with a single Matplotlib plot of the fit curves with raw data points.

.. _curve_analysis_define_new:

Defining new curves
===================

You can intuitively write the definition of a new curve, as shown below:

.. code-block:: python3

    from qiskit_experiments.curve_analysis import SeriesDef, fit_function

    SeriesDef(
        fit_func=lambda x, p0, p1, p2: fit_function.exponential_decay(
            x, amp=p0, lamb=p1, baseline=p2
        ),
        model_description="p0 * exp(-p1 * x) + p2",
    )

The minimum field you must fill with is the ``fit_func``, which is a callback function used
with the optimization solver. Here you must call one of the fit functions from the module
:mod:`qiskit_experiments.curve_analysis.fit_function` because they implement
special logic to compute error propagation.
Note that argument name of the fit function, i.e. ``[p0, p1, p2]``, is important because
the signature of the provided fit function is parsed behind the scenes and
used as a parameter name of the analysis result instance.
Thus, this name may be used to populate your experiment database with the result.

Optionally you can set ``model_description`` which is a string representation of your
fitting model that will be passed to the analysis result as a part of metadata.
This instance should be set to :attr:`CurveAnalysis.__series__` as a python list.

For multi-objective optimization, i.e. if you have more than two curves that you
want to optimize simultaneously, you can create a list consisting of multiple curve entries.
In this case, the curves defined in the series definition form a single composite function
and the analysis solves the optimization problem:

.. math::

    \Theta_{\mbox{opt}} = \arg\min_\Theta \sum_i \sigma_i^{-2} (f_i(x_i, \theta_i) -  y_i)^2,

where :math:`\Theta = \{\theta_0, \theta_1, ..., \theta_N\} \in \mathbb{R}` and
this analysis has multiple fit functions each defined in the :attr:`SeriesDef.fit_func`
:math:`f_i(x_i, \theta_i)` with fit parameters :math:`\theta_i`.
Note that each fit model can take different parameters distinguished by function argument name
:math:`\theta_i = \{ p_{i0}, p_{i1}, ..., p_{iM} \}`.
Now we run a set of experiments that scans experiment parameters :math:`x_i`
and measure the outcomes :math:`y_i` with uncertainties :math:`\sigma_i`.
In the analysis, the solver will find the parameters :math:`\Theta_{\mbox{opt}}`
that simultaneously minimize the chi-squared values of all fit models defined in the series.
Here is an example how to implement such multi-objective optimization task:

.. code-block:: python3

    [
        SeriesDef(
            name="my_experiment1",
            fit_func=lambda x, p0, p1, p3: fit_function.exponential_decay(
                x, amp=p0, lamb=p1, baseline=p3
            ),
            filter_kwargs={"tag": 1},
            plot_color="red",
            plot_symbol="^",
        ),
        SeriesDef(
            name="my_experiment2",
            fit_func=lambda x, p0, p2, p3: fit_function.exponential_decay(
                x, amp=p0, lamb=p2, baseline=p3
            ),
            filter_kwargs={"tag": 2},
            plot_color="blue",
            plot_symbol="o",
        ),
    ]

Note that now you also need to provide ``name`` and ``filter_kwargs`` to
distinguish the entries and filter the corresponding (x, y) data from the experiment results.
Optionally, you can provide ``plot_color`` and ``plot_symbol`` to visually
separate two curves in the plot. In this model, you have 4 parameters ``[p0, p1, p2, p3]``
and the two curves share ``p0`` (``p3``) for ``amp`` (``baseline``) of
the :func:`exponential_decay` fit function.
Here one should expect the experiment results will have two classes of data with metadata
``"tag": 1`` and ``"tag": 2`` for ``my_experiment1`` and ``my_experiment2``, respectively.

By using this model, one can flexibly set up your fit model. Here is another example:

.. code-block:: python3

    [
        SeriesDef(
            name="my_experiment1",
            fit_func=lambda x, p0, p1, p2, p3: fit_function.cos(
                x, amp=p0, freq=p1, phase=p2, baseline=p3
            ),
            filter_kwargs={"tag": 1},
            plot_color="red",
            plot_symbol="^",
        ),
        SeriesDef(
            name="my_experiment2",
            fit_func=lambda x, p0, p1, p2, p3: fit_function.sin(
                x, amp=p0, freq=p1, phase=p2, baseline=p3
            ),
            filter_kwargs={"tag": 2},
            plot_color="blue",
            plot_symbol="o",
        ),
    ]

You have the same set of fit parameters for two curves, but now you fit two datasets
with different trigonometric functions.

.. _curve_analysis_fixed_param:

Fitting with fixed parameters
=============================

You can also fix certain parameters during the curve fitting by specifying
parameter names in the class attribute :attr:`CurveAnalysis.__fixed_parameters__`.
This feature is useful especially when you want to define a subclass of
a particular analysis class.

.. code-block:: python3

    class AnalysisA(CurveAnalysis):

        __series__ = [
            SeriesDef(
                fit_func=lambda x, p0, p1, p2: fit_function.exponential_decay(
                    x, amp=p0, lamb=p1, baseline=p2
                ),
            ),
        ]

    class AnalysisB(AnalysisA):

        __fixed_parameters__ = ["p0"]

        @classmethod
        def _default_options(cls) -> Options:
            options = super()._default_options()
            options.p0 = 3

            return options

The parameter specified in :attr:`CurveAnalysis.__fixed_parameters__` should be provided
via the analysis options. Thus you may need to define a default value of the parameter in the
:meth:`CurveAnalysis._default_options`.
This code will give you identical fit model to the one defined in the following class:

.. code-block:: python3

    class AnalysisB(CurveAnalysis):

        __series__ = [
            SeriesDef(
                fit_func=lambda x, p1, p2: fit_function.exponential_decay(
                    x, amp=3, lamb=p1, baseline=p2
                ),
            ),
        ]

However, note that you can also inherit other features, e.g. the algorithm to
generate initial guesses, from the :class:`AnalysisA` in the first example.
On the other hand, in the latter case, you need to manually copy and paste
every logic defined in the :class:`AnalysisA`.

.. _curve_analysis_multiple_tasks:

Defining multiple tasks
=======================

The code below shows how a subclass can define separate optimization tasks.

.. code-block:: python3

    [
        SeriesDef(
            name="my_experiment1",
            fit_func=lambda x, p0, p1, p2, p3: fit_function.cos(
                x, amp=p0, freq=p1, phase=p2, baseline=p3
            ),
            filter_kwargs={"tag": 1},
            plot_color="red",
            plot_symbol="^",
            group="cos",
        ),
        SeriesDef(
            name="my_experiment2",
            fit_func=lambda x, p0, p1, p2, p3: fit_function.sin(
                x, amp=p0, freq=p1, phase=p2, baseline=p3
            ),
            filter_kwargs={"tag": 2},
            plot_color="blue",
            plot_symbol="o",
            group="sin",
        ),
    ]

The code looks almost identical to one in :ref:`curve_analysis_define_new`,
however, here we are providing a unique ``group`` value to each series definition.
In this configuration, the parameters ``[p0, p1, p2, p3]`` are not shared among
underlying curve fittings, thus we will get two fit parameter sets as a result.
This means any fit parameter value may change between curves.
The parameters can be distinguished by the ``group`` value passed to the result metadata.

This is identical to running individual ``my_experiment1`` and ``my_experiment2`` as a
:class:`~qiskit_experiments.framework.BatchExperiment` and collect fit results afterwards
in the analysis class attached to the batch experiment instance.

.. code-block:: python3

    from qiskit_experiments.framework import BatchExperiment

    exp1 = MyExperiment1(...)
    exp2 = MyExperiment2(...)

    batch_exp = BatchExperiment([exp1, exp2])
    batch_exp.analysis = MyAnalysis(...)

However, this may require a developer to write many classes, for example,
here you may want to implement :class:`MyAnalysis` analysis class in addition to the
analysis classes for :class:`MyExperiment1` and :class:`MyExperiment2`.
On the other hand, using ``group`` feature allows you to complete the same analysis
within a single class instance.

.. _curve_analysis_format:

Pre-processing the fit data
===========================

A subclass may override :meth:`CurveAnalysis._format_data` to perform custom pre-processing
on experiment data before computing the initial guesses.
Here a subclass may perform data smoothing, removal of outliers, etc...
By default, it performs averaging of y values over the same x values,
followed by the data sort by x values.
This method should return a :class:`CurveData` instance with `label="fit_ready"`.

.. _curve_analysis_init_guess:

Providing initial guesses and boundaries
========================================

A template for initial guesses and boundaries is automatically generated in
:attr:`CurveAnalysis.options` as a dictionary keyed on the parameter names parsed from
the series definition. The default values are set to ``None``.
The list of parameter names is also available in the property
:attr:`CurveAnalysis.fit_params`.

A developer of the curve analysis subclass is recommended to override
:meth:`CurveAnalysis._generate_fit_guesses` to provide systematic guesses and boundaries
based on the experimental result.
For accessing the formatted experiment result, you can use the :meth:`CurveAnalysis._data` method.

.. code-block:: python3

    curve_data = self._data(series_name="my_experiment1")

    x = curve_data.x  # you can get x-values
    y = curve_data.y  # you can get y-values

In addition, there are several common initial guess estimators available in
:mod:`qiskit_experiments.curve_analysis.guess`.

When fit is performed without any prior information of parameters, it usually
falls into unsatisfactory result. This method is called with :class:`FitOptions`
instance which is dict-like object. This class implements convenient methods to
manage conflict with user provided values, i.e. user provided values have higher priority,
thus systematically generated values cannot override user values.

.. code-block:: python3

    opt1 = user_opt.copy()
    opt1.p0.set_if_empty(p0=3)
    opt1.bounds = set_if_empty(p0=(0, 10))
    opt1.add_extra_options(method="lm")

    opt2 = user_opt.copy()
    opt2.p0.set_if_empty(p0=4)

    return [opt1, opt2]

``user_opt`` is a :class:`FitOptions` instance, which consists of sub-dictionaries for
initial guesses (``.p0``) and boundaries (``.bounds``).
The :meth:`.set_if_empty` method overrides the parameter value only when the user doesn't provide
any prior information.
``user_opt`` also has extra configuration dictionary that is directly passed to
the curve fitting function. Note that the :class:`CurveAnalysis` uses
SciPy `curve_fit`_ function as a core solver. See the API documentation for available options.

The final fitting outcome is determined with the following procedure.

1. ``user_opt`` is initialized with the values provided by the user via the analysis options.

2. The algorithmic guess is generated in :meth:`_generate_fit_guesses`,
   where the logic implemented by a subclass may override the ``user_opt``.
   If you want, you can copy it to create multiple fitting configurations.
   When multiple configurations are generated here, the curve fitter runs fitting multiple times.

3. If multiple configurations are created, the curve analysis framework checks
   duplication of configurations and performs fitting multiple times with a unique configuration set.

4. The curve fitter computes a reduced chi-squared value for every attempt,
   and finds the outcome with the minimum reduced chi-squared value.
   If the fitting fails, or the solver cannot find reasonable parameters within the maximum recursion,
   it just ignores the current configuration and moves to the next.
   If all provided configurations fail, it raises ``UserWarning`` and continues
   the rest of the analysis.

5. Analysis results are automatically generated if the curve fitter
   successfully finds the best-fit outcome.

.. _curve_fit: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

.. _curve_analysis_evaluate:

Evaluating fit quality
======================

A subclass can override :meth:`CurveAnalysis._evaluate_quality` method to provide an algorithm to
evaluate quality of the fitting. This method is called with the :class:`FitData` object
which contains fit parameters and the reduced chi-squared value.
Qiskit Experiments often uses the empirical condition chi-squared < 3 as a goodness of fitting.

.. _curve_analysis_new_quantity:

Computing new quantity with fit parameters
==========================================

Once the best fit parameters are found, the :meth:`CurveAnalysis._extra_database_entry` method is
called with the same :class:`FitData` object.
You can compute new quantities by combining multiple fit parameters.

.. code-block:: python3

    from qiskit_experiments.framework import AnalysisResultData

    p0 = fit_data.fitval("p0")
    p1 = fit_data.fitval("p1")

    extra_entry = AnalysisResultData(
        name="p01",
        value=p0 * p1,
    )

Note that both ``p0`` and ``p1`` are `ufloat`_ object consisting of
a nominal value and an error value which assumes the standard deviation.
Since this object natively supports error propagation, you don't need to manually compute errors.

.. _ufloat: https://pythonhosted.org/uncertainties/user_guide.html

.. _curve_analysis_saved_entry:

Managing fit parameters to be saved in the database
===================================================

By default :class:`CurveAnalysis` only stores a single entry ``@Parameters_<name_of_class>``.
This entry consists of a value which is a list of all fitting parameters
with extra metadata involving their covariance matrix.
If you want to save a particular parameter as a standalone entry,
you can override the ``result_parameters`` option of the analysis.
By using :class:`ParameterRepr` representation, you can rename the parameter in the database.

.. code-block:: python3

    from qiskit_experiments.curve_analysis import ParameterRepr

    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.result_parameters = [ParameterRepr("p0", "amp", "Hz")]

        return options

Here the first argument ``p0`` is the target parameter defined in the series definition,
``amp`` is the representation of ``p0`` in the database, and ``Hz`` is the unit of the value
which might be optionally provided.


If there is any missing feature you can write a feature request as an issue in our
`GitHub <https://github.com/Qiskit/qiskit-experiments/issues>`_.


Classes
=======

These are the base class and internal data structures to implement a curve analysis.

.. autosummary::
    :toctree: ../stubs/

    CurveAnalysis
    SeriesDef
    CurveData
    FitData
    ParameterRepr
    FitOptions

Standard Analysis
=================

These classes provide typical analysis functionality.
These are expected to be reused in multiple experiments.
By overriding default options from the class method :meth:`_default_analysis_options` of
your experiment class, you can still tailor the standard analysis classes to your experiment.

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/analysis.rst

    DecayAnalysis
    DumpedOscillationAnalysis
    OscillationAnalysis
    ResonanceAnalysis
    GaussianAnalysis
    ErrorAmplificationAnalysis

Functions
=========

These are the helper functions to realize a part of curve fitting functionality.

Curve Fitting
*************

.. autosummary::
    :toctree: ../stubs/

    curve_fit
    multi_curve_fit

Fit Functions
*************
.. autosummary::
    :toctree: ../stubs/

    fit_function.cos
    fit_function.cos_decay
    fit_function.exponential_decay
    fit_function.gaussian
    fit_function.sqrt_lorentzian
    fit_function.sin
    fit_function.sin_decay
    fit_function.bloch_oscillation_x
    fit_function.bloch_oscillation_y
    fit_function.bloch_oscillation_z

Initial Guess
*************
.. autosummary::
    :toctree: ../stubs/

    guess.constant_sinusoidal_offset
    guess.constant_spectral_offset
    guess.exp_decay
    guess.full_width_half_max
    guess.frequency
    guess.max_height
    guess.min_height
    guess.oscillation_exp_decay

Visualization
*************
.. autosummary::
    :toctree: ../stubs/

    plot_curve_fit
    plot_errorbar
    plot_scatter

Utilities
*********
.. autosummary::
    :toctree: ../stubs/

    is_error_not_significant
"""
from .curve_analysis import CurveAnalysis, is_error_not_significant
from .curve_data import CurveData, SeriesDef, FitData, ParameterRepr, FitOptions
from .curve_fit import (
    curve_fit,
    multi_curve_fit,
    process_curve_data,
    process_multi_curve_data,
)
from .visualization import plot_curve_fit, plot_errorbar, plot_scatter, FitResultPlotters
from . import guess
from . import fit_function

# standard analysis
from .standard_analysis import (
    DecayAnalysis,
    DumpedOscillationAnalysis,
    OscillationAnalysis,
    ResonanceAnalysis,
    GaussianAnalysis,
    ErrorAmplificationAnalysis,
)
