===============================
Visualization: Creating figures
===============================

The Visualization module provides plotting functionality for creating figures from experiment and analysis results.
This includes plotter and drawer classes to plot data in :py:class:`CurveAnalysis` and its subclasses.
Plotters inherit from :class:`BasePlotter` and define a type of figure that may be generated from
experiment or analysis data. For example, the results from :class:`CurveAnalysis` --- or any other
experiment where results are plotted against a single parameter (i.e., :math:`x`) --- can be plotted
using the :class:`CurvePlotter` class, which plots X-Y-like values.

These plotter classes act as a bridge (from the common bridge pattern in software development) between
analysis classes (or even users) and plotting backends such as Matplotlib. Drawers are the backends, with
a common interface defined in :class:`BaseDrawer`. Though Matplotlib is the only officially supported
plotting backend in Qiskit Experiments (i.e., through :class:`MplDrawer`), custom drawers can be
implemented by users to use alternative backends. As long as the backend is a subclass of
:class:`BaseDrawer`, and implements all the necessary functionality, all plotters should be able to
generate figures with the alternative backend.

Data to feed into the module is split into `series` and `supplementary`:

- **Series Data**: Values from experiment data or analysis instances, or values to be plotted as points,
  lines, etc. A good rule-of-thumb: if it could have a legend entry, it's series data.
- **Supplementary Data**: Values unrelated to a series or curve, only related to the figure. Examples
  include fit-reports, figure-wide text, or metadata.

You can think of the structure of series and supplementary data as dictionaries with the data-keys and
series-names. This is an example for CurvePlotter, representing the dummy data we generated.

.. code-block:: python

    series_data = {
        "A": {                      # Series-name 'A'
            "x": ...,               # Data for data-key 'x' and series 'A'
            "y": ...,
            "x_interp": ...,
            "y_interp": ...,
            "y_interp_err": ...,
        },
        "B": {                      # Series-name 'B'
            "x": ...,               # Data for data-key 'x' and series 'B'
            "y": ...,
            "x_interp": ...,
            "y_interp": ...,
        }
    }


.. code-block:: python

    supplementary_data = {
        "primary_results": ...,     # Supplementary data only has a data-key, no series-name.
        "fit_red_chi": ...,
    }


Data consists of the ``{'y_interp_err', 'x', 'x_interp', 'y_interp', 'y'}`` keys.


Generating a figure using a plotter
===================================

.. code-block:: python

    # Create plotter and set options and style.
    plotter = CurvePlotter(MplDrawer())
    plotter.set_options(
        plot_sigma=[
            (1.0, 0.5)
        ],  # Controls confidence-intervals for `y_interp_err` data-keys.
    )
    plotter.set_figure_options(
        series_params={
            "A": {"symbol": "o", "color": "C0", "label": "Qubit 0"},
            "B": {"symbol": "X", "color": "C1", "label": "Qubit 1"},
            "C": {"symbol": "v", "color": "k", "label": "Ideal 0"},
            "D": {"symbol": "^", "color": "k", "label": "Ideal 1"},
        },
        xlabel="Parameter",
        ylabel="${\\langle{}Z\\rangle{}}$",
        figure_title="Expectation Values",
    )
