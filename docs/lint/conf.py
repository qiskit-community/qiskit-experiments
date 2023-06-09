# This is the configuration file to run sphinx linting for `tox -edocs-nitpick`.
# It will output warnings for each missing reference.

import sys, os, datetime, subprocess

sys.path.insert(0, os.path.abspath("../"))
sys.path.append(os.path.abspath("../_ext"))
sys.path.append(os.path.abspath("../../"))

# Set env flag so that we can doc functions that may otherwise not be loaded
# see for example interactive visualizations in qiskit.visualization.
os.environ["QISKIT_DOCS"] = "TRUE"

# -- Project information -----------------------------------------------------
# The short X.Y version
version = "0.5"
# The full version, including alpha/beta/rc tags
release = "0.5.0"
project = f"Qiskit Experiments {version}"
copyright = f"2021-{datetime.date.today().year}, Qiskit Development Team"  # pylint: disable=redefined-builtin
author = "Qiskit Development Team"


project = "Qiskit Experiments"
nitpicky = True

# within nit-picking build, do not refer to any intersphinx object
intersphinx_mapping = {}

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx_copybutton",
    "jupyter_sphinx",
    "sphinx_autodoc_typehints",
    "reno.sphinxext",
    "sphinx_design",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "autoref",
    "autodoc_experiment",
    "autodoc_analysis",
    "autodoc_visualization",
    "jupyter_execute_custom",
]

# Minimal options to let the build run successfully
nbsphinx_timeout = 360
nbsphinx_execute = os.getenv("QISKIT_DOCS_BUILD_TUTORIALS", "never")
nbsphinx_widgets_path = ""
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
napoleon_custom_sections = [("data keys", "params_style"), ("style parameters", "params_style")]
autosummary_generate = True
autodoc_default_options = {"inherited-members": None}
numfig = True
numfig_format = {"table": "Table %s"}
language = "en"
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
pygments_style = "colorful"
add_module_names = False
modindex_common_prefix = ["qiskit_experiments."]
html_theme = "qiskit_sphinx_theme"  # use the theme in subdir 'theme'
html_context = {
    "analytics_enabled": True,
    "expandable_sidebar": True,
}
html_last_updated_fmt = "%Y/%m/%d"
html_theme_options = {
    "logo_only": True,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
}
autoclass_content = "both"
intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "qiskit": ("https://qiskit.org/documentation/", None),
    "uncertainties": ("https://pythonhosted.org/uncertainties", None),
}
if os.getenv("EXPERIMENTS_DEV_DOCS", None):
    rst_prolog = """
.. note::
    This is the documentation for the current state of the development branch
    of Qiskit Experiments. The documentation or APIs here can change prior to being
    released.
"""


def setup(app):
    app.connect("autodoc-skip-member", maybe_skip_member)


# Hardcoded list of class variables to skip in autodoc to avoid warnings
# Should come up with better way to address this

from qiskit_experiments.curve_analysis import ParameterRepr
from qiskit_experiments.curve_analysis import SeriesDef


def maybe_skip_member(app, what, name, obj, skip, options):
    skip_names = [
        "analysis",
        "set_run_options",
        "data_allocation",
        "labels",
        "shots",
        "x",
        "y",
        "y_err",
        "name",
        "filter_kwargs",
        "fit_func",
        "signature",
    ]
    skip_members = [
        ParameterRepr.repr,
        ParameterRepr.unit,
        SeriesDef.plot_color,
        SeriesDef.plot_symbol,
        SeriesDef.model_description,
        SeriesDef.canvas,
    ]
    if not skip:
        return (name in skip_names or obj in skip_members) and what == "attribute"
    return skip


###

# Ignore these objects
nitpick_ignore_regex = [
    ("py:.*", "qiskit.*"),
    ("py:.*", "numpy.*"),
    ("py:.*", "sklearn.*"),
    ("py:.*", "scipy.*"),
    ("py:.*", "datetime.*"),
    ("py:.*", "IBM.*"),
    ("py:.*", ".*\._.*"),
    ("py:.*", "_.*"),
    ("py:.*", "lmfit.*"),
    ("py:.*", "uncertainties.*"),
    ("py:.*", ".*__.*"),
    ("py:.*", "typing.*"),
    ("py:.*", ".*Error"),
    ("py:.*", "Ellipsis"),
]

# Deprecated objects that should be ignored in the release notes
nitpick_ignore_regex += [
    ("py:*", "MplCurveDrawer.*"),
    ("py:.*", "CliffordUtils.*"),
]
