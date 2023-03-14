import sys, os

# This is the configuration file to run sphinx linting for `tox -edocs-nitpick`.
# It will output warnings for each missing reference.

sys.path.insert(0, os.path.abspath("../"))
sys.path.append(os.path.abspath("../_ext"))
sys.path.append(os.path.abspath("../../"))

exclude_patterns = ["_build"]

# # Set env flag so that we can doc functions that may otherwise not be loaded
# # see for example interactive visualizations in qiskit.visualization.
# os.environ["QISKIT_DOCS"] = "TRUE"

project = "Qiskit Experiments"
nitpicky = True

# within nit-picking build, do not refer to any intersphinx object
intersphinx_mapping = {}

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "jupyter_sphinx",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "autoref",
    "autodoc_experiment",
    "autodoc_analysis",
    "autodoc_visualization",
    "jupyter_execute_custom",
]

nbsphinx_allow_errors = True

# Ignore these objects
nitpick_ignore_regex = [
    ("py:.*", "qiskit.*"),
    ("py:.*", ".*\._.*"),
    ("py:.*", "_.*"),
    ("py:.*", "numpy.*"),
    ("py:.*", ".*__.*"),
    ("py:.*", "typing.*"),
    ("py:.*", ".*Error"),
    ("py:.*", "Ellipsis"),
]

# Deprecated objects that should be ignored in the release notes
nitpick_ignore_regex += [
    ("py:class", "MplCurveDrawer"),
    ("py:.*", "CliffordUtils.*"),
]
