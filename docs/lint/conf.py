# This is the configuration file to run sphinx linting for `tox -edocs-nitpick`.
# It will output warnings for each missing reference.

import sys, os

sys.path.insert(0, os.path.abspath("../"))
sys.path.append(os.path.abspath("../_ext"))
sys.path.append(os.path.abspath("../../"))

exclude_patterns = ["_build"]

project = "Qiskit Experiments"
nitpicky = True

# within nit-picking build, do not refer to any intersphinx object
intersphinx_mapping = {}

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "reno.sphinxext",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "autoref",
    "autodoc_experiment",
    "autodoc_analysis",
    "autodoc_visualization",
    "sphinx_design",
    "jupyter_sphinx",
    "jupyter_execute_custom",
]

nbsphinx_allow_errors = True

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
