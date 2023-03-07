# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Customizations of :mod:`jupyter-sphinx`.
"""
from jupyter_sphinx import JupyterCell
from sphinx.application import Sphinx
import os


class JupyterCellCheckEnv(JupyterCell):
    """This class overrides the JupyterCell class in :mod:`jupyter-sphinx`
    to skip cell execution when `QISKIT_DOCS_SKIP_RST` is true in the environment.
    """

    def run(self):
        [cell] = super().run()
        if os.getenv("QISKIT_DOCS_SKIP_RST", False):
            cell["execute"] = False
            cell["hide_code"] = False
        return [cell]


def setup(app: Sphinx):
    app.add_directive("jupyter-execute", JupyterCellCheckEnv, override=True)
