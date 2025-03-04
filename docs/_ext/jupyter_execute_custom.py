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
import os
import time

from jupyter_sphinx.execute import ExecuteJupyterCells
from jupyter_sphinx import JupyterCell
from sphinx.application import Sphinx
from sphinx.util import logging


logger = logging.getLogger(__name__)

class JupyterCellCheckEnv(JupyterCell):
    """This class overrides the JupyterCell class in :mod:`jupyter-sphinx`
    to skip cell execution when `QISKIT_DOCS_SKIP_EXECUTE` is true in the environment.
    """

    def run(self):
        [cell] = super().run()
        if os.getenv("QISKIT_DOCS_SKIP_EXECUTE", False):
            cell["execute"] = False
            cell["hide_code"] = False
        return [cell]


class TimedExecuteJupyterCells(ExecuteJupyterCells):
    def apply(self):
        start_time = time.perf_counter()
        super().apply()
        execution_time = time.perf_counter() - start_time
        if execution_time > 1:
            # Only log for significant duration since this runs on every
            # document, even ones without jupyter content.
            logger.info(
                f"Jupyter execution in {self.env.docname} took {execution_time:.2f} seconds"
            )


def setup(app: Sphinx):
    app.add_directive("jupyter-execute", JupyterCellCheckEnv, override=True)
    app.registry.transforms.remove(ExecuteJupyterCells)
    app.add_transform(TimedExecuteJupyterCells)
    return {"parallel_read_safe": True}
