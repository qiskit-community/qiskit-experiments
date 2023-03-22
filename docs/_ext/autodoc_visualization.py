# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Documentation extension for visualization classes.
"""

from typing import Any

from docs._ext.custom_styles.styles import VisualizationDocstring
from qiskit.exceptions import QiskitError
from qiskit_experiments.visualization import BasePlotter, BaseDrawer
from sphinx.application import Sphinx
from sphinx.ext.autodoc import ClassDocumenter


class VisualizationDocumenter(ClassDocumenter):
    """Sphinx extension for the custom documentation of the standard visualization classes."""

    objtype = "visualization"  # Must be overwritten by subclasses.
    directivetype = "class"
    priority = 10 + ClassDocumenter.priority
    option_spec = dict(ClassDocumenter.option_spec)

    def add_content(self, more_content: Any, no_docstring: bool = False) -> None:
        sourcename = self.get_sourcename()
        try:
            class_doc, init_doc = self.get_doc()
        except ValueError:
            raise QiskitError(
                f"Documentation of {self.name} doesn't match with the expected format."
                "Please run sphinx build without using the visualization template."
            )

        # format visualization class documentation into the visualization style
        class_doc_parser = VisualizationDocstring(
            target_cls=self.object,
            docstring_lines=class_doc,
            config=self.env.app.config,
            indent=self.content_indent,
        )

        # write introduction
        for i, line in enumerate(self.process_doc(class_doc_parser.generate_class_docs())):
            self.add_line(line, sourcename, i)
        self.add_line("", sourcename)

        # write init method documentation
        self.add_line(".. rubric:: Initialization", sourcename)
        self.add_line("", sourcename)
        for i, line in enumerate(self.process_doc([init_doc])):
            self.add_line(line, sourcename, i)
        self.add_line("", sourcename)

        # method and attributes
        if more_content:
            for line, src in zip(more_content.data, more_content.items):
                self.add_line(line, src[0], src[1])


class PlotterDocumenter(VisualizationDocumenter):
    """Sphinx extension for the custom documentation of plotter classes."""

    objtype = "plotter"

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any) -> bool:
        return isinstance(member, BasePlotter)


class DrawerDocumenter(VisualizationDocumenter):
    """Sphinx extension for the custom documentation of drawer classes."""

    objtype = "drawer"

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any) -> bool:
        return isinstance(member, BaseDrawer)


def setup(app: Sphinx):
    # Add plotter documenter
    existing_documenter = app.registry.documenters.get(PlotterDocumenter.objtype)
    if existing_documenter is None or not issubclass(existing_documenter, PlotterDocumenter):
        app.add_autodocumenter(PlotterDocumenter, override=True)

    # Add drawer documenter
    existing_documenter = app.registry.documenters.get(DrawerDocumenter.objtype)
    if existing_documenter is None or not issubclass(existing_documenter, DrawerDocumenter):
        app.add_autodocumenter(DrawerDocumenter, override=True)
    return {"parallel_read_safe": True}
