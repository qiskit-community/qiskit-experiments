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
Documentation extension for analysis class.
"""

from typing import Any

from docs._ext.custom_styles.styles import AnalysisDocstring
from qiskit_experiments.framework.base_analysis import BaseAnalysis
from sphinx.application import Sphinx
from sphinx.ext.autodoc import ClassDocumenter


class AnalysisDocumenter(ClassDocumenter):
    """Sphinx extension for the custom documentation of the standard analysis class."""

    objtype = "analysis"
    directivetype = 'class'
    priority = 10 + ClassDocumenter.priority
    option_spec = dict(ClassDocumenter.option_spec)

    @classmethod
    def can_document_member(
        cls, member: Any, membername: str, isattr: bool, parent: Any
    ) -> bool:
        return isinstance(member, BaseAnalysis)

    def add_content(self, more_content: Any, no_docstring: bool = False) -> None:
        sourcename = self.get_sourcename()

        # analysis class doesn't have explicit init method.
        class_doc = self.get_doc()[0]

        # format experiment documentation into the analysis style
        class_doc_parser = AnalysisDocstring(
            target_cls=self.object,
            docstring_lines=class_doc,
            config=self.env.app.config,
            indent=self.content_indent,
        )

        # write introduction
        for i, line in enumerate(self.process_doc(class_doc_parser.generate_class_docs())):
            self.add_line(line, sourcename, i)
        self.add_line("", sourcename)

        # method and attributes
        if more_content:
            for line, src in zip(more_content.data, more_content.items):
                self.add_line(line, src[0], src[1])


def setup(app: Sphinx):
    existing_documenter = app.registry.documenters.get(AnalysisDocumenter.objtype)
    if existing_documenter is None or not issubclass(existing_documenter, AnalysisDocumenter):
        app.add_autodocumenter(AnalysisDocumenter, override=True)
