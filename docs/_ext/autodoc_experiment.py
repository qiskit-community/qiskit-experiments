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
Documentation extension for experiment class.
"""

from typing import Any

from docs._ext.custom_styles.styles import ExperimentDocstring
from qiskit.exceptions import QiskitError
from qiskit_experiments.framework.base_experiment import BaseExperiment
from sphinx.application import Sphinx
from sphinx.ext.autodoc import ClassDocumenter


class ExperimentDocumenter(ClassDocumenter):
    """Sphinx extension for the custom documentation of the standard experiment class."""

    objtype = "experiment"
    directivetype = 'class'
    priority = 10 + ClassDocumenter.priority
    option_spec = dict(ClassDocumenter.option_spec)

    @classmethod
    def can_document_member(
        cls, member: Any, membername: str, isattr: bool, parent: Any
    ) -> bool:
        return isinstance(member, BaseExperiment)

    def add_content(self, more_content: Any, no_docstring: bool = False) -> None:
        sourcename = self.get_sourcename()

        try:
            class_doc, init_doc = self.get_doc()
        except ValueError:
            raise QiskitError(
                f"Documentation of {self.name} doesn't match with the expected format."
                "Please run sphinx build without using the experiment template."
            )

        # format experiment documentation into the experiment style
        class_doc_parser = ExperimentDocstring(
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


def setup(app: Sphinx):
    existing_documenter = app.registry.documenters.get(ExperimentDocumenter.objtype)
    if existing_documenter is None or not issubclass(existing_documenter, ExperimentDocumenter):
        app.add_autodocumenter(ExperimentDocumenter, override=True)
