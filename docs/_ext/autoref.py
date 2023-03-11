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
Helper directive to generate reference in convenient form.
"""
import arxiv

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx


class WebSite(Directive):
    """A custom helper directive for showing website link.

    This can be used, for example,

    .. code-block::

        .. ref_website:: qiskit-experiments, https://github.com/Qiskit/qiskit-experiments

    """
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True

    def run(self):
        try:
            name, url = self.arguments[0].split(",")
        except ValueError:
            raise ValueError(
                f"{self.arguments[0]} is invalid website directive format. "
                "Name and URL should be separated by a single comma."
            )

        link_name = nodes.paragraph(text=f"{name} ")
        link_name += nodes.reference(text="(open)", refuri=url)

        return [link_name]


class Arxiv(Directive):
    """A custom helper directive for generating journal information from arXiv id.

    This directive takes two arguments

    - Arbitrary reference name (no white space should be included)

    - arXiv ID

    This can be used, for example,

    .. code-block::

        .. ref_arxiv:: qasm3-paper 2104.14722

    If an article is not found, no journal information will be shown.

    """
    required_arguments = 2
    optional_arguments = 0
    final_argument_whitespace = False

    def run(self):

        # search arXiv database
        try:
            search = arxiv.Search(id_list=[self.arguments[1]])
            paper = next(search.results())
        except Exception:
            return []

        # generate journal link nodes
        ret_node = nodes.paragraph()

        journal = ""
        if paper.journal_ref:
            journal += f", {paper.journal_ref}, "
        if paper.doi:
            journal += f"doi: {paper.doi}"

        ret_node += nodes.Text(f"[{self.arguments[0]}] ")
        ret_node += nodes.Text(", ".join([author.name for author in paper.authors]) + ", ")
        ret_node += nodes.emphasis(text=f"{paper.title}")
        if journal:
            ret_node += nodes.Text(journal)
        ret_node += nodes.Text(" ")
        ret_node += nodes.reference(text="(open)", refuri=paper.pdf_url)

        return [ret_node]


def setup(app: Sphinx):
    app.add_directive("ref_arxiv", Arxiv)
    app.add_directive("ref_website", WebSite)
