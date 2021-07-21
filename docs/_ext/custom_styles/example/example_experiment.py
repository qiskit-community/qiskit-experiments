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
Class documentation examples.

.. warning::

    This module is just an example for documentation. Do not import.

"""

from qiskit.providers import Options

from qiskit_experiments.curve_analysis.curve_analysis import CurveAnalysis
from qiskit_experiments.framework.base_experiment import BaseExperiment


class DocumentedCurveAnalysis(CurveAnalysis):
    r"""One line summary of this class. This is shown in the top level contains list.

    # section: overview
        Overview of this analysis. It is recommended to write this section.
        Here you can explain technical aspect of fit algorithm or fit model.
        Standard reStructuredText directives can be used.

        You can use following sections

        - ``warning``
        - ``note``
        - ``example``
        - ``reference``
        - ``tutorial``

        See :class:`DocumentedExperiment` for description of these sections.
        In addition to above sections, analysis template provides following extra sections.

    # section: fit_model
        Here you can describe your fitting model.
        Standard reStructuredText directives can be used. For example:

        .. math::

            F(x) = a \exp(-(x-f)^2/(2\sigma^2)) + b

        enables you to use the Latex syntax to write your equation.

    # section: fit_parameters
        Here you can explain fit parameter details.
        This section provides a special syntax to describe details of each parameter.
        Documentation except for this syntax will be just ignored.

        defpar a:
            desc: Description of parameter :math:`a`.
            init_guess: Here you can describe how this analysis estimate initial guess of
                parameter :math:`a`.
            bounds: Here you can describe how this analysis bounds parameter :math:`a` value
                during the fit.

        defpar b:
            desc: Description of parameter :math:`b`.
            init_guess: Here you can describe how this analysis estimate initial guess of
                parameter :math:`b`.
            bounds: Here you can describe how this analysis bounds parameter :math:`b` value
                during the fit.

        Note that you cannot write text block (i.e. bullet lines, math mode, parsed literal, ...)
        in the ``defpar`` syntax items. These are a single line description of parameters.
        You can write multiple ``defpar`` block for each fitting parameter.

        It would be nice if parameter names conform to the parameter key values appearing in the
        analysis result. For example, if fit model defines the parameter :math:`\sigma` and
        this appears as ``eta`` in the result, user cannot find correspondence of these parameters.

    """

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options.

        .. note::

            This method documentation should conforms to the below documentation syntax.
            Namely, the title should be "Analysis Options" followed by a single colon
            and description should be written in the Google docstring style.
            Numpy style is not accepted.

            Google style docstring guideline:
            https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

            Documentation except for the analysis options will be just ignored, e.g. this note.
            If analysis options contains some values from the parent class,
            the custom Sphinx parser searches for the parent class method documentation
            and automatically generate documentation for all available options.
            If there is any missing documentation the Sphinx build will fail.

        Analysis Options:
            opt1 (int): Description for the option1.
            opt2 (bool): Description for the option2.
            opt3 (str): Description for the option3.

        """
        opts = super()._default_options()
        opts.opt1 = 1.0
        opts.opt2 = True
        opts.opt3 = "opt3"

        return opts


class DocumentedExperiment(BaseExperiment):
    """One line summary of this class. This is shown in the top level contains list.

    # section: overview
        Overview of this experiment. It is recommended to write this section.
        Here you can explain technical aspect of experiment, protocol, etc...
        Standard reStructuredText directives can be used.

    # section: warning
        Warning about this experiment if exist.
        Some functionality is not available or under development,
        you should write these details here.

    # section: note
        Notification about this experiment if exist.

    # section: example
        Example code of this experiment.
        If this experiment requires user to manage complicated options,
        it might be convenient for users to have some code example here.

        You can write code example, for example, as follows

        .. code-block:: python

            import qiskit_experiments
            my_experiment = qiskit_experiments.MyExperiment(**options)

    # section: reference
        Currently this supports article reference in arXiv database.
        You can use following helper directive.

        .. ref_arxiv:: Auth2020a 21xx.01xxx

        This directive takes two arguments separated by a whitespace.
        The first argument is arbitrary label for this article, which may be used to
        refer to this paper from other sections.
        Second argument is the arXiv ID of the paper referring to.
        Once this directive is inserted, Sphinx searches the arXiv database and
        automatically generates a formatted bibliography with the hyperlink to the online PDF.

    # section: tutorial
        You can refer to the arbitrary web page here.
        Following helper directive can be used.

        .. ref_website:: Qiskit Experiment Github, https://github.com/Qiskit/qiskit-experiments

        This directive takes two arguments separated by a comma.
        The first argument is arbitrary label shown before the link. Whitespace can be included.
        The second argument is the URL of the website to hyperlink.

    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        .. note::

            This method documentation should conforms to the below documentation syntax.
            Namely, the title should be "Experiment Options" followed by a single colon
            and description should be written in the Google docstring style.
            Numpy style is not accepted.

            Google style docstring guideline:
            https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

            Documentation except for the experiment options will be just ignored, e.g. this note.
            If experiment options contains some values from the parent class,
            the custom Sphinx parser searches for the parent class method documentation
            and automatically generate documentation for all available options.
            If there is any missing documentation the Sphinx build will fail.

        Experiment Options:
            opt1 (int): Description for the option1.
            opt2 (bool): Description for the option2.
            opt3 (str): Description for the option3.

        """
        opts = super()._default_experiment_options()
        opts.opt1 = 1.0
        opts.opt2 = True
        opts.opt3 = "opt3"

        return opts

    def __init__(self, qubit: int):
        """Create new experiment.

        .. note::

            This documentation is shown as-is.

        Args:
            qubit: The qubit to run experiment.
        """
        super().__init__(qubits=[qubit])

    def circuits(self, backend=None):
        pass
