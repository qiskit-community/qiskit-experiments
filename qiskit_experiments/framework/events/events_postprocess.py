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
Post processing events.
"""

import copy


def set_analysis(experiment, analysis, job, experiment_data, **kwargs):

    if analysis and experiment.__analysis_class__ is not None:
        run_analysis_callback = experiment.run_analysis
    else:
        run_analysis_callback = None

    experiment_data.add_data(job, post_processing_callback=run_analysis_callback)


def add_job_metadata(experiment, experiment_data, job, run_options, **kwargs):
    """Add runtime job metadata to ExperimentData.

    Args:
        experiment: Experiment object.
        experiment_data: The experiment data container.
        job: The job object.
        run_options: Backend run options for the job.
    """
    metadata = {
        "job_id": job.job_id(),
        "experiment_options": copy.copy(experiment.experiment_options.__dict__),
        "transpile_options": copy.copy(experiment.transpile_options.__dict__),
        "analysis_options": copy.copy(experiment.analysis_options.__dict__),
        "run_options": copy.copy(run_options)
    }

    experiment_data._metadata["job_metadata"].append(metadata)
