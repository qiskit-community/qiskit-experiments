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

"""Class to test composite experiments."""

from typing import Optional, List, Dict, Type, Any, Union, Tuple
import json
import copy

from test.fake_backend import FakeBackend
from test.fake_experiment import FakeExperiment

from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeMelbourne

from qiskit_experiments.framework import (
    ParallelExperiment,
    Options,
    CompositeExperimentData,
    BatchExperiment,
)
from qiskit_experiments.database_service import DatabaseServiceV1
from qiskit_experiments.database_service.device_component import DeviceComponent

# pylint: disable=missing-raises-doc


class TestComposite(QiskitTestCase):
    """
    Test composite experiment behavior.
    """

    def test_parallel_options(self):
        """
        Test parallel experiments overriding sub-experiment run and transpile options.
        """
        # These options will all be overridden
        exp0 = FakeExperiment([0])
        exp0.set_transpile_options(optimization_level=1)
        exp2 = FakeExperiment([2])
        exp2.set_experiment_options(dummyoption="test")
        exp2.set_run_options(shots=2000)
        exp2.set_transpile_options(optimization_level=1)
        exp2.set_analysis_options(dummyoption="test")

        par_exp = ParallelExperiment([exp0, exp2])

        with self.assertWarnsRegex(
            Warning,
            "Sub-experiment run and transpile options"
            " are overridden by composite experiment options.",
        ):
            self.assertEqual(par_exp.experiment_options, Options())
            self.assertEqual(par_exp.run_options, Options(meas_level=2))
            self.assertEqual(par_exp.transpile_options, Options(optimization_level=0))
            self.assertEqual(par_exp.analysis_options, Options())

            par_exp.run(FakeBackend())


class DummyService(DatabaseServiceV1):
    """
    Extremely simple database for testing
    """

    def __init__(self):
        self.database = {}

    def create_experiment(
        self,
        experiment_type: str,
        backend_name: str,
        metadata: Optional[Dict] = None,
        experiment_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        job_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        json_encoder: Type[json.JSONEncoder] = json.JSONEncoder,
        **kwargs: Any,
    ) -> str:
        """Create a new experiment in the database.

        Args:
            experiment_type: Experiment type.
            backend_name: Name of the backend the experiment ran on.
            metadata: Experiment metadata.
            experiment_id: Experiment ID. It must be in the ``uuid4`` format.
                One will be generated if not supplied.
            parent_id: The experiment ID of the parent experiment.
                The parent experiment must exist, must be on the same backend as the child,
                and an experiment cannot be its own parent.
            job_ids: IDs of experiment jobs.
            tags: Tags to be associated with the experiment.
            notes: Freeform notes about the experiment.
            json_encoder: Custom JSON encoder to use to encode the experiment.
            kwargs: Additional keywords supported by the service provider.

        Returns:
            Experiment ID.
        """

        self.database[experiment_id] = {
            "experiment_type": experiment_type,
            "parent_id": parent_id,
            "backend_name": backend_name,
            "metadata": metadata,
            "job_ids": job_ids,
            "tags": tags,
            "notes": notes,
            "share_level": kwargs.get("share_level", None),
            "figure_names": kwargs.get("figure_names", None),
        }
        return experiment_id

    def update_experiment(
        self,
        experiment_id: str,
        metadata: Optional[Dict] = None,
        job_ids: Optional[List[str]] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Update an existing experiment.

        Args:
            experiment_id: Experiment ID.
            metadata: Experiment metadata.
            job_ids: IDs of experiment jobs.
            notes: Freeform notes about the experiment.
            tags: Tags to be associated with the experiment.
            kwargs: Additional keywords supported by the service provider.
        """
        raise Exception("not implemented")

    def experiment(
        self, experiment_id: str, json_decoder: Type[json.JSONDecoder] = json.JSONDecoder
    ) -> Dict:
        """Retrieve a previously stored experiment.

        Args:
            experiment_id: Experiment ID.
            json_decoder: Custom JSON decoder to use to decode the retrieved experiment.

        Returns:
            A dictionary containing the retrieved experiment data.
        """

        db_entry = copy.deepcopy(self.database[experiment_id])
        backend_name = db_entry.pop("backend_name")
        backend = FakeMelbourne()
        if backend_name == backend.name():
            db_entry["backend"] = backend
        db_entry["experiment_id"] = experiment_id

        return db_entry

    def experiments(
        self,
        limit: Optional[int] = 10,
        json_decoder: Type[json.JSONDecoder] = json.JSONDecoder,
        device_components: Optional[Union[str, DeviceComponent]] = None,
        experiment_type: Optional[str] = None,
        backend_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
        tags_operator: Optional[str] = "OR",
        **filters: Any,
    ) -> List[Dict]:
        raise Exception("not implemented")

    def delete_experiment(self, experiment_id: str) -> None:
        raise Exception("not implemented")

    def create_analysis_result(
        self,
        experiment_id: str,
        result_data: Dict,
        result_type: str,
        device_components: Optional[Union[str, DeviceComponent]] = None,
        tags: Optional[List[str]] = None,
        quality: Optional[str] = None,
        verified: bool = False,
        result_id: Optional[str] = None,
        json_encoder: Type[json.JSONEncoder] = json.JSONEncoder,
        **kwargs: Any,
    ) -> str:
        raise Exception("not implemented")

    def update_analysis_result(
        self,
        result_id: str,
        result_data: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        quality: Optional[str] = None,
        verified: bool = None,
        **kwargs: Any,
    ) -> None:
        raise Exception("not implemented")

    def analysis_result(
        self, result_id: str, json_decoder: Type[json.JSONDecoder] = json.JSONDecoder
    ) -> Dict:
        raise Exception("not implemented")

    def analysis_results(
        self,
        limit: Optional[int] = 10,
        json_decoder: Type[json.JSONDecoder] = json.JSONDecoder,
        device_components: Optional[Union[str, DeviceComponent]] = None,
        experiment_id: Optional[str] = None,
        result_type: Optional[str] = None,
        backend_name: Optional[str] = None,
        quality: Optional[str] = None,
        verified: Optional[bool] = None,
        tags: Optional[List[str]] = None,
        tags_operator: Optional[str] = "OR",
        **filters: Any,
    ) -> List[Dict]:
        raise Exception("not implemented")

    def delete_analysis_result(self, result_id: str) -> None:
        raise Exception("not implemented")

    def create_figure(
        self, experiment_id: str, figure: Union[str, bytes], figure_name: Optional[str]
    ) -> Tuple[str, int]:
        raise Exception("not implemented")

    def update_figure(
        self, experiment_id: str, figure: Union[str, bytes], figure_name: str
    ) -> Tuple[str, int]:
        raise Exception("not implemented")

    def figure(
        self, experiment_id: str, figure_name: str, file_name: Optional[str] = None
    ) -> Union[int, bytes]:
        raise Exception("not implemented")

    def delete_figure(
        self,
        experiment_id: str,
        figure_name: str,
    ) -> None:
        raise Exception("not implemented")

    @property
    def preferences(self) -> Dict:
        raise Exception("not implemented")


class TestCompositeExperimentData(QiskitTestCase):
    """
    Test operations on objects of CompositeExperimentData
    """

    def setUp(self):
        super().setUp()

        self.backend = FakeMelbourne()
        self.share_level = "hey"

        exp1 = FakeExperiment([0, 2])
        exp2 = FakeExperiment([1, 3])
        par_exp = ParallelExperiment([exp1, exp2])
        exp3 = FakeExperiment(4)
        batch_exp = BatchExperiment([par_exp, exp3])

        self.rootdata = CompositeExperimentData(
            batch_exp, backend=self.backend
        )

        self.rootdata.share_level = self.share_level

    def check_attributes(self, expdata):
        """
        Recursively traverse the tree to verify attributes
        """
        self.assertEqual(expdata.backend, self.backend)
        self.assertEqual(expdata.share_level, self.share_level)

        if isinstance(expdata, CompositeExperimentData):
            components = expdata.component_experiment_data()
            comp_ids = expdata.metadata["component_ids"]
            comp_classes = expdata.metadata["component_classes"]
            for childdata, comp_id, comp_class in zip(components, comp_ids, comp_classes):
                self.check_attributes(childdata)
                self.assertEqual(childdata.parent_id, expdata.experiment_id)
                self.assertEqual(childdata.experiment_id, comp_id)
                self.assertEqual(childdata.__class__.__name__, comp_class)

    def check_if_equal(self, expdata1, expdata2, is_a_copy):
        """
        Recursively traverse the tree and check equality of expdata1 and expdata2
        """
        self.assertEqual(expdata1.backend.name(), expdata2.backend.name())
        self.assertEqual(expdata1.tags, expdata2.tags)
        self.assertEqual(expdata1.experiment_type, expdata2.experiment_type)
        self.assertEqual(expdata1.share_level, expdata2.share_level)

        metadata1 = copy.copy(expdata1.metadata)
        metadata2 = copy.copy(expdata2.metadata)
        if is_a_copy:
            comp_ids1 = metadata1.pop("component_ids", None)
            comp_ids2 = metadata2.pop("component_ids", None)
            if comp_ids1 is None:
                self.assertEqual(comp_ids2, None)
            else:
                self.assertNotEqual(comp_ids1, comp_ids2)
            if expdata1.parent_id is None:
                self.assertEqual(expdata2.parent_id, None)
            else:
                self.assertNotEqual(expdata1.parent_id, expdata2.parent_id)
        else:
            self.assertEqual(expdata1.parent_id, expdata2.parent_id)
        self.assertEqual(metadata1, metadata2)

        if isinstance(expdata1, CompositeExperimentData):
            for childdata1, childdata2 in zip(
                expdata1.component_experiment_data(), expdata2.component_experiment_data()
            ):
                self.check_if_equal(childdata1, childdata2, is_a_copy)

    def test_composite_experiment_data_attributes(self):
        """
        Verify correct attributes of parents and children
        """
        self.check_attributes(self.rootdata)
        self.assertEqual(self.rootdata.parent_id, None)

    def test_composite_save_load(self):
        """
        Verify that saving and loading restores the original composite experiment data object
        """

        self.rootdata.service = DummyService()
        self.rootdata.save()
        loaded_data = CompositeExperimentData.load(
            self.rootdata.experiment_id, self.rootdata.service
        )

        self.check_if_equal(loaded_data, self.rootdata, is_a_copy=False)

    def test_composite_save_metadata(self):
        """
        Verify that saving metadata and loading restores the original composite experiment data object
        """

        self.rootdata.service = DummyService()
        self.rootdata.save_metadata()
        loaded_data = CompositeExperimentData.load(
            self.rootdata.experiment_id, self.rootdata.service
        )

        self.check_if_equal(loaded_data, self.rootdata, is_a_copy=False)

    def test_composite_copy_metadata(self):
        """
        Test CompositeExperimentData._copy_metadata
        """
        new_instance = self.rootdata._copy_metadata()
        self.check_if_equal(new_instance, self.rootdata, is_a_copy=True)
        self.check_attributes(new_instance)
