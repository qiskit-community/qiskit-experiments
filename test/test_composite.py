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

from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeMelbourne
from qiskit.providers.backend import Backend

from qiskit_experiments.framework import (
    ParallelExperiment,
    Options,
    CompositeExperimentData,
    BaseExperiment,
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
        exp0 = FakeExperiment(0)
        exp0.set_transpile_options(optimization_level=1)
        exp2 = FakeExperiment(2)
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


class DummyExperiment(BaseExperiment):
    """
    An experiment that does nothing, to fill in the experiment tree
    """

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        return []


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
        self.job_ids = [1, 2, 3, 4, 5]
        self.share_level = "hey"

        exp1 = DummyExperiment([0, 2])
        exp2 = DummyExperiment([1, 3])
        par_exp = ParallelExperiment([exp1, exp2])
        exp3 = DummyExperiment(4)
        batch_exp = BatchExperiment([par_exp, exp3])

        self.rootdata = CompositeExperimentData(
            batch_exp, backend=self.backend, job_ids=self.job_ids
        )
        self.rootdata.share_level = self.share_level

    def check_attributes(self, expdata):
        """
        Recursively traverse the tree to verify attributes
        """
        self.assertEqual(expdata.backend, self.backend)
        self.assertEqual(expdata.job_ids, self.job_ids)
        self.assertEqual(expdata.share_level, self.share_level)

        # Experiments have to be tagged with their direct parents and the root
        self.assertTrue(len(expdata.tags) == 1 or len(expdata.tags) == 2)
        if len(expdata.tags) == 2:
            self.assertNotEqual(expdata.tags[0], expdata.tags[1])
        self.assertTrue("root exp id: " + self.rootdata.experiment_id in expdata.tags)

        if isinstance(expdata, CompositeExperimentData):
            for childdata in expdata.component_experiment_data():
                self.check_attributes(childdata)
                self.assertTrue("parent exp id: " + expdata.experiment_id in childdata.tags)

    def check_if_equal(self, expdata1, expdata2):
        """
        Recursively traverse the tree and checkequality of expdata1 and expdata2
        """
        self.assertEqual(expdata1.backend.name(), expdata2.backend.name())
        self.assertEqual(expdata1.job_ids, expdata2.job_ids)
        self.assertEqual(expdata1.tags, expdata2.tags)
        self.assertEqual(expdata1.experiment_type, expdata2.experiment_type)
        self.assertEqual(expdata1.metadata, expdata2.metadata)
        self.assertEqual(expdata1.share_level, expdata2.share_level)

        if isinstance(expdata1, CompositeExperimentData):
            for childdata1, childdata2 in zip(
                expdata1.component_experiment_data(), expdata2.component_experiment_data()
            ):
                self.check_if_equal(childdata1, childdata2)

    def test_composite_experiment_data_attributes(self):
        """
        Verify correct attributes of parents and children
        """
        self.check_attributes(self.rootdata)

    def test_composite_save_load(self):
        """
        Verify that saving and loading restores the original composite experiment data object
        """

        self.rootdata.service = DummyService()
        self.rootdata.save()
        loaded_data = CompositeExperimentData.load(
            self.rootdata.experiment_id, self.rootdata.service
        )

        self.check_if_equal(loaded_data, self.rootdata)

    def test_composite_save_metadata(self):
        """
        Verify that saving metadata and loading restores the original composite experiment data object
        """

        self.rootdata.service = DummyService()
        self.rootdata.save_metadata()
        loaded_data = CompositeExperimentData.load(
            self.rootdata.experiment_id, self.rootdata.service
        )

        self.check_if_equal(loaded_data, self.rootdata)
