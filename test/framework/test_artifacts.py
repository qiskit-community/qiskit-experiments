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

"""Test artifacts."""

from test.base import QiskitExperimentsTestCase
from datetime import datetime

from qiskit_experiments.framework import ArtifactData


class TestArtifacts(QiskitExperimentsTestCase):
    """Test cases for the ArtifactData class."""

    def test_basic_artifact(self):
        """Test artifact properties."""
        timestamp = datetime.now()
        artifact = ArtifactData(artifact_id=0, name="test", data="foo", created_time=timestamp)
        self.assertEqual(artifact.artifact_id, 0)
        self.assertEqual(artifact.name, "test")
        self.assertEqual(artifact.experiment, None)
        self.assertEqual(artifact.device_components, [])
        self.assertEqual(artifact.dtype, "str")
        self.assertEqual(artifact.created_time, timestamp)
        self.assertEqual(
            str(artifact),
            "ArtifactData(name=test, dtype=str, uid=0, experiment=None, device_components=[])",
        )

    def test_artifact_equality(self):
        """Test artifact equality."""
        timestamp = datetime.now()
        artifact1 = ArtifactData(name="test", data="foo")
        artifact2 = ArtifactData(name="test", data="foo")
        self.assertNotEqual(artifact1, artifact2)
        artifact1 = ArtifactData(artifact_id=0, name="test", data="foo", created_time=timestamp)
        artifact2 = ArtifactData(artifact_id=0, name="test", data="foo", created_time=timestamp)
        self.assertEqual(artifact1, artifact2)

    def test_serialize_artifact(self):
        """Test serializing the artifact."""
        obj = ArtifactData(name="test", data="foo")
        self.assertRoundTripSerializable(obj)
        obj2 = ArtifactData(name="test", data={"foo": 123, "blah": obj})
        self.assertRoundTripSerializable(obj2)
