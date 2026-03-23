# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

import scipp as sc

from ess.livedata.config.workflow_spec import JobId
from ess.livedata.core.message import StreamKind
from ess.livedata.dashboard.range_publisher import FakeRangePublisher, RangePublisher
from ess.livedata.fakes import FakeMessageSink


def test_range_publisher_publishes_range():
    sink = FakeMessageSink()
    publisher = RangePublisher(sink=sink)
    job_id = JobId(source_name='detector1', job_number=uuid.uuid4())

    publisher.publish(job_id, low=1000.0, high=5000.0, unit='ns')

    assert len(sink.messages) == 1
    msg = sink.messages[0]
    assert msg.stream.kind == StreamKind.LIVEDATA_ROI
    assert msg.stream.name == f"detector1/{job_id.job_number}/histogram_slice"
    assert isinstance(msg.value, sc.DataArray)
    assert msg.value.sizes == {'bound': 2}
    assert sc.identical(msg.value['bound', 0].data, sc.scalar(1000.0, unit='ns'))
    assert sc.identical(msg.value['bound', 1].data, sc.scalar(5000.0, unit='ns'))


def test_range_publisher_publishes_without_unit():
    sink = FakeMessageSink()
    publisher = RangePublisher(sink=sink)
    job_id = JobId(source_name='detector1', job_number=uuid.uuid4())

    publisher.publish(job_id, low=10.0, high=50.0, unit=None)

    msg = sink.messages[0]
    assert msg.value.sizes == {'bound': 2}
    # sc.scalar with unit=None produces dimensionless
    assert msg.value['bound', 0].value == 10.0
    assert msg.value['bound', 1].value == 50.0


def test_range_publisher_clear():
    sink = FakeMessageSink()
    publisher = RangePublisher(sink=sink)
    job_id = JobId(source_name='detector1', job_number=uuid.uuid4())

    publisher.clear(job_id)

    assert len(sink.messages) == 1
    msg = sink.messages[0]
    assert msg.stream.kind == StreamKind.LIVEDATA_ROI
    assert msg.stream.name == f"detector1/{job_id.job_number}/histogram_slice"
    assert msg.value.sizes == {'bound': 0}


def test_fake_range_publisher_records_publishes():
    publisher = FakeRangePublisher()
    job_id = JobId(source_name='detector1', job_number=uuid.uuid4())

    publisher.publish(job_id, low=1.0, high=5.0, unit='ns')

    assert len(publisher.published) == 1
    assert publisher.published[0] == (job_id, 1.0, 5.0, 'ns')


def test_fake_range_publisher_records_clear():
    publisher = FakeRangePublisher()
    job_id = JobId(source_name='detector1', job_number=uuid.uuid4())

    publisher.clear(job_id)

    assert len(publisher.published) == 1
    assert publisher.published[0] == (job_id, None, None, None)


def test_fake_range_publisher_reset():
    publisher = FakeRangePublisher()
    job_id = JobId(source_name='detector1', job_number=uuid.uuid4())

    publisher.publish(job_id, low=1.0, high=5.0, unit='ns')
    publisher.reset()

    assert len(publisher.published) == 0
