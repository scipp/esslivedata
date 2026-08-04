# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest

from ess.livedata import StreamId, StreamKind
from ess.livedata.config.instrument import Instrument
from ess.livedata.config.stream import Device, F144Stream
from ess.livedata.core.preprocessor import Accumulator
from ess.livedata.preprocessors.accumulators import LogData
from ess.livedata.preprocessors.timeseries import (
    BACKFILL_MAX_SIZE,
    LogdataPreprocessorFactory,
)
from ess.livedata.workflows.wavelength_lut_workflow_specs import CHOPPER_CASCADE_SOURCE


class TestBackfillBudget:
    """Retention in this service is the budget for a job's activation backfill.

    Every log stream holds a buffer whether or not anyone plots it, and starting
    one job per stream publishes all of them at once, so the bound applies to
    every path that creates a preprocessor here.
    """

    @pytest.fixture
    def factory(self) -> LogdataPreprocessorFactory:
        streams = {
            'temp_sensor': F144Stream(source='temp_sensor', topic='topic', units='K'),
            'motion': Device(value='motion_rbv', target='motion_val', units='mm'),
            'motion_rbv': F144Stream(source='motion_rbv', topic='topic', units='mm'),
            'motion_val': F144Stream(source='motion_val', topic='topic', units='mm'),
        }
        return LogdataPreprocessorFactory(
            instrument=Instrument(name='test_instrument', streams=streams)
        )

    def fill(self, accumulator: Accumulator, n: int) -> None:
        for i in range(n):
            accumulator.add(
                0, LogData(time=i * 1_000_000_000, value=float(i), target=0.0)
            )

    @pytest.mark.parametrize(
        'stream_id',
        [
            StreamId(kind=StreamKind.LOG, name='temp_sensor'),
            StreamId(kind=StreamKind.DEVICE, name='motion'),
            StreamId(kind=StreamKind.LOG, name=CHOPPER_CASCADE_SOURCE),
        ],
        ids=['f144', 'device', 'synthetic'],
    )
    def test_retention_is_bounded_by_backfill_budget(self, factory, stream_id):
        accumulator = factory.make_preprocessor(stream_id)
        self.fill(accumulator, BACKFILL_MAX_SIZE + 100)
        assert accumulator.get().sizes['time'] == BACKFILL_MAX_SIZE
