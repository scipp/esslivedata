# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for range request plotter edit handler logic."""

import uuid

import scipp as sc

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard.range_publisher import FakeRangePublisher
from ess.livedata.dashboard.range_request_plots import (
    RangeRequestParams,
    RangeRequestPlotter,
)
from ess.livedata.handlers.detector_view.types import HistogramSliceReadback


def _make_data_key() -> ResultKey:
    return ResultKey(
        workflow_id=WorkflowId(
            instrument='dummy', namespace='ns', name='test', version=1
        ),
        job_id=JobId(source_name='detector1', job_number=uuid.uuid4()),
        output_name='histogram_slice',
    )


def _make_readback_data(
    low: float = 1000.0, high: float = 5000.0, unit: str = 'ns'
) -> sc.DataArray:
    """Create readback DataArray in the format the backend produces."""
    return HistogramSliceReadback(
        sc.DataArray(
            data=sc.concat(
                [sc.scalar(low, unit=unit), sc.scalar(high, unit=unit)], dim='bound'
            )
        )
    )


def _make_empty_readback(unit: str = 'ns') -> sc.DataArray:
    return HistogramSliceReadback(
        sc.DataArray(data=sc.zeros(sizes={'bound': 0}, unit=unit))
    )


class TestRangeRequestPlotter:
    """Tests for RangeRequestPlotter domain logic."""

    def test_compute_extracts_spectral_unit(self):
        plotter = RangeRequestPlotter(RangeRequestParams())
        data_key = _make_data_key()
        data = _make_readback_data(unit='us')

        plotter.compute({data_key: data})

        assert plotter._spectral_unit == str(sc.Unit('us'))
        assert plotter._data_key == data_key

    def test_compute_extracts_unit_from_empty_readback(self):
        plotter = RangeRequestPlotter(RangeRequestParams())
        data_key = _make_data_key()
        data = _make_empty_readback(unit='ms')

        plotter.compute({data_key: data})

        assert plotter._data_key == data_key
        assert plotter._spectral_unit == 'ms'

    def test_edit_handler_publishes_range(self):
        publisher = FakeRangePublisher()
        plotter = RangeRequestPlotter(RangeRequestParams(), range_publisher=publisher)
        data_key = _make_data_key()
        plotter.compute({data_key: _make_readback_data(unit='ns')})

        handler = plotter._create_edit_handler()
        handler((1000.0, 5000.0))

        assert len(publisher.published) == 1
        job_id, low, high, unit = publisher.published[0]
        assert job_id == data_key.job_id
        assert low == 1000.0
        assert high == 5000.0
        assert unit == 'ns'

    def test_edit_handler_normalizes_min_max(self):
        publisher = FakeRangePublisher()
        plotter = RangeRequestPlotter(RangeRequestParams(), range_publisher=publisher)
        data_key = _make_data_key()
        plotter.compute({data_key: _make_readback_data()})

        handler = plotter._create_edit_handler()
        # high before low (user dragged right-to-left)
        handler((5000.0, 1000.0))

        _, low, high, _ = publisher.published[0]
        assert low == 1000.0
        assert high == 5000.0

    def test_edit_handler_skips_unchanged(self):
        publisher = FakeRangePublisher()
        plotter = RangeRequestPlotter(RangeRequestParams(), range_publisher=publisher)
        data_key = _make_data_key()
        plotter.compute({data_key: _make_readback_data()})

        handler = plotter._create_edit_handler()
        handler((100.0, 200.0))
        handler((100.0, 200.0))

        assert len(publisher.published) == 1

    def test_edit_handler_publishes_clear_on_none(self):
        publisher = FakeRangePublisher()
        plotter = RangeRequestPlotter(RangeRequestParams(), range_publisher=publisher)
        data_key = _make_data_key()
        plotter.compute({data_key: _make_readback_data()})

        handler = plotter._create_edit_handler()
        # Set a range first
        handler((100.0, 200.0))
        # Then clear it
        handler(None)

        assert len(publisher.published) == 2
        _, low, _, _ = publisher.published[1]
        assert low is None  # FakeRangePublisher records None for clear

    def test_edit_handler_ignores_none_when_already_empty(self):
        publisher = FakeRangePublisher()
        plotter = RangeRequestPlotter(RangeRequestParams(), range_publisher=publisher)
        data_key = _make_data_key()
        plotter.compute({data_key: _make_readback_data()})

        handler = plotter._create_edit_handler()
        # None when no range was ever set — should not publish
        handler(None)

        assert len(publisher.published) == 0

    def test_edit_handler_does_not_publish_without_publisher(self):
        plotter = RangeRequestPlotter(RangeRequestParams())
        data_key = _make_data_key()
        plotter.compute({data_key: _make_readback_data()})

        handler = plotter._create_edit_handler()
        # Should not raise even without publisher
        handler((100.0, 200.0))

    def test_set_range_publisher(self):
        plotter = RangeRequestPlotter(RangeRequestParams())
        publisher = FakeRangePublisher()
        plotter.set_range_publisher(publisher)
        assert plotter._range_publisher is publisher

    def test_from_params(self):
        params = RangeRequestParams()
        plotter = RangeRequestPlotter.from_params(params)
        assert isinstance(plotter, RangeRequestPlotter)
