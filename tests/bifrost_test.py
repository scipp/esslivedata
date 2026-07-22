# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc
from ess.reduce.nexus.types import (
    EmptyDetector,
    NeXusData,
    SampleRun,
)
from scipp.testing import assert_identical
from scippnexus import NXdetector

from ess.livedata.config.instruments.bifrost import specs
from ess.livedata.config.instruments.bifrost.factories import (
    _create_base_reduction_workflow,
)
from ess.livedata.core.timestamp import Timestamp


@pytest.fixture
def bifrost_workflow():
    """
    Create and configure the base reduction workflow for tests.

    Returns ``(workflow, DetectorRegionCountsRaw)``. The workflow has the
    ratemeter providers inserted; the raw per-update counts are what the sciline
    graph computes directly (accumulation into current/cumulative views happens
    in ``StreamProcessorWorkflow``, not in the bare graph).
    """
    workflow, _DetectorRegionCounts, DetectorRegionCountsRaw = (
        _create_base_reduction_workflow()
    )
    return workflow, DetectorRegionCountsRaw


@pytest.mark.slow
def test_workflow_produces_detector_with_consecutive_detector_number(bifrost_workflow):
    wf, _DetectorRegionCounts = bifrost_workflow
    da = wf.compute(EmptyDetector[SampleRun])
    assert_identical(
        da.coords['detector_number'].transpose(da.dims),
        sc.arange('', 1, 13500 + 1, unit=None, dtype='int32').fold(
            dim='', sizes=da.sizes
        ),
    )


def _make_test_event_data(
    *,
    arc_events: dict[int, int] | None = None,
    pulse_times_ns: list[int] | None = None,
) -> sc.DataArray:
    """
    Create test event data for Bifrost detector.

    Parameters
    ----------
    arc_events:
        Dictionary mapping arc index (0-4) to number of events per arc.
        Events are distributed evenly across the arc's 900 positions.
    pulse_times_ns:
        List of pulse times in nanoseconds. Defaults to [1_000_000, 2_000_000].

    Returns
    -------
    :
        Event data in NXevent_data format with dims (event_time_zero,)
        and binned event dimension containing event_id and event_time_offset.
    """
    if arc_events is None:
        arc_events = {0: 10}  # 10 events in arc 0 by default
    if pulse_times_ns is None:
        pulse_times_ns = [1_000_000, 2_000_000]

    # Create events for each arc
    all_event_ids = []
    all_event_time_offsets = []

    n_pulses = len(pulse_times_ns)
    total_events = 0
    for arc_idx, n_events in arc_events.items():
        # Distribute events across 900 positions in this arc
        # (3 tubes x 9 channels x 100 pixels)
        # detector_number ranges: arc_idx * 2700 + 1 to arc_idx * 2700 + 2700
        positions = np.linspace(0, 899, n_events, dtype=int)
        # Map position to detector_number: tube=0, channel varies, pixel varies
        detector_numbers = arc_idx * 2700 + 1 + positions * 3  # *3 for tube spacing

        all_event_ids.extend(detector_numbers)
        # Distribute events across pulses
        all_event_time_offsets.extend([100 + i * 10 for i in range(n_events)])
        total_events += n_events

    # Distribute all events evenly across pulses
    events_per_pulse = [total_events // n_pulses] * n_pulses
    # Handle remainder
    remainder = total_events - sum(events_per_pulse)
    if remainder > 0:
        events_per_pulse[-1] += remainder

    # Build event data structure
    epoch = sc.epoch(unit='ns')
    event_time_zero = epoch + sc.array(
        dims=['event_time_zero'], values=pulse_times_ns, unit='ns', dtype='int64'
    )

    event_id = sc.array(
        dims=['event'],
        values=all_event_ids[: sum(events_per_pulse)],
        unit=None,
        dtype='int32',
    )
    event_time_offset = sc.array(
        dims=['event'],
        values=all_event_time_offsets[: sum(events_per_pulse)],
        unit='ns',
        dtype='int64',
    )
    weights = sc.ones(
        sizes={'event': sum(events_per_pulse)}, dtype='float64', unit='counts'
    )

    events = sc.DataArray(data=weights, coords={'event_time_offset': event_time_offset})
    events.coords['event_id'] = event_id

    sizes = sc.array(
        dims=['event_time_zero'], values=events_per_pulse, unit=None, dtype='int64'
    )
    begin = sc.cumsum(sizes, mode='exclusive')
    nexus_data = sc.DataArray(sc.bins(begin=begin, dim='event', data=events))
    nexus_data.coords['event_time_zero'] = event_time_zero

    return nexus_data


@pytest.mark.slow
class TestDetectorRatemeter:
    """Tests for the detector ratemeter workflow."""

    def test_ratemeter_sums_events_in_selected_arc(self, bifrost_workflow):
        """Test that ratemeter correctly sums events in the selected arc."""
        reduction_workflow, DetectorRegionCountsRaw = bifrost_workflow
        # Create test data with 100 events in arc 2
        nexus_data = _make_test_event_data(arc_events={2: 100})

        # Set up workflow with arc 2 selected
        wf = reduction_workflow.copy()
        region_params = specs.DetectorRatemeterRegionParams(
            arc=specs.ArcEnergy.ARC_3_8,  # Arc index 2
            pixel_start=0,
            pixel_stop=900,
        )
        wf[NeXusData[NXdetector, SampleRun]] = nexus_data
        wf[specs.DetectorRatemeterRegionParams] = region_params

        # Compute result
        result = wf.compute(DetectorRegionCountsRaw)

        # Check that we get the expected count
        assert result.value == 100
        assert result.variance == 100  # Poisson statistics

    def test_ratemeter_with_pixel_range_selection(self, bifrost_workflow):
        """Test that pixel range selection works correctly."""
        reduction_workflow, DetectorRegionCountsRaw = bifrost_workflow
        # Create test data with events evenly distributed across arc 0
        nexus_data = _make_test_event_data(arc_events={0: 900})

        # Select first half of pixels (0-450)
        wf = reduction_workflow.copy()
        region_params = specs.DetectorRatemeterRegionParams(
            arc=specs.ArcEnergy.ARC_2_7,  # Arc index 0
            pixel_start=0,
            pixel_stop=450,
        )
        wf[NeXusData[NXdetector, SampleRun]] = nexus_data
        wf[specs.DetectorRatemeterRegionParams] = region_params

        result = wf.compute(DetectorRegionCountsRaw)

        # Should get approximately half the events (allowing some tolerance)
        assert 400 <= result.value <= 500

    def test_ratemeter_with_different_arcs(self, bifrost_workflow):
        """Test that arc selection correctly isolates events."""
        reduction_workflow, DetectorRegionCountsRaw = bifrost_workflow
        # Create test data with different event counts per arc
        nexus_data = _make_test_event_data(arc_events={0: 50, 2: 150, 4: 200})

        # Test arc 4 (5.0 meV)
        wf = reduction_workflow.copy()
        region_params = specs.DetectorRatemeterRegionParams(
            arc=specs.ArcEnergy.ARC_5_0,  # Arc index 4
            pixel_start=0,
            pixel_stop=900,
        )
        wf[NeXusData[NXdetector, SampleRun]] = nexus_data
        wf[specs.DetectorRatemeterRegionParams] = region_params

        result = wf.compute(DetectorRegionCountsRaw)
        assert result.value == 200

    def test_ratemeter_raw_counts_carry_no_time_coords(self, bifrost_workflow):
        """Per-update counts are bare: time coords are stamped on accumulation.

        Stamping a ``time`` coord here would suppress the ``start_time``/
        ``end_time`` the dashboard's rate normalization needs (see
        ``_add_time_coords``) and break coord matching across the accumulator sum.
        """
        reduction_workflow, DetectorRegionCountsRaw = bifrost_workflow
        nexus_data = _make_test_event_data(arc_events={1: 50})

        wf = reduction_workflow.copy()
        wf[NeXusData[NXdetector, SampleRun]] = nexus_data
        wf[specs.DetectorRatemeterRegionParams] = specs.DetectorRatemeterRegionParams(
            arc=specs.ArcEnergy.ARC_3_2,
            pixel_start=0,
            pixel_stop=900,
        )

        result = wf.compute(DetectorRegionCountsRaw)

        assert result.unit == 'counts'
        assert 'time' not in result.coords
        assert 'start_time' not in result.coords
        assert 'end_time' not in result.coords

    @pytest.mark.slow
    def test_ratemeter_current_output_normalizes_to_rate(self):
        """The current output carries the time bounds needed for counts/s.

        Drives the real ``StreamProcessorWorkflow`` (as the production factory
        builds it) and checks the current output can be normalized to a rate,
        while the cumulative output defers its time bounds to the job layer.
        """
        from ess.livedata.dashboard.plots import _normalize_to_rate
        from ess.livedata.preprocessors.accumulation_mode import Cumulative, Current
        from ess.livedata.preprocessors.accumulators import (
            make_no_copy_accumulator_pair,
        )
        from ess.livedata.preprocessors.stream_processor_workflow import (
            StreamProcessorWorkflow,
        )

        reduction_workflow, DetectorRegionCounts, _ = _create_base_reduction_workflow()
        wf = reduction_workflow.copy()
        wf[specs.DetectorRatemeterRegionParams] = specs.DetectorRatemeterRegionParams(
            arc=specs.ArcEnergy.ARC_5_0, pixel_start=0, pixel_stop=900
        )
        cumulative, window = make_no_copy_accumulator_pair()
        workflow = StreamProcessorWorkflow(
            wf,
            dynamic_keys={'unified_detector': NeXusData[NXdetector, SampleRun]},
            target_keys={
                'detector_region_counts': DetectorRegionCounts[Current],
                'detector_region_counts_cumulative': DetectorRegionCounts[Cumulative],
            },
            accumulators={
                DetectorRegionCounts[Cumulative]: cumulative,
                DetectorRegionCounts[Current]: window,
            },
            window_outputs=['detector_region_counts'],
        )
        workflow.build()
        workflow.accumulate(
            {'unified_detector': _make_test_event_data(arc_events={4: 100})},
            start_time=Timestamp.from_ns(0),
            end_time=Timestamp.from_ns(2_000_000_000),  # 2 s window
        )
        results = workflow.finalize()

        current = results['detector_region_counts']
        assert current.value == 100
        assert {'time', 'start_time', 'end_time'} <= set(current.coords)
        rate = _normalize_to_rate(current)
        assert rate.unit == 'counts/s'
        assert rate.value == 50  # 100 counts / 2 s

        # Cumulative defers time bounds to the job layer (_add_time_coords).
        cumulative_out = results['detector_region_counts_cumulative']
        assert cumulative_out.value == 100
        assert 'start_time' not in cumulative_out.coords


class TestUnifiedDetectorViewLayout:
    """Image and spectrum layouts the static triplet-bounds grid depends on.

    The grid overlay (a second plot layer) is drawn in fixed coordinates for a
    15-row by 900-column image, so the image must stack (arc, tube) as rows and
    (channel, pixel) as columns. These pin that contract.
    """

    @pytest.fixture
    def raw_detector(self) -> sc.DataArray:
        """Raw combined detector data with dims (arc, tube, channel, pixel).

        Values are bounded so float32 count sums stay exact for conservation.
        """
        n = 5 * 3 * 9 * 100
        values = sc.array(
            dims=['flat'],
            values=[i % 7 for i in range(n)],
            unit='counts',
            dtype='float64',
        )
        return sc.DataArray(
            values.fold('flat', sizes={'arc': 5, 'tube': 3, 'channel': 9, 'pixel': 100})
        )

    @pytest.fixture
    def cumulative_histogram(self, raw_detector: sc.DataArray) -> sc.DataArray:
        """Cumulative histogram: the 2D image with an appended spectral axis."""
        image = specs._logical_view(raw_detector, 'unified_detector')
        n_toa = 7
        return sc.DataArray(
            sc.broadcast(
                image.data, dims=[*image.dims, 'toa'], shape=[*image.shape, n_toa]
            ).copy()
        )

    def test_logical_view_produces_15x900_image(
        self, raw_detector: sc.DataArray
    ) -> None:
        image = specs._logical_view(raw_detector, 'unified_detector')
        assert image.dims == ('arc/tube', 'channel/pixel')
        assert image.sizes == {'arc/tube': 15, 'channel/pixel': 900}

    @pytest.mark.parametrize('pixels_per_tube', [1, 10, 100])
    def test_spectrum_transform_groups_pixels_per_arc(
        self, cumulative_histogram: sc.DataArray, pixels_per_tube: int
    ) -> None:
        params = specs.BifrostSpectrumParams(pixels_per_tube=pixels_per_tube)
        spectrum = specs._bifrost_spectrum_transform(cumulative_histogram, params)
        assert spectrum.dims == ('arc', 'detector_number', 'toa')
        assert spectrum.sizes == {
            'arc': 5,
            'detector_number': 3 * 9 * pixels_per_tube,
            'toa': cumulative_histogram.sizes['toa'],
        }

    def test_spectrum_transform_conserves_counts(
        self, cumulative_histogram: sc.DataArray
    ) -> None:
        params = specs.BifrostSpectrumParams(pixels_per_tube=10)
        spectrum = specs._bifrost_spectrum_transform(cumulative_histogram, params)
        assert_identical(spectrum.sum().data, cumulative_histogram.sum().data)
