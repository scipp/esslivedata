# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the monitor view workflow using StreamProcessor."""

import pytest
import scipp as sc

from ess.livedata.handlers.accumulators import NoCopyWindowAccumulator
from ess.livedata.handlers.monitor_workflow import (
    build_monitor_workflow,
    counts_in_range,
    counts_total,
    create_monitor_workflow,
    cumulative_view,
    histogram_monitor_data,
    window_view,
)
from ess.livedata.handlers.monitor_workflow_specs import MonitorDataParams
from ess.livedata.handlers.monitor_workflow_types import (
    MonitorHistogram,
    TOARangeHigh,
    TOARangeLow,
    WindowMonitorHistogram,
)
from ess.livedata.handlers.monitor_workflow_types import (
    TOAEdges as TOAEdgesKey,
)
from ess.livedata.parameter_models import TimeUnit, TOAEdges
from ess.reduce import streaming


class TestMonitorDataParams:
    """Tests for MonitorDataParams Pydantic model."""

    def test_default_values(self):
        """Test that MonitorDataParams has correct default values."""
        params = MonitorDataParams()

        assert params.toa_edges.start == 0.0
        assert params.toa_edges.stop == 1000.0 / 14
        assert params.toa_edges.num_bins == 100
        assert params.toa_edges.unit == TimeUnit.MS
        # Ratemeter defaults
        assert params.toa_range.enabled is False
        assert params.toa_range.start == 0.0
        assert params.toa_range.stop == 10.0

    def test_custom_values(self):
        """Test MonitorDataParams with custom values."""
        custom_edges = TOAEdges(
            start=10.0,
            stop=50.0,
            num_bins=200,
            unit=TimeUnit.US,
        )
        params = MonitorDataParams(toa_edges=custom_edges)

        assert params.toa_edges.start == 10.0
        assert params.toa_edges.stop == 50.0
        assert params.toa_edges.num_bins == 200
        assert params.toa_edges.unit == TimeUnit.US

    def test_get_edges(self):
        """Test that get_edges returns correct scipp Variable."""
        params = MonitorDataParams()
        edges = params.toa_edges.get_edges()

        assert isinstance(edges, sc.Variable)
        assert edges.unit == sc.Unit('ms')
        assert len(edges) == 101  # num_bins + 1


class TestNoCopyWindowAccumulator:
    """Tests for NoCopyWindowAccumulator."""

    def test_is_empty_initially(self):
        acc = NoCopyWindowAccumulator()
        assert acc.is_empty

    def test_push_makes_not_empty(self):
        acc = NoCopyWindowAccumulator()
        acc.push(sc.array(dims=['x'], values=[1.0, 2.0]))
        assert not acc.is_empty

    def test_value_returns_pushed_data(self):
        acc = NoCopyWindowAccumulator()
        data = sc.array(dims=['x'], values=[1.0, 2.0, 3.0])
        acc.push(data)
        result = acc.value
        assert sc.identical(result, data)

    def test_on_finalize_clears_accumulator(self):
        acc = NoCopyWindowAccumulator()
        acc.push(sc.array(dims=['x'], values=[1.0, 2.0]))
        assert not acc.is_empty
        acc.on_finalize()
        assert acc.is_empty

    def test_accumulates_values(self):
        acc = NoCopyWindowAccumulator()
        data1 = sc.array(dims=['x'], values=[1.0, 2.0])
        data2 = sc.array(dims=['x'], values=[3.0, 4.0])
        acc.push(data1)
        acc.push(data2)
        result = acc.value
        expected = data1 + data2
        assert sc.identical(result, expected)

    def test_value_after_on_finalize_raises(self):
        acc = NoCopyWindowAccumulator()
        acc.push(sc.array(dims=['x'], values=[1.0]))
        acc.on_finalize()
        with pytest.raises(ValueError, match="empty"):
            _ = acc.value

    def test_differs_from_eternal_accumulator_behavior(self):
        """NoCopyWindowAccumulator clears after on_finalize.

        Unlike EternalAccumulator which preserves its state."""
        window_acc = NoCopyWindowAccumulator()
        eternal_acc = streaming.EternalAccumulator()

        data = sc.array(dims=['x'], values=[1.0, 2.0])
        window_acc.push(data)
        eternal_acc.push(data)

        # Both are non-empty before on_finalize
        assert not window_acc.is_empty
        assert not eternal_acc.is_empty

        # Call on_finalize
        window_acc.on_finalize()
        eternal_acc.on_finalize()

        # NoCopyWindowAccumulator is cleared, EternalAccumulator is not
        assert window_acc.is_empty
        assert not eternal_acc.is_empty


class TestMonitorWorkflowProviders:
    """Tests for individual workflow provider functions."""

    @pytest.fixture
    def sample_events(self):
        """Create sample binned events like ToNXevent_data produces."""
        toa = sc.array(dims=['event'], values=[1.0, 2.0, 3.0, 4.0, 5.0], unit='ns')
        weights = sc.ones(sizes={'event': 5}, dtype='float64', unit='counts')
        events = sc.DataArray(data=weights, coords={'event_time_offset': toa})
        sizes = sc.array(dims=['event_time_zero'], values=[5], unit=None, dtype='int64')
        begin = sc.cumsum(sizes, mode='exclusive')
        binned = sc.DataArray(sc.bins(begin=begin, dim='event', data=events))
        return binned

    @pytest.fixture
    def toa_edges(self):
        return sc.linspace('time_of_arrival', 0, 10, num=6, unit='ns')

    def test_histogram_monitor_data_event_mode(self, sample_events, toa_edges):
        result = histogram_monitor_data(sample_events, toa_edges)
        assert isinstance(result, sc.DataArray)
        assert result.dims == ('time_of_arrival',)
        assert result.sizes['time_of_arrival'] == 5  # 6 edges -> 5 bins
        assert result.unit == 'counts'
        # All 5 events should be histogrammed
        assert result.sum().value == 5.0

    def test_histogram_monitor_data_histogram_mode(self, toa_edges):
        """Test rebinning of already-histogrammed monitor data."""
        # Create input histogram with different edges (finer binning)
        input_edges = sc.linspace('tof', 0, 10, num=11, unit='ns')
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        histogram = sc.DataArray(
            sc.array(dims=['tof'], values=values, unit='counts'),
            coords={'tof': input_edges},
        )
        # Rebin to target edges (coarser binning)
        result = histogram_monitor_data(histogram, toa_edges)
        assert isinstance(result, sc.DataArray)
        assert result.dims == ('time_of_arrival',)
        assert result.sizes['time_of_arrival'] == 5  # 6 edges -> 5 bins
        # Total counts should be preserved after rebinning
        assert result.sum().value == sum(values)

    def test_histogram_monitor_data_histogram_mode_same_dim(self, toa_edges):
        """Test histogram-mode when input already has same dim name."""
        input_edges = sc.linspace('time_of_arrival', 0, 10, num=11, unit='ns')
        histogram = sc.DataArray(
            sc.array(dims=['time_of_arrival'], values=[1.0] * 10, unit='counts'),
            coords={'time_of_arrival': input_edges},
        )
        result = histogram_monitor_data(histogram, toa_edges)
        assert result.dims == ('time_of_arrival',)
        assert result.sum().value == 10.0

    def test_cumulative_view_returns_same_data(self):
        hist = sc.DataArray(sc.array(dims=['x'], values=[1.0, 2.0], unit='counts'))
        result = cumulative_view(MonitorHistogram(hist))
        assert sc.identical(result, hist)

    def test_window_view_returns_same_data(self):
        hist = sc.DataArray(sc.array(dims=['x'], values=[1.0, 2.0], unit='counts'))
        result = window_view(MonitorHistogram(hist))
        assert sc.identical(result, hist)

    def test_counts_total(self):
        hist = sc.DataArray(
            sc.array(dims=['time_of_arrival'], values=[1.0, 2.0, 3.0], unit='counts')
        )
        result = counts_total(WindowMonitorHistogram(hist))
        assert result.value == 6.0
        assert result.unit == 'counts'

    def test_counts_in_range(self):
        edges = sc.linspace('time_of_arrival', 0, 10, num=6, unit='ns')
        hist = sc.DataArray(
            sc.array(dims=['time_of_arrival'], values=[1.0, 2.0, 3.0, 4.0, 5.0]),
            coords={'time_of_arrival': edges},
        )
        # Select bins 1-3 (values 2.0, 3.0, 4.0)
        low = sc.scalar(2, unit='ns')
        high = sc.scalar(8, unit='ns')
        result = counts_in_range(WindowMonitorHistogram(hist), low, high)
        assert result.value == 9.0  # 2 + 3 + 4


class TestBuildMonitorWorkflow:
    """Tests for the sciline workflow builder."""

    def test_returns_pipeline(self):
        workflow = build_monitor_workflow()
        import sciline

        assert isinstance(workflow, sciline.Pipeline)

    def test_workflow_can_compute_histogram(self):
        """Test that the workflow can compute a histogram from events."""
        workflow = build_monitor_workflow()

        # Create test data
        from scippnexus import NXmonitor

        from ess.reduce.nexus.types import NeXusData, SampleRun

        toa = sc.array(dims=['event'], values=[1.0, 2.0, 3.0, 4.0, 5.0], unit='ns')
        weights = sc.ones(sizes={'event': 5}, dtype='float64', unit='counts')
        events = sc.DataArray(data=weights, coords={'event_time_offset': toa})
        sizes = sc.array(dims=['event_time_zero'], values=[5], unit=None, dtype='int64')
        begin = sc.cumsum(sizes, mode='exclusive')
        binned = sc.DataArray(sc.bins(begin=begin, dim='event', data=events))

        edges = sc.linspace('time_of_arrival', 0, 10, num=6, unit='ns')
        workflow[NeXusData[NXmonitor, SampleRun]] = binned
        workflow[TOAEdgesKey] = edges
        workflow[TOARangeLow] = edges['time_of_arrival', 0]
        workflow[TOARangeHigh] = edges['time_of_arrival', -1]

        result = workflow.compute(MonitorHistogram)
        assert isinstance(result, sc.DataArray)
        assert result.dims == ('time_of_arrival',)
        assert result.sum().value == 5.0


class TestCreateMonitorWorkflow:
    """Tests for the factory function."""

    @pytest.fixture
    def toa_edges(self):
        return sc.linspace('time_of_arrival', 0, 71_000_000, num=101, unit='ns')

    def test_creates_stream_processor_workflow(self, toa_edges):
        from ess.livedata.handlers.stream_processor_workflow import (
            StreamProcessorWorkflow,
        )

        workflow = create_monitor_workflow('monitor_1', toa_edges)
        assert isinstance(workflow, StreamProcessorWorkflow)

    def test_workflow_has_required_methods(self, toa_edges):
        workflow = create_monitor_workflow('monitor_1', toa_edges)
        assert hasattr(workflow, 'accumulate')
        assert hasattr(workflow, 'finalize')
        assert hasattr(workflow, 'clear')

    def test_workflow_with_toa_range(self, toa_edges):
        toa_range = (sc.scalar(10_000_000, unit='ns'), sc.scalar(60_000_000, unit='ns'))
        workflow = create_monitor_workflow('monitor_1', toa_edges, toa_range=toa_range)
        assert workflow is not None


class TestMonitorWorkflowIntegration:
    """Integration tests for the V2 monitor workflow."""

    @pytest.fixture
    def toa_edges(self):
        return sc.linspace('time_of_arrival', 0, 10, num=11, unit='ns')

    @pytest.fixture
    def sample_binned_events(self):
        """Create sample binned events like ToNXevent_data produces."""
        toa = sc.array(dims=['event'], values=[1.5, 2.5, 3.5, 7.5, 8.5], unit='ns')
        weights = sc.ones(sizes={'event': 5}, dtype='float64', unit='counts')
        events = sc.DataArray(data=weights, coords={'event_time_offset': toa})
        sizes = sc.array(dims=['event_time_zero'], values=[5], unit=None, dtype='int64')
        begin = sc.cumsum(sizes, mode='exclusive')
        binned = sc.DataArray(sc.bins(begin=begin, dim='event', data=events))
        return binned

    def test_full_workflow_cycle(self, toa_edges, sample_binned_events):
        workflow = create_monitor_workflow('monitor_1', toa_edges)

        # Accumulate data
        workflow.accumulate(
            {'monitor_1': sample_binned_events}, start_time=0, end_time=1000
        )

        # Finalize to get results
        results = workflow.finalize()

        assert 'cumulative' in results
        assert 'current' in results
        assert 'counts_total' in results
        assert 'counts_in_toa_range' in results

        # Check counts
        assert results['cumulative'].sum().value == 5.0
        assert results['current'].sum().value == 5.0
        assert results['counts_total'].value == 5.0
        assert results['counts_in_toa_range'].value == 5.0

    def test_time_coords_on_delta_outputs(self, toa_edges, sample_binned_events):
        """Delta outputs get time, start_time, end_time coords."""
        workflow = create_monitor_workflow('monitor_1', toa_edges)
        workflow.accumulate(
            {'monitor_1': sample_binned_events}, start_time=1000, end_time=2000
        )
        results = workflow.finalize()

        # Current (window histogram) should have time coords
        assert 'time' in results['current'].coords
        assert 'start_time' in results['current'].coords
        assert 'end_time' in results['current'].coords
        assert results['current'].coords['time'].value == 1000
        assert results['current'].coords['start_time'].value == 1000
        assert results['current'].coords['end_time'].value == 2000

        # counts_total should have time coords
        assert 'time' in results['counts_total'].coords
        assert 'start_time' in results['counts_total'].coords
        assert 'end_time' in results['counts_total'].coords
        assert results['counts_total'].coords['time'].value == 1000
        assert results['counts_total'].coords['start_time'].value == 1000
        assert results['counts_total'].coords['end_time'].value == 2000

        # counts_in_toa_range should have time coords
        assert 'time' in results['counts_in_toa_range'].coords
        assert 'start_time' in results['counts_in_toa_range'].coords
        assert 'end_time' in results['counts_in_toa_range'].coords
        assert results['counts_in_toa_range'].coords['time'].value == 1000
        assert results['counts_in_toa_range'].coords['start_time'].value == 1000
        assert results['counts_in_toa_range'].coords['end_time'].value == 2000

    def test_cumulative_output_has_no_time_coords(
        self, toa_edges, sample_binned_events
    ):
        """Cumulative output should not have time coords (spans all time)."""
        workflow = create_monitor_workflow('monitor_1', toa_edges)
        workflow.accumulate(
            {'monitor_1': sample_binned_events}, start_time=1000, end_time=2000
        )
        results = workflow.finalize()

        assert 'time' not in results['cumulative'].coords
        assert 'start_time' not in results['cumulative'].coords
        assert 'end_time' not in results['cumulative'].coords

    def test_time_coords_track_first_start_last_end(
        self, toa_edges, sample_binned_events
    ):
        """Time coords should track first start_time and last end_time."""
        workflow = create_monitor_workflow('monitor_1', toa_edges)
        # Multiple accumulate calls before finalize
        workflow.accumulate(
            {'monitor_1': sample_binned_events}, start_time=1000, end_time=2000
        )
        workflow.accumulate(
            {'monitor_1': sample_binned_events}, start_time=2000, end_time=3000
        )
        workflow.accumulate(
            {'monitor_1': sample_binned_events}, start_time=3000, end_time=4000
        )
        results = workflow.finalize()

        # start_time should be from first accumulate, end_time from last
        assert results['current'].coords['time'].value == 1000
        assert results['current'].coords['start_time'].value == 1000
        assert results['current'].coords['end_time'].value == 4000

    def test_time_coords_reset_after_finalize(self, toa_edges, sample_binned_events):
        """Time coords should reset between finalize cycles."""
        workflow = create_monitor_workflow('monitor_1', toa_edges)

        # First cycle
        workflow.accumulate(
            {'monitor_1': sample_binned_events}, start_time=1000, end_time=2000
        )
        results1 = workflow.finalize()
        assert results1['current'].coords['start_time'].value == 1000
        assert results1['current'].coords['end_time'].value == 2000

        # Second cycle should have fresh time tracking
        workflow.accumulate(
            {'monitor_1': sample_binned_events}, start_time=5000, end_time=6000
        )
        results2 = workflow.finalize()
        assert results2['current'].coords['start_time'].value == 5000
        assert results2['current'].coords['end_time'].value == 6000

    def test_cumulative_accumulates_window_clears(
        self, toa_edges, sample_binned_events
    ):
        """Verify cumulative accumulates while window clears each cycle."""
        workflow = create_monitor_workflow('monitor_1', toa_edges)

        # First cycle
        workflow.accumulate(
            {'monitor_1': sample_binned_events}, start_time=0, end_time=1000
        )
        results1 = workflow.finalize()
        assert results1['cumulative'].sum().value == 5.0
        assert results1['current'].sum().value == 5.0

        # Second cycle
        workflow.accumulate(
            {'monitor_1': sample_binned_events}, start_time=1000, end_time=2000
        )
        results2 = workflow.finalize()

        # Cumulative has accumulated both cycles
        assert results2['cumulative'].sum().value == 10.0
        # Current only has the latest cycle
        assert results2['current'].sum().value == 5.0

    def test_full_workflow_cycle_histogram_mode(self, toa_edges):
        """Test full workflow cycle with histogram-mode monitor data."""
        workflow = create_monitor_workflow('monitor_1', toa_edges)

        # Create histogram data like Cumulative preprocessor produces
        input_edges = sc.linspace('tof', 0, 10, num=11, unit='ns')
        histogram = sc.DataArray(
            sc.array(dims=['tof'], values=[1.0] * 10, unit='counts'),
            coords={'tof': input_edges},
        )

        # Accumulate data
        workflow.accumulate({'monitor_1': histogram}, start_time=0, end_time=1000)

        # Finalize to get results
        results = workflow.finalize()

        assert 'cumulative' in results
        assert 'current' in results
        assert 'counts_total' in results
        assert 'counts_in_toa_range' in results

        # Check counts (10 total from input histogram)
        assert results['cumulative'].sum().value == 10.0
        assert results['current'].sum().value == 10.0
        assert results['counts_total'].value == 10.0
        assert results['counts_in_toa_range'].value == 10.0


class TestRegisterMonitorWorkflowSpecs:
    """Tests for spec registration."""

    @pytest.fixture
    def test_instrument(self):
        """Create a minimal test instrument."""
        from ess.livedata.config.instrument import Instrument

        return Instrument(
            name='test_inst_monitor',
            monitors=['monitor_1', 'monitor_2'],
        )

    def test_auto_registers_via_post_init(self, test_instrument):
        """Verify the spec is auto-registered via __post_init__."""
        handle = test_instrument._monitor_workflow_handle
        assert handle is not None

    def test_register_with_empty_source_names_returns_none(self):
        # Need a new instrument with no monitors
        from ess.livedata.config.instrument import Instrument

        inst = Instrument(name='test_inst_empty_monitors', monitors=[])
        # The auto-registration should have returned None
        assert inst._monitor_workflow_handle is None

    def test_registered_spec_has_correct_namespace(self, test_instrument):
        """Verify the spec is registered in monitor_data namespace."""
        factory = test_instrument.workflow_factory

        # Find the spec that was registered
        for workflow_id, spec in factory.items():
            if spec.name == 'monitor_histogram':
                assert workflow_id.namespace == 'monitor_data'
                return

        pytest.fail("monitor_histogram spec not found in workflow_factory")

    def test_spec_uses_monitor_data_params(self, test_instrument):
        factory = test_instrument.workflow_factory

        for spec in factory.values():
            if spec.name == 'monitor_histogram':
                assert spec.params is MonitorDataParams
                return

        pytest.fail("monitor_histogram spec not found")

    def test_can_attach_factory_to_handle(self, test_instrument):
        """Test that we can attach the factory to the registered spec."""
        from ess.livedata.handlers.monitor_workflow import create_monitor_workflow

        # Get the handle for monitor_histogram spec
        factory = test_instrument.workflow_factory
        workflow_id = None
        for _workflow_id, spec in factory.items():
            if spec.name == 'monitor_histogram':
                workflow_id = _workflow_id
                break
        if workflow_id is None:
            pytest.fail("monitor_histogram spec not found")

        # Find the handle in the instrument
        handle = test_instrument._monitor_workflow_handle
        assert handle is not None

        # Attach factory
        @handle.attach_factory()
        def _factory(source_name: str, params: MonitorDataParams):
            return create_monitor_workflow(
                source_name=source_name,
                edges=params.toa_edges.get_edges(),
                toa_range=(
                    params.toa_range.range_ns if params.toa_range.enabled else None
                ),
            )

        # Verify factory is attached
        assert workflow_id in factory._factories
