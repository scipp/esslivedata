# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the monitor view workflow using StreamProcessor."""

import pydantic
import pytest
import scipp as sc

from ess.livedata.handlers.detector_view_specs import CoordinateModeSettings
from ess.livedata.handlers.monitor_workflow import (
    build_monitor_workflow,
    counts_in_range,
    counts_total,
    create_monitor_workflow,
    cumulative_view,
    histogram_raw_monitor,
    window_view,
)
from ess.livedata.handlers.monitor_workflow_specs import (
    MonitorDataParams,
    register_monitor_workflow_specs,
)
from ess.livedata.handlers.monitor_workflow_types import (
    HistogramEdges,
    HistogramRangeHigh,
    HistogramRangeLow,
    MonitorHistogram,
    WindowMonitorHistogram,
)
from ess.livedata.parameter_models import TimeUnit, TOAEdges, TOFEdges, TOFRange


class TestMonitorDataParams:
    """Tests for MonitorDataParams Pydantic model."""

    def test_default_values(self):
        """Test that MonitorDataParams has correct default values."""
        params = MonitorDataParams()

        # Default coordinate mode is TOA
        assert params.coordinate_mode.mode == 'toa'
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

    def test_get_active_edges_toa_mode(self):
        """Test get_active_edges returns TOA edges in TOA mode."""
        params = MonitorDataParams(
            coordinate_mode=CoordinateModeSettings(mode='toa'),
            toa_edges=TOAEdges(start=0.0, stop=100.0, num_bins=50),
        )
        edges = params.get_active_edges()
        assert edges.dim == 'time_of_arrival'
        assert len(edges) == 51

    def test_get_active_edges_tof_mode(self):
        """Test get_active_edges returns TOF edges in TOF mode."""
        params = MonitorDataParams(
            coordinate_mode=CoordinateModeSettings(mode='tof'),
            tof_edges=TOFEdges(start=0.0, stop=100.0, num_bins=50),
        )
        edges = params.get_active_edges()
        assert edges.dim == 'tof'
        assert len(edges) == 51

    def test_get_active_range_returns_none_when_disabled(self):
        """Test get_active_range returns None when range filter is disabled."""
        params = MonitorDataParams(
            coordinate_mode=CoordinateModeSettings(mode='toa'),
        )
        assert params.get_active_range() is None

    def test_get_active_range_tof_mode(self):
        """Test get_active_range returns TOF range in TOF mode."""
        params = MonitorDataParams(
            coordinate_mode=CoordinateModeSettings(mode='tof'),
            tof_range=TOFRange(enabled=True, start=10.0, stop=50.0, unit=TimeUnit.MS),
        )
        range_filter = params.get_active_range()
        assert range_filter is not None
        low, high = range_filter
        assert low.unit == 'ms'
        assert high.unit == 'ms'


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

    def test_histogram_raw_monitor_event_mode(self, sample_events, toa_edges):
        result = histogram_raw_monitor(sample_events, toa_edges)
        assert isinstance(result, sc.DataArray)
        assert result.dims == ('time_of_arrival',)
        assert result.sizes['time_of_arrival'] == 5  # 6 edges -> 5 bins
        assert result.unit == 'counts'
        # All 5 events should be histogrammed
        assert result.sum().value == 5.0

    def test_histogram_raw_monitor_histogram_mode(self, toa_edges):
        """Test rebinning of already-histogrammed monitor data."""
        # Create input histogram with different edges (finer binning)
        input_edges = sc.linspace('tof', 0, 10, num=11, unit='ns')
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        histogram = sc.DataArray(
            sc.array(dims=['tof'], values=values, unit='counts'),
            coords={'tof': input_edges},
        )
        # Rebin to target edges (coarser binning)
        result = histogram_raw_monitor(histogram, toa_edges)
        assert isinstance(result, sc.DataArray)
        assert result.dims == ('time_of_arrival',)
        assert result.sizes['time_of_arrival'] == 5  # 6 edges -> 5 bins
        # Total counts should be preserved after rebinning
        assert result.sum().value == sum(values)

    def test_histogram_raw_monitor_histogram_mode_same_dim(self, toa_edges):
        """Test histogram-mode when input already has same dim name."""
        input_edges = sc.linspace('time_of_arrival', 0, 10, num=11, unit='ns')
        histogram = sc.DataArray(
            sc.array(dims=['time_of_arrival'], values=[1.0] * 10, unit='counts'),
            coords={'time_of_arrival': input_edges},
        )
        result = histogram_raw_monitor(histogram, toa_edges)
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
        low = HistogramRangeLow(sc.scalar(2, unit='ns'))
        high = HistogramRangeHigh(sc.scalar(8, unit='ns'))
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
        from ess.reduce.nexus.types import RawMonitor, SampleRun
        from scippnexus import NXmonitor

        toa = sc.array(dims=['event'], values=[1.0, 2.0, 3.0, 4.0, 5.0], unit='ns')
        weights = sc.ones(sizes={'event': 5}, dtype='float64', unit='counts')
        events = sc.DataArray(data=weights, coords={'event_time_offset': toa})
        sizes = sc.array(dims=['event_time_zero'], values=[5], unit=None, dtype='int64')
        begin = sc.cumsum(sizes, mode='exclusive')
        binned = sc.DataArray(sc.bins(begin=begin, dim='event', data=events))

        edges = sc.linspace('time_of_arrival', 0, 10, num=6, unit='ns')
        # Set RawMonitor directly (bypasses NeXus loading providers)
        workflow[RawMonitor[SampleRun, NXmonitor]] = binned
        workflow[HistogramEdges] = edges
        workflow[HistogramRangeLow] = edges['time_of_arrival', 0]
        workflow[HistogramRangeHigh] = edges['time_of_arrival', -1]

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

    def test_workflow_with_range_filter(self, toa_edges):
        range_filter = (
            sc.scalar(10_000_000, unit='ns'),
            sc.scalar(60_000_000, unit='ns'),
        )
        workflow = create_monitor_workflow(
            'monitor_1', toa_edges, range_filter=range_filter
        )
        assert workflow is not None

    def test_workflow_with_tof_mode_requires_lookup_table(self, toa_edges):
        """Test that TOF mode requires tof_lookup_table_filename."""
        with pytest.raises(ValueError, match="tof_lookup_table_filename is required"):
            create_monitor_workflow('monitor_1', toa_edges, coordinate_mode='tof')

    def test_workflow_with_tof_mode_requires_geometry_file(self, toa_edges):
        """Test that TOF mode requires geometry_filename."""
        with pytest.raises(ValueError, match="geometry_filename is required"):
            create_monitor_workflow(
                'monitor_1',
                toa_edges,
                coordinate_mode='tof',
                tof_lookup_table_filename='/path/to/lookup.h5',
            )


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
    """Tests for explicit spec registration."""

    @pytest.fixture
    def test_instrument(self):
        """Create a minimal test instrument."""
        from ess.livedata.config.instrument import Instrument

        return Instrument(
            name='test_inst_monitor',
            monitors=['monitor_1', 'monitor_2'],
        )

    def test_register_with_empty_source_names_returns_none(self, test_instrument):
        """Explicit registration with empty source names returns None."""
        handle = register_monitor_workflow_specs(test_instrument, source_names=[])
        assert handle is None

    def test_registered_spec_has_correct_namespace(self, test_instrument):
        """Verify the spec is registered in monitor_data namespace."""
        # Explicit registration
        handle = register_monitor_workflow_specs(
            test_instrument, source_names=['monitor_1', 'monitor_2']
        )
        assert handle is not None
        assert handle.workflow_id.namespace == 'monitor_data'

    def test_spec_uses_monitor_data_params(self, test_instrument):
        """Verify the spec uses MonitorDataParams by default."""
        handle = register_monitor_workflow_specs(
            test_instrument, source_names=['monitor_1']
        )
        assert handle is not None

        factory = test_instrument.workflow_factory
        spec = factory[handle.workflow_id]
        assert spec.params is MonitorDataParams

    def test_can_attach_factory_to_handle(self, test_instrument):
        """Test that we can attach the factory to the registered spec."""
        from ess.livedata.config.workflow_spec import WorkflowConfig
        from ess.livedata.handlers.monitor_workflow import create_monitor_workflow

        # Explicit registration
        handle = register_monitor_workflow_specs(
            test_instrument, source_names=['monitor_1', 'monitor_2']
        )
        assert handle is not None

        factory = test_instrument.workflow_factory
        workflow_id = handle.workflow_id

        # Attach factory
        @handle.attach_factory()
        def _factory(source_name: str, params: MonitorDataParams):
            mode = params.coordinate_mode.mode
            if mode == 'wavelength':
                raise NotImplementedError("wavelength mode not yet implemented")
            return create_monitor_workflow(
                source_name=source_name,
                edges=params.get_active_edges(),
                range_filter=params.get_active_range(),
                coordinate_mode=mode,
            )

        # Verify factory works by creating a workflow
        config = WorkflowConfig(identifier=workflow_id)
        workflow = factory.create(source_name='monitor_1', config=config)
        assert workflow is not None
        assert hasattr(workflow, 'accumulate')


class TestMonitorWorkflowFactoryCoordinateMode:
    """Tests for coordinate mode in monitor workflow factory."""

    def test_wavelength_mode_raises_validation_error(self):
        """Test that wavelength mode raises ValidationError."""
        with pytest.raises(
            pydantic.ValidationError, match="wavelength mode is not yet supported"
        ):
            CoordinateModeSettings(mode='wavelength')

    def test_tof_mode_requires_geometry_and_lookup_table(self):
        """Test that TOF mode requires geometry and lookup table files.

        The create_monitor_workflow_factory doesn't provide these parameters,
        so TOF mode should raise ValueError. Instrument-specific factories
        (like DREAM) are responsible for providing these files.
        """
        from ess.livedata.handlers.monitor_workflow_specs import (
            create_monitor_workflow_factory,
        )

        params = MonitorDataParams(
            coordinate_mode=CoordinateModeSettings(mode='tof'),
            tof_edges=TOFEdges(start=0.0, stop=100.0, num_bins=10),
            tof_range=TOFRange(enabled=True, start=20.0, stop=80.0, unit=TimeUnit.MS),
        )

        with pytest.raises(ValueError, match="tof_lookup_table_filename is required"):
            create_monitor_workflow_factory('monitor_1', params)


@pytest.mark.slow
class TestDreamMonitorWorkflowFactory:
    """Tests for DREAM-specific monitor workflow factory validation."""

    @pytest.fixture
    def dream_params_tof_mode(self):
        """Create DreamMonitorDataParams with TOF mode enabled."""
        from ess.livedata.config.instruments.dream.specs import DreamMonitorDataParams

        return DreamMonitorDataParams(
            coordinate_mode=CoordinateModeSettings(mode='tof'),
        )

    def test_tof_mode_rejected_for_monitor_bunker(self, dream_params_tof_mode):
        """Test that TOF mode raises ValueError for monitor_bunker.

        The bunker monitor's flight path (6.62 m) is outside the DREAM TOF
        lookup table range (59.85-80.15 m), so TOF mode is not supported.
        """
        from ess.livedata.config.instruments.dream.factories import setup_factories
        from ess.livedata.config.instruments.dream.specs import instrument
        from ess.livedata.config.workflow_spec import WorkflowId

        setup_factories(instrument)
        workflow_id = WorkflowId(
            instrument='dream',
            namespace='monitor_data',
            name='monitor_histogram',
            version=1,
        )
        factory = instrument.workflow_factory._factories[workflow_id]

        with pytest.raises(
            ValueError, match="TOF mode is not supported for 'monitor_bunker'"
        ):
            factory('monitor_bunker', dream_params_tof_mode)

    def test_tof_mode_allowed_for_monitor_cave(self, dream_params_tof_mode):
        """Test that TOF mode is allowed for monitor_cave.

        The cave monitor's flight path (72.33 m) is within the DREAM TOF
        lookup table range (59.85-80.15 m).
        """
        from ess.livedata.config.instruments.dream.factories import setup_factories
        from ess.livedata.config.instruments.dream.specs import instrument
        from ess.livedata.config.workflow_spec import WorkflowId

        setup_factories(instrument)
        workflow_id = WorkflowId(
            instrument='dream',
            namespace='monitor_data',
            name='monitor_histogram',
            version=1,
        )
        factory = instrument.workflow_factory._factories[workflow_id]

        # Should not raise - cave monitor is compatible with TOF mode
        workflow = factory('monitor_cave', dream_params_tof_mode)
        assert workflow is not None


@pytest.mark.slow
class TestMonitorWorkflowTofModeHistogramInput:
    """Tests for TOF coordinate mode with histogram input data.

    These tests require DREAM dependencies (essdiffraction) for the lookup table.
    """

    @pytest.fixture
    def tof_edges(self):
        return sc.linspace('tof', 0, 71_000_000, num=101, unit='ns')

    @pytest.fixture
    def geometry_filename(self):
        from ess.livedata.handlers.detector_data_handler import (
            get_nexus_geometry_filename,
        )

        return get_nexus_geometry_filename('dream-no-shape')

    @pytest.fixture
    def lookup_table_filename(self):
        import ess.dream

        return ess.dream.workflows._get_lookup_table_filename_from_configuration(
            ess.dream.InstrumentConfiguration.high_flux_BC215
        )

    def test_tof_mode_with_histogram_input(
        self, tof_edges, geometry_filename, lookup_table_filename
    ):
        """Test TOF mode workflow with histogram input data (da00/MONITOR_COUNTS).

        This is a regression test for the issue where histogram data with
        frame_time coordinate was not properly handled by the TOF conversion.
        The essreduce _time_of_flight_data_histogram function expects one of:
        'time_of_flight', 'tof', or 'frame_time' as the coordinate name.
        """
        # Use monitor_cave because its Ltotal (72.33 m) is within the DREAM
        # lookup table range (59.85-80.15 m). monitor_bunker has Ltotal of
        # only 6.62 m which is outside the lookup table range.
        workflow = create_monitor_workflow(
            'monitor_cave',
            tof_edges,
            coordinate_mode='tof',
            geometry_filename=str(geometry_filename),
            tof_lookup_table_filename=str(lookup_table_filename),
        )

        # Create histogram data like fake_monitors da00 mode produces
        # Using 'frame_time' which is what the production data uses
        input_edges = sc.linspace('frame_time', 0, 71_000_000, num=1001, unit='ns')
        histogram = sc.DataArray(
            sc.array(dims=['frame_time'], values=[1.0] * 1000, unit='counts'),
            coords={'frame_time': input_edges},
        )

        # Accumulate data
        workflow.accumulate({'monitor_cave': histogram}, start_time=0, end_time=1000)

        # Finalize to get results
        results = workflow.finalize()

        assert 'cumulative' in results
        assert 'current' in results
        assert 'counts_total' in results
        assert 'counts_in_toa_range' in results

        # Check that we got valid results. In TOF mode, counts may not be
        # exactly preserved due to rebinning - some frame_time bins may fall
        # outside the target TOF range. We verify non-zero counts to confirm
        # the workflow processed data successfully.
        assert results['cumulative'].sum().value > 0
        assert results['current'].sum().value > 0
        assert results['counts_total'].value > 0
