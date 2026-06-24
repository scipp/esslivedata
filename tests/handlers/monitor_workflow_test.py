# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the monitor view workflow using StreamProcessor."""

import copy
import uuid

import pytest
import scipp as sc
from ess.reduce.nexus.types import (
    NeXusComponent,
    NeXusTransformationChain,
    SampleRun,
)
from scippnexus import NXmonitor

from ess.livedata.config.stream import ChainPatchBinding
from ess.livedata.config.value_log import ValueLog
from ess.livedata.config.workflow_spec import JobId
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.handlers.accumulation_mode import Cumulative, Current
from ess.livedata.handlers.accumulators import make_no_copy_accumulator_pair
from ess.livedata.handlers.detector_view_specs import CoordinateModeSettings
from ess.livedata.handlers.monitor_workflow import (
    MONITOR_TRANSFORM,
    accumulated_monitor_histogram,
    build_monitor_workflow,
    counts_in_range,
    counts_total,
    create_monitor_workflow,
    histogram_raw_monitor,
    histogram_wavelength_monitor,
)
from ess.livedata.handlers.monitor_workflow_specs import (
    MonitorDataParams,
    register_monitor_workflow_specs,
)
from ess.livedata.handlers.monitor_workflow_types import (
    AccumulatedMonitorHistogram,
    HistogramEdges,
    HistogramRangeHigh,
    HistogramRangeLow,
    MonitorHistogram,
)
from ess.livedata.parameter_models import (
    TimeUnit,
    TOAEdges,
    TOARange,
    WavelengthEdges,
    WavelengthRangeFilter,
    WavelengthUnit,
)


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

    def test_get_active_edges_wavelength_mode(self):
        """Test get_active_edges returns wavelength edges in wavelength mode."""
        params = MonitorDataParams(
            coordinate_mode=CoordinateModeSettings(mode='wavelength'),
            wavelength_edges=WavelengthEdges(start=0.1, stop=10.0, num_bins=50),
        )
        edges = params.get_active_edges()
        assert edges.dim == 'wavelength'
        assert len(edges) == 51

    def test_get_active_range_returns_none_when_disabled(self):
        """Test get_active_range returns None when range filter is disabled."""
        params = MonitorDataParams(
            coordinate_mode=CoordinateModeSettings(mode='toa'),
        )
        assert params.get_active_range() is None

    @pytest.mark.parametrize(
        'unit', [TimeUnit.NS, TimeUnit.US, TimeUnit.MS, TimeUnit.S]
    )
    def test_get_active_range_toa_mode_preserves_user_unit(self, unit: TimeUnit):
        params = MonitorDataParams(
            coordinate_mode=CoordinateModeSettings(mode='toa'),
            toa_range=TOARange(enabled=True, start=0.0, stop=71.4, unit=unit),
        )
        range_filter = params.get_active_range()
        assert range_filter is not None
        low, high = range_filter
        assert low.unit == unit.value
        assert high.unit == unit.value

    @pytest.mark.parametrize(
        'unit', [WavelengthUnit.ANGSTROM, WavelengthUnit.NANOMETER]
    )
    def test_get_active_range_wavelength_mode_preserves_user_unit(
        self, unit: WavelengthUnit
    ):
        params = MonitorDataParams(
            coordinate_mode=CoordinateModeSettings(mode='wavelength'),
            wavelength_range=WavelengthRangeFilter(
                enabled=True, start=1.0, stop=5.0, unit=unit
            ),
        )
        range_filter = params.get_active_range()
        assert range_filter is not None
        low, high = range_filter
        assert low.unit == unit.value
        assert high.unit == unit.value


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
        result = histogram_raw_monitor(sample_events, toa_edges, geometry=None)
        assert isinstance(result, sc.DataArray)
        assert result.dims == ('time_of_arrival',)
        assert result.sizes['time_of_arrival'] == 5  # 6 edges -> 5 bins
        assert result.unit == 'counts'
        # All 5 events should be histogrammed
        assert result.sum().value == 5.0

    def test_histogram_raw_monitor_preserves_edge_unit(self, sample_events):
        edges = sc.linspace('time_of_arrival', 0, 10, num=6, unit='us')
        result = histogram_raw_monitor(sample_events, edges, geometry=None)
        assert result.coords['time_of_arrival'].unit == 'us'

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
        result = histogram_raw_monitor(histogram, toa_edges, geometry=None)
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
        result = histogram_raw_monitor(histogram, toa_edges, geometry=None)
        assert result.dims == ('time_of_arrival',)
        assert result.sum().value == 10.0

    def test_accumulated_monitor_histogram_returns_same_data(self):
        hist = sc.DataArray(sc.array(dims=['x'], values=[1.0, 2.0], unit='counts'))
        result = accumulated_monitor_histogram(MonitorHistogram(hist))
        assert sc.identical(result, hist)

    @pytest.mark.parametrize('mode', [Current, Cumulative])
    def test_counts_total(self, mode):
        hist = sc.DataArray(
            sc.array(dims=['time_of_arrival'], values=[1.0, 2.0, 3.0], unit='counts')
        )
        result = counts_total(AccumulatedMonitorHistogram[mode](hist))
        assert result.value == 6.0
        assert result.unit == 'counts'

    @pytest.mark.parametrize('mode', [Current, Cumulative])
    def test_counts_in_range(self, mode):
        edges = sc.linspace('time_of_arrival', 0, 10, num=6, unit='ns')
        hist = sc.DataArray(
            sc.array(dims=['time_of_arrival'], values=[1.0, 2.0, 3.0, 4.0, 5.0]),
            coords={'time_of_arrival': edges},
        )
        # Select bins 1-3 (values 2.0, 3.0, 4.0)
        low = HistogramRangeLow(sc.scalar(2, unit='ns'))
        high = HistogramRangeHigh(sc.scalar(8, unit='ns'))
        result = counts_in_range(AccumulatedMonitorHistogram[mode](hist), low, high)
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

    def test_workflow_with_wavelength_mode_requires_lookup_table(self, toa_edges):
        """Test that wavelength mode requires lookup_table_filename."""
        with pytest.raises(ValueError, match="lookup_table_filename is required"):
            create_monitor_workflow(
                'monitor_1', toa_edges, coordinate_mode='wavelength'
            )

    def test_workflow_with_wavelength_mode_requires_geometry_file(self, toa_edges):
        """Test that wavelength mode requires geometry_filename."""
        with pytest.raises(ValueError, match="geometry_filename is required"):
            create_monitor_workflow(
                'monitor_1',
                toa_edges,
                coordinate_mode='wavelength',
                lookup_table_filename='/path/to/lookup.h5',
            )

    def test_context_keys_injected_after_creation(self, toa_edges):
        """Context bindings are injected post-creation, not via the factory."""
        from ess.livedata.handlers.stream_processor_workflow import (
            StreamProcessorWorkflow,
        )

        workflow = create_monitor_workflow('monitor_1', toa_edges)
        assert isinstance(workflow, StreamProcessorWorkflow)
        assert workflow._context_keys == {}

        context_keys = {'position': sc.Variable}
        workflow.add_context_keys(context_keys)
        assert workflow._context_keys == context_keys


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
        workflow.build()

        # Accumulate data
        workflow.accumulate(
            {'monitor_1': sample_binned_events},
            start_time=Timestamp.from_ns(0),
            end_time=Timestamp.from_ns(1000),
        )

        # Finalize to get results
        results = workflow.finalize()

        assert 'cumulative' in results
        assert 'current' in results
        assert 'counts_total' in results
        assert 'counts_in_toa_range' in results
        assert 'counts_total_cumulative' in results
        assert 'counts_in_toa_range_cumulative' in results

        # Check counts
        assert results['cumulative'].sum().value == 5.0
        assert results['current'].sum().value == 5.0
        assert results['counts_total'].value == 5.0
        assert results['counts_in_toa_range'].value == 5.0
        assert results['counts_total_cumulative'].value == 5.0
        assert results['counts_in_toa_range_cumulative'].value == 5.0

    def test_time_coords_on_delta_outputs(self, toa_edges, sample_binned_events):
        """Delta outputs get time, start_time, end_time coords."""
        workflow = create_monitor_workflow('monitor_1', toa_edges)
        workflow.build()
        workflow.accumulate(
            {'monitor_1': sample_binned_events},
            start_time=Timestamp.from_ns(1000),
            end_time=Timestamp.from_ns(2000),
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
        workflow.build()
        workflow.accumulate(
            {'monitor_1': sample_binned_events},
            start_time=Timestamp.from_ns(1000),
            end_time=Timestamp.from_ns(2000),
        )
        results = workflow.finalize()

        assert 'time' not in results['cumulative'].coords
        assert 'start_time' not in results['cumulative'].coords
        assert 'end_time' not in results['cumulative'].coords
        # Cumulative scalar totals also span all time, so no time coords.
        assert 'time' not in results['counts_total_cumulative'].coords
        assert 'time' not in results['counts_in_toa_range_cumulative'].coords

    def test_time_coords_track_first_start_last_end(
        self, toa_edges, sample_binned_events
    ):
        """Time coords should track first start_time and last end_time."""
        workflow = create_monitor_workflow('monitor_1', toa_edges)
        workflow.build()
        # Multiple accumulate calls before finalize
        workflow.accumulate(
            {'monitor_1': sample_binned_events},
            start_time=Timestamp.from_ns(1000),
            end_time=Timestamp.from_ns(2000),
        )
        workflow.accumulate(
            {'monitor_1': sample_binned_events},
            start_time=Timestamp.from_ns(2000),
            end_time=Timestamp.from_ns(3000),
        )
        workflow.accumulate(
            {'monitor_1': sample_binned_events},
            start_time=Timestamp.from_ns(3000),
            end_time=Timestamp.from_ns(4000),
        )
        results = workflow.finalize()

        # start_time should be from first accumulate, end_time from last
        assert results['current'].coords['time'].value == 1000
        assert results['current'].coords['start_time'].value == 1000
        assert results['current'].coords['end_time'].value == 4000

    def test_time_coords_reset_after_finalize(self, toa_edges, sample_binned_events):
        """Time coords should reset between finalize cycles."""
        workflow = create_monitor_workflow('monitor_1', toa_edges)
        workflow.build()

        # First cycle
        workflow.accumulate(
            {'monitor_1': sample_binned_events},
            start_time=Timestamp.from_ns(1000),
            end_time=Timestamp.from_ns(2000),
        )
        results1 = workflow.finalize()
        assert results1['current'].coords['start_time'].value == 1000
        assert results1['current'].coords['end_time'].value == 2000

        # Second cycle should have fresh time tracking
        workflow.accumulate(
            {'monitor_1': sample_binned_events},
            start_time=Timestamp.from_ns(5000),
            end_time=Timestamp.from_ns(6000),
        )
        results2 = workflow.finalize()
        assert results2['current'].coords['start_time'].value == 5000
        assert results2['current'].coords['end_time'].value == 6000

    def test_cumulative_accumulates_window_clears(
        self, toa_edges, sample_binned_events
    ):
        """Verify cumulative accumulates while window clears each cycle."""
        workflow = create_monitor_workflow('monitor_1', toa_edges)
        workflow.build()

        # First cycle
        workflow.accumulate(
            {'monitor_1': sample_binned_events},
            start_time=Timestamp.from_ns(0),
            end_time=Timestamp.from_ns(1000),
        )
        results1 = workflow.finalize()
        assert results1['cumulative'].sum().value == 5.0
        assert results1['current'].sum().value == 5.0

        # Second cycle
        workflow.accumulate(
            {'monitor_1': sample_binned_events},
            start_time=Timestamp.from_ns(1000),
            end_time=Timestamp.from_ns(2000),
        )
        results2 = workflow.finalize()

        # Cumulative has accumulated both cycles
        assert results2['cumulative'].sum().value == 10.0
        assert results2['counts_total_cumulative'].value == 10.0
        assert results2['counts_in_toa_range_cumulative'].value == 10.0
        # Current only has the latest cycle
        assert results2['current'].sum().value == 5.0
        assert results2['counts_total'].value == 5.0
        assert results2['counts_in_toa_range'].value == 5.0

    def test_full_workflow_cycle_histogram_mode(self, toa_edges):
        """Test full workflow cycle with histogram-mode monitor data."""
        workflow = create_monitor_workflow('monitor_1', toa_edges)
        workflow.build()

        # Create histogram data like Cumulative preprocessor produces
        input_edges = sc.linspace('tof', 0, 10, num=11, unit='ns')
        histogram = sc.DataArray(
            sc.array(dims=['tof'], values=[1.0] * 10, unit='counts'),
            coords={'tof': input_edges},
        )

        # Accumulate data
        workflow.accumulate(
            {'monitor_1': histogram},
            start_time=Timestamp.from_ns(0),
            end_time=Timestamp.from_ns(1000),
        )

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

    def test_registered_spec_has_correct_group(self, test_instrument):
        """Verify the spec is registered in the monitor_data group."""
        # Explicit registration
        handle = register_monitor_workflow_specs(
            test_instrument, source_names=['monitor_1', 'monitor_2']
        )
        assert handle is not None
        spec = test_instrument.workflow_factory[handle.workflow_id]
        assert spec.group.name == 'monitor_data'

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
        config = WorkflowConfig(
            identifier=workflow_id,
            job_id=JobId(source_name='monitor_1', job_number=uuid.uuid4()),
        )
        workflow = factory.create(source_name='monitor_1', config=config)
        assert workflow is not None
        assert hasattr(workflow, 'accumulate')


class TestMonitorWorkflowFactoryCoordinateMode:
    """Tests for coordinate mode in monitor workflow factory."""

    def test_wavelength_mode_requires_geometry_and_lookup_table(self):
        """Test that wavelength mode requires geometry and lookup table files.

        The create_monitor_workflow_factory doesn't provide these parameters,
        so wavelength mode should raise ValueError. Instrument-specific factories
        (like DREAM) are responsible for providing these files.
        """
        from ess.livedata.handlers.monitor_workflow_specs import (
            create_monitor_workflow_factory,
        )

        params = MonitorDataParams(
            coordinate_mode=CoordinateModeSettings(mode='wavelength'),
            wavelength_edges=WavelengthEdges(start=0.1, stop=10.0, num_bins=10),
            wavelength_range=WavelengthRangeFilter(
                enabled=True, start=1.0, stop=5.0, unit=WavelengthUnit.ANGSTROM
            ),
        )

        with pytest.raises(ValueError, match="lookup_table_filename is required"):
            create_monitor_workflow_factory('monitor_1', params)


@pytest.mark.slow
class TestDreamMonitorWorkflowFactory:
    """Tests for DREAM-specific monitor workflow factory validation."""

    @pytest.fixture
    def dream_params_wavelength_mode(self):
        """Create DreamMonitorDataParams with wavelength mode enabled."""
        from ess.livedata.config.instruments.dream.specs import DreamMonitorDataParams

        return DreamMonitorDataParams(
            coordinate_mode=CoordinateModeSettings(mode='wavelength'),
        )

    @pytest.mark.parametrize(
        'monitor_name',
        [
            'monitor_bunker',  # Ltotal 6.62 m
            'monitor_cave',  # Ltotal 72.33 m
        ],
    )
    def test_wavelength_mode_allowed_for_all_monitors(
        self, monitor_name, dream_params_wavelength_mode
    ):
        """Test that wavelength mode is allowed for all DREAM monitors.

        Both monitors fall within the DREAM lookup table range (5-80 m).
        """
        from ess.livedata.config.instruments.dream.factories import setup_factories
        from ess.livedata.config.instruments.dream.specs import instrument
        from ess.livedata.config.workflow_spec import WorkflowId

        setup_factories(instrument)
        workflow_id = WorkflowId(
            instrument='dream',
            name='monitor_histogram',
            version=1,
        )
        factory = instrument.workflow_factory.registration(workflow_id).factory

        workflow = factory(monitor_name, dream_params_wavelength_mode)
        assert workflow is not None


@pytest.mark.slow
class TestMonitorWorkflowWavelengthModeHistogramInput:
    """Tests for wavelength coordinate mode with histogram input data.

    These tests require DREAM dependencies (essdiffraction) for the lookup table.
    """

    @pytest.fixture
    def wavelength_edges(self):
        return sc.linspace('wavelength', 0.1, 10.0, num=101, unit='Å')

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

    def test_wavelength_mode_with_histogram_input(
        self, wavelength_edges, geometry_filename, lookup_table_filename
    ):
        """Test wavelength mode workflow with histogram input data.

        Histogram input data from da00/MONITOR_COUNTS.

        This is a regression test for the issue where histogram data with
        frame_time coordinate was not properly handled by the TOA-to-wavelength
        conversion. The essreduce unwrap function expects one of:
        'time_of_flight', 'tof', or 'frame_time' as the coordinate name.
        """
        # Use monitor_cave (Ltotal 72.33 m); both monitors are within the DREAM
        # lookup table range (5-80 m). monitor_bunker has Ltotal of 6.62 m.
        workflow = create_monitor_workflow(
            'monitor_cave',
            wavelength_edges,
            coordinate_mode='wavelength',
            geometry_filename=str(geometry_filename),
            lookup_table_filename=str(lookup_table_filename),
        )
        workflow.build()

        # Create histogram data like fake_monitors da00 mode produces
        # Using 'frame_time' which is what the production data uses
        input_edges = sc.linspace('frame_time', 0, 71_000_000, num=1001, unit='ns')
        histogram = sc.DataArray(
            sc.array(dims=['frame_time'], values=[1.0] * 1000, unit='counts'),
            coords={'frame_time': input_edges},
        )

        # Accumulate data
        workflow.accumulate(
            {'monitor_cave': histogram},
            start_time=Timestamp.from_ns(0),
            end_time=Timestamp.from_ns(1000),
        )

        # Finalize to get results
        results = workflow.finalize()

        assert 'cumulative' in results
        assert 'current' in results
        assert 'counts_total' in results
        assert 'counts_in_toa_range' in results

        # Check that we got valid results. In wavelength mode, counts may not be
        # exactly preserved due to rebinning - some frame_time bins may fall
        # outside the target wavelength range. We verify non-zero counts to confirm
        # the workflow processed data successfully.
        assert results['cumulative'].sum().value > 0
        assert results['current'].sum().value > 0
        assert results['counts_total'].value > 0


class _MonitorZLog(ValueLog):
    """Typed key for a monitor's f144 z-position log (issue #780)."""


# Sentinel so a test can request create_monitor_workflow's own reset_coord default
# rather than passing a value explicitly.
_USE_FACTORY_DEFAULT = object()


@pytest.mark.slow
class TestMonitorMotion:
    """A motorised monitor's position drives the wavelength output (issue #780).

    Mirrors the detector-view dynamic-transform mechanism on a monitor: the
    geometry artifact stores the monitor's motorised translation as a length-0
    NXlog placeholder along its ``depends_on`` chain, and a chain-patch
    :class:`ContextBinding` replaces it with the latest sample of a live f144
    position stream. The monitor workflow itself is unchanged — motion support
    falls out of the same component-agnostic ``StreamProcessorWorkflow`` seam
    the detector uses.

    Drives the real wavelength-mode workflow (DREAM geometry + lookup table) and
    asserts the wavelength spectrum responds to the streamed position: the
    monitor's ``Ltotal`` is a lookup parameter into the time-of-flight table, so
    moving the monitor changes which wavelengths a given arrival time maps to.
    Whether that table is loaded from file or streamed as context does not affect
    the lookup, so this holds for the current file-based table.
    """

    # monitor_cave's depends_on chain holds a single static translation; the test
    # swaps it for a placeholder the streamed position then drives.
    _PATH = '/entry/instrument/monitor_cave/transformations/translation'

    @pytest.fixture
    def wavelength_edges(self):
        return sc.linspace('wavelength', 0.1, 10.0, num=101, unit='angstrom')

    @pytest.fixture
    def geometry_filename(self):
        from ess.livedata.handlers.detector_data_handler import (
            get_nexus_geometry_filename,
        )

        return str(get_nexus_geometry_filename('dream-no-shape'))

    @pytest.fixture
    def lookup_table_filename(self):
        import ess.dream

        return str(
            ess.dream.workflows._get_lookup_table_filename_from_configuration(
                ess.dream.InstrumentConfiguration.high_flux_BC215
            )
        )

    @staticmethod
    def _histogram_input() -> sc.DataArray:
        """A flat monitor spectrum across the frame, in arrival-time bins."""
        edges = sc.linspace('frame_time', 0, 71_000_000, num=1001, unit='ns')
        return sc.DataArray(
            sc.array(dims=['frame_time'], values=[1.0] * 1000, unit='counts'),
            coords={'frame_time': edges},
        )

    @staticmethod
    def _position_log(z: float) -> sc.DataArray:
        """An f144 z-position log holding a single sample, in metres."""
        return sc.DataArray(
            sc.array(dims=['time'], values=[z], unit='m'),
            coords={
                'time': sc.array(dims=['time'], values=[0], unit='ns', dtype='int64')
            },
        )

    def _build(
        self,
        *,
        wavelength_edges,
        geometry_filename,
        lookup_table_filename,
        reset_coord=_USE_FACTORY_DEFAULT,
    ):
        """A wavelength monitor workflow whose monitor_cave position is streamed."""
        extra = (
            {} if reset_coord is _USE_FACTORY_DEFAULT else {'reset_coord': reset_coord}
        )
        workflow = create_monitor_workflow(
            'monitor_cave',
            wavelength_edges,
            coordinate_mode='wavelength',
            geometry_filename=geometry_filename,
            lookup_table_filename=lookup_table_filename,
            **extra,
        )
        pipeline = workflow.base_pipeline
        # Replace the static translation with a length-0 NXlog placeholder, as the
        # geometry artifact stores dynamic geometry; keep all other real fields so
        # the rest of the loaded monitor geometry is untouched.
        chain = copy.deepcopy(
            pipeline.compute(NeXusTransformationChain[NXmonitor, SampleRun])
        )
        chain.transformations[self._PATH].value = sc.DataArray(
            sc.array(dims=['time'], values=[], unit='m', dtype='float64'),
            coords={
                'time': sc.array(dims=['time'], values=[], unit='ns', dtype='int64')
            },
        )
        component = copy.copy(pipeline.compute(NeXusComponent[NXmonitor, SampleRun]))
        component['depends_on'] = chain
        pipeline[NeXusComponent[NXmonitor, SampleRun]] = component

        binding = ChainPatchBinding(
            stream_name='mon_z',
            transform_path=self._PATH,
            workflow_key=_MonitorZLog,
            dependent_sources=frozenset({'monitor_cave'}),
        )
        workflow.build(
            context_keys={'mon_z': _MonitorZLog}, chain_patch_bindings=[binding]
        )
        return workflow

    def _cycle(self, workflow, z, *, start, end):
        """Push one window of monitor data with the monitor parked at ``z``."""
        workflow.accumulate(
            {'monitor_cave': self._histogram_input(), 'mon_z': self._position_log(z)},
            start_time=Timestamp.from_ns(start),
            end_time=Timestamp.from_ns(end),
        )
        return workflow.finalize()

    def test_wavelength_output_tracks_streamed_monitor_position(
        self, wavelength_edges, geometry_filename, lookup_table_filename
    ):
        files = {
            'wavelength_edges': wavelength_edges,
            'geometry_filename': geometry_filename,
            'lookup_table_filename': lookup_table_filename,
        }

        def counts_at(z):
            return self._cycle(self._build(**files), z, start=0, end=1000)[
                'counts_total'
            ].value

        near = counts_at(-4.22)
        far = counts_at(-24.0)

        # Both positions keep the monitor within the lookup table's Ltotal range.
        assert near > 0
        assert far > 0
        # Moving the monitor changes Ltotal, hence the arrival-time -> wavelength
        # mapping, hence how many counts fall in the wavelength range.
        assert near != pytest.approx(far)
        # The streamed value is what drives it: the same position reproduces.
        assert counts_at(-24.0) == pytest.approx(far)

    def test_cumulative_resets_when_monitor_moves(
        self, wavelength_edges, geometry_filename, lookup_table_filename
    ):
        """A move resets the cumulative instead of crashing.

        The monitor workflow defaults to ``reset_coord=MONITOR_TRANSFORM``, so the
        stale pre-move histogram is discarded and the cumulative restarts from the
        new configuration -- after the move it matches the post-move window rather
        than summing across it.
        """
        workflow = self._build(
            wavelength_edges=wavelength_edges,
            geometry_filename=geometry_filename,
            lookup_table_filename=lookup_table_filename,
        )
        self._cycle(workflow, -4.22, start=0, end=1000)
        moved = self._cycle(workflow, -24.0, start=1000, end=2000)
        assert sc.allclose(moved['cumulative'].data, moved['current'].data)

    def test_disabling_reset_raises_on_move(
        self, wavelength_edges, geometry_filename, lookup_table_filename
    ):
        """Opting out (``reset_coord=None``) lets a move crash again.

        Documents what the default guards against: a loaded geometry stamps the
        scalar ``monitor_transform`` coord (and wavelength mode carries ``position``
        too), so summing the cumulative across a move raises on the coord mismatch.
        """
        workflow = self._build(
            wavelength_edges=wavelength_edges,
            geometry_filename=geometry_filename,
            lookup_table_filename=lookup_table_filename,
            reset_coord=None,
        )
        self._cycle(workflow, -4.22, start=0, end=1000)
        with pytest.raises(sc.DatasetError, match=r'position|monitor_transform'):
            self._cycle(workflow, -24.0, start=1000, end=2000)

    def test_toa_mode_with_geometry_stamps_reset_signal(self, geometry_filename):
        """Uniform with the detector view: a loaded geometry stamps the reset
        signal in TOA mode too, not only wavelength mode. The arrival-time
        spectrum does shift when the monitor moves (the flight path changes), so
        the cumulative must reset there as well.
        """
        from ess.reduce.nexus.types import RawMonitor

        edges = sc.linspace('time_of_arrival', 0, 71_000_000, num=11, unit='ns')
        workflow = create_monitor_workflow(
            'monitor_cave',
            edges,
            coordinate_mode='toa',
            geometry_filename=geometry_filename,
        )
        pipeline = workflow.base_pipeline
        toa = sc.array(dims=['event'], values=[1.0, 2.0, 3.0], unit='ns')
        events = sc.DataArray(
            sc.ones(sizes={'event': 3}, dtype='float64', unit='counts'),
            coords={'event_time_offset': toa},
        )
        begin = sc.array(dims=['event_time_zero'], values=[0], unit=None, dtype='int64')
        pipeline[RawMonitor[SampleRun, NXmonitor]] = sc.DataArray(
            sc.bins(begin=begin, dim='event', data=events)
        )

        hist = pipeline.compute(MonitorHistogram)
        assert MONITOR_TRANSFORM in hist.coords
        assert hist.coords[MONITOR_TRANSFORM].ndim == 0


class TestPixellatedMonitorReset:
    """A pixellated monitor breaks the single-point assumption (issue #780).

    The standard monitor histogram collapses the pixel dimension, dropping the
    per-pixel ``position`` coord, so a position-based reset cannot see a move. The
    0-dim ``monitor_transform`` stamped by the histogram providers survives the
    collapse and resets correctly -- the same signal the detector view uses.
    """

    @staticmethod
    def _edges() -> sc.Variable:
        return sc.linspace('wavelength', 0.5, 3.5, num=4, unit='angstrom')

    @staticmethod
    def _pixellated_monitor(z_positions: list[float]) -> sc.DataArray:
        """Binned monitor events over a pixel dim, tagged with per-pixel position."""
        npix = len(z_positions)
        pixel = sc.array(dims=['event'], values=list(range(npix)) * 3, dtype='int64')
        wavelength = sc.array(
            dims=['event'], values=[1.0, 2.0, 3.0] * npix, unit='angstrom'
        )
        events = sc.DataArray(
            sc.ones(dims=['event'], shape=[3 * npix], unit='counts'),
            coords={'wavelength': wavelength, 'detector_number': pixel},
        )
        binned = events.group('detector_number')
        binned.coords['position'] = sc.vectors(
            dims=['detector_number'],
            values=[[0.0, 0.0, z] for z in z_positions],
            unit='m',
        )
        return binned

    def test_pixel_collapse_drops_position_coord(self):
        """Histogramming sums over pixels, so the per-pixel position coord is gone."""
        hist = histogram_wavelength_monitor(
            self._pixellated_monitor([10.0, 11.0]), self._edges(), geometry=None
        )
        assert 'detector_number' not in hist.dims
        assert 'position' not in hist.coords

    def test_position_reset_silently_fails_for_pixellated_monitor(self):
        """Reset keyed on the dropped ``position`` coord is a no-op: the cumulative
        sums across a move instead of restarting -- the bug this change fixes."""
        cumulative, _ = make_no_copy_accumulator_pair(reset_coord='position')
        before = histogram_wavelength_monitor(
            self._pixellated_monitor([10.0, 11.0]), self._edges(), geometry=None
        )
        after = histogram_wavelength_monitor(
            self._pixellated_monitor([20.0, 21.0]), self._edges(), geometry=None
        )
        cumulative.push(before)
        cumulative.push(after)
        assert cumulative.value.sum().value == pytest.approx(
            before.sum().value + after.sum().value
        )

    def test_transform_signal_survives_collapse_and_resets(self):
        """The 0-dim transform stamped by the provider survives the pixel collapse,
        so the default reset coord restarts accumulation on a move."""
        near = sc.spatial.translation(value=[0.0, 0.0, 10.0], unit='m')
        far = sc.spatial.translation(value=[0.0, 0.0, 20.0], unit='m')
        before = histogram_wavelength_monitor(
            self._pixellated_monitor([10.0, 11.0]), self._edges(), geometry=near
        )
        after = histogram_wavelength_monitor(
            self._pixellated_monitor([20.0, 21.0]), self._edges(), geometry=far
        )
        assert MONITOR_TRANSFORM in before.coords  # survived the collapse

        cumulative, _ = make_no_copy_accumulator_pair(reset_coord=MONITOR_TRANSFORM)
        cumulative.push(before)
        cumulative.push(after)
        assert sc.identical(cumulative.value, after)
