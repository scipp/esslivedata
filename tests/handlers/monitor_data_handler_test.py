# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_identical

from ess.livedata import StreamId, StreamKind
from ess.livedata.config.instrument import Instrument
from ess.livedata.config.workflow_spec import WorkflowConfig
from ess.livedata.handlers.accumulators import CollectTOA, Cumulative
from ess.livedata.handlers.monitor_data_handler import (
    MonitorDataParams,
    MonitorHandlerFactory,
    MonitorStreamProcessor,
)
from ess.livedata.handlers.monitor_workflow_specs import (
    register_monitor_workflow_specs,
)
from ess.livedata.parameter_models import TimeUnit, TOAEdges, TOARange


class TestMonitorDataParams:
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


class TestMonitorStreamProcessor:
    @pytest.fixture
    def edges(self):
        """Create test edges."""
        return sc.linspace("tof", 0.0, 100.0, 11, unit="ms")

    @pytest.fixture
    def processor(self, edges):
        """Create MonitorStreamProcessor instance."""
        return MonitorStreamProcessor(edges)

    def test_initialization(self, edges):
        """Test MonitorStreamProcessor initialization."""
        processor = MonitorStreamProcessor(edges)
        # Test public behavior: processor should be able to accumulate data
        toa_data = np.array([10e6])  # Test with minimal data
        processor.accumulate({"det1": toa_data}, start_time=1000, end_time=2000)
        result = processor.finalize()
        assert "cumulative" in result
        assert "current" in result

    def test_accumulate_numpy_array(self, processor):
        """Test accumulation with numpy array (from CollectTOA)."""
        # Create mock TOA data in nanoseconds
        toa_data = np.array(
            [10e6, 25e6, 45e6, 75e6, 95e6]
        )  # 10, 25, 45, 75, 95 ms in ns
        data = {"detector1": toa_data}

        processor.accumulate(data, start_time=1000, end_time=2000)

        # Test by finalizing and checking the result
        result = processor.finalize()
        assert "current" in result
        assert result["current"].dims == ("tof",)
        assert result["current"].unit == sc.units.counts
        # Check that events were histogrammed correctly
        assert result["current"].sum().value > 0

    def test_accumulate_scipp_dataarray(self, processor):
        """Test accumulation with scipp DataArray."""
        # Create mock histogram data
        tof_coords = sc.linspace("time", 5.0, 95.0, 10, unit="ms")
        counts = sc.ones(dims=["time"], shape=[9], unit="counts")
        hist_data = sc.DataArray(data=counts, coords={"time": tof_coords})
        data = {"detector1": hist_data}

        processor.accumulate(data, start_time=1000, end_time=2000)

        # Test by finalizing and checking the result
        result = processor.finalize()
        assert "current" in result
        assert result["current"].dims == ("tof",)
        assert result["current"].unit == sc.units.counts

    def test_accumulate_multiple_calls(self, processor):
        """Test multiple accumulate calls add data correctly."""
        # First accumulation
        toa_data1 = np.array([10e6, 25e6])  # 10, 25 ms in ns
        processor.accumulate({"det1": toa_data1}, start_time=1000, end_time=2000)
        first_result = processor.finalize()
        first_sum = first_result["current"].sum().value

        # Second accumulation - need new processor since finalize clears current
        processor2 = MonitorStreamProcessor(processor._edges)
        toa_data2 = np.array([35e6, 45e6])  # 35, 45 ms in ns
        processor2.accumulate({"det1": toa_data2}, start_time=1000, end_time=2000)
        second_result = processor2.finalize()
        second_sum = second_result["current"].sum().value

        # Both should have data
        assert first_sum > 0
        assert second_sum > 0

    def test_accumulate_wrong_number_of_items(self, processor):
        """Test that accumulate raises error with wrong number of data items."""
        data = {"det1": np.array([10e6]), "det2": np.array([20e6])}

        with pytest.raises(ValueError, match="exactly one data item"):
            processor.accumulate(data, start_time=1000, end_time=2000)

    def test_finalize_first_time(self, processor):
        """Test finalize on first call."""
        toa_data = np.array([10e6, 25e6, 45e6])
        processor.accumulate({"det1": toa_data}, start_time=1000, end_time=2000)

        result = processor.finalize()

        assert "cumulative" in result
        assert "current" in result
        # Check cumulative data (excluding time coord which current has)
        assert_identical(result["cumulative"], result["current"].drop_coords("time"))

        # Verify time coordinate is present
        assert "time" in result["current"].coords
        assert result["current"].coords["time"].value == 1000
        assert result["current"].coords["time"].unit == "ns"

    def test_finalize_subsequent_calls(self, processor):
        """Test finalize accumulates over multiple calls."""
        # First round
        processor.accumulate(
            {"det1": np.array([10e6, 25e6])}, start_time=1000, end_time=2000
        )
        first_result = processor.finalize()
        first_cumulative_sum = first_result["cumulative"].sum().value

        # Second round
        processor.accumulate(
            {"det1": np.array([35e6, 45e6])}, start_time=1000, end_time=2000
        )
        second_result = processor.finalize()
        second_cumulative_sum = second_result["cumulative"].sum().value

        assert second_cumulative_sum > first_cumulative_sum
        # Current should only contain the latest data
        assert second_result["current"].sum().value < second_cumulative_sum

    def test_finalize_without_data(self, processor):
        """Test finalize raises error when no data has been added."""
        with pytest.raises(ValueError, match="No data has been added"):
            processor.finalize()

    def test_finalize_without_accumulate(self, processor):
        """Test finalize raises error without accumulate since last finalize."""
        processor.accumulate(
            {"det1": np.array([10e6, 25e6])}, start_time=1000, end_time=2000
        )
        processor.finalize()

        # After finalize, calling finalize again without accumulate should fail
        with pytest.raises(
            RuntimeError,
            match="finalize called without any data accumulated via accumulate",
        ):
            processor.finalize()

    def test_time_coordinate_tracks_first_accumulate(self, processor):
        """Test time coordinate uses start_time of the first accumulate call."""
        # First accumulate with start_time=1000
        processor.accumulate({"det1": np.array([10e6])}, start_time=1000, end_time=2000)
        # Second accumulate with start_time=3000 (should be ignored)
        processor.accumulate({"det1": np.array([20e6])}, start_time=3000, end_time=4000)

        result = processor.finalize()

        # Time coordinate should use the first start_time
        assert result["current"].coords["time"].value == 1000

        # After finalize, the next accumulate should set a new start_time
        processor.accumulate({"det1": np.array([30e6])}, start_time=5000, end_time=6000)
        result2 = processor.finalize()
        assert result2["current"].coords["time"].value == 5000

    def test_clear(self, processor):
        """Test clear method resets processor state."""
        processor.accumulate(
            {"det1": np.array([10e6, 25e6])}, start_time=1000, end_time=2000
        )
        processor.finalize()

        processor.clear()

        # After clear, should not be able to finalize without new data
        with pytest.raises(ValueError, match="No data has been added"):
            processor.finalize()

    def test_coordinate_unit_conversion(self, processor):
        """Test that coordinates are properly converted to match edges unit."""
        # Create data with different time unit
        tof_coords = sc.linspace("time", 5000.0, 95000.0, 10, unit="us")  # microseconds
        counts = sc.ones(dims=["time"], shape=[9], unit="counts")
        hist_data = sc.DataArray(data=counts, coords={"time": tof_coords})

        processor.accumulate({"det1": hist_data}, start_time=1000, end_time=2000)
        result = processor.finalize()

        assert "current" in result
        assert result["current"].coords["tof"].unit == 'ms'


class TestMonitorStreamProcessorRatemeter:
    """Test integrated ratemeter functionality in MonitorStreamProcessor."""

    @pytest.fixture
    def edges(self):
        """Create test edges: 0-100 ms in 10 bins."""
        return sc.linspace("tof", 0.0, 100.0, 11, unit="ms")

    def test_counts_always_present(self, edges):
        """Test that both counts outputs are always included."""
        processor = MonitorStreamProcessor(edges)
        toa_data = np.array([10e6, 25e6, 45e6])  # 10, 25, 45 ms in ns
        processor.accumulate({"det1": toa_data}, start_time=1000, end_time=2000)

        result = processor.finalize()

        assert "cumulative" in result
        assert "current" in result
        assert "counts_total" in result
        assert "counts_in_toa_range" in result
        # Total counts should equal sum of all events
        assert result["counts_total"].value == 3
        # When TOA range not enabled, counts_in_toa_range equals total
        assert result["counts_in_toa_range"].value == 3
        # Should have time coordinate
        assert "time" in result["counts_total"].coords
        assert result["counts_total"].coords["time"].value == 1000

    def test_counts_in_toa_range_when_enabled(self, edges):
        """Test that counts_in_toa_range is filtered when TOA range enabled."""
        toa_range = (sc.scalar(20.0, unit='ms'), sc.scalar(50.0, unit='ms'))
        processor = MonitorStreamProcessor(
            edges, toa_range_enabled=True, toa_range=toa_range
        )
        toa_data = np.array([10e6, 25e6, 45e6, 75e6])  # 10, 25, 45, 75 ms in ns
        processor.accumulate({"det1": toa_data}, start_time=1000, end_time=2000)

        result = processor.finalize()

        # Both always present
        assert "counts_total" in result
        assert "counts_in_toa_range" in result
        # Total counts includes all events
        assert result["counts_total"].value == 4
        # counts_in_toa_range is filtered (25ms and 45ms inside range)
        assert result["counts_in_toa_range"].value == 2
        # Both should have time coordinate
        assert "time" in result["counts_total"].coords
        assert "time" in result["counts_in_toa_range"].coords

    def test_ratemeter_counts_correct_range(self, edges):
        """Test that ratemeter correctly sums counts in TOA range."""
        # Range 20-50 ms should capture bins [20-30), [30-40), [40-50)
        toa_range = (sc.scalar(20.0, unit='ms'), sc.scalar(50.0, unit='ms'))
        processor = MonitorStreamProcessor(
            edges, toa_range_enabled=True, toa_range=toa_range
        )
        # Events: 10ms (outside), 25ms (inside), 35ms (inside), 75ms (outside)
        toa_data = np.array([10e6, 25e6, 35e6, 75e6])
        processor.accumulate({"det1": toa_data}, start_time=1000, end_time=2000)

        result = processor.finalize()

        # Total counts includes all 4 events
        assert result["counts_total"].value == 4
        # counts_in_toa_range: only 2 events (25ms and 35ms) inside range
        assert result["counts_in_toa_range"].value == 2

    def test_ratemeter_with_create_workflow(self):
        """Test ratemeter through create_workflow factory."""
        params = MonitorDataParams(
            toa_edges=TOAEdges(start=0.0, stop=100.0, num_bins=10, unit=TimeUnit.MS),
            toa_range=TOARange(enabled=True, start=20.0, stop=50.0, unit=TimeUnit.MS),
        )

        processor = MonitorStreamProcessor.create_workflow(params)

        toa_data = np.array([10e6, 25e6, 35e6, 75e6])
        processor.accumulate({"det1": toa_data}, start_time=1000, end_time=2000)
        result = processor.finalize()

        assert result["counts_total"].value == 4
        assert result["counts_in_toa_range"].value == 2

    def test_ratemeter_disabled_via_create_workflow(self):
        """Test that disabled TOA range outputs equal counts."""
        params = MonitorDataParams(
            toa_edges=TOAEdges(start=0.0, stop=100.0, num_bins=10, unit=TimeUnit.MS),
            toa_range=TOARange(enabled=False, start=20.0, stop=50.0, unit=TimeUnit.MS),
        )

        processor = MonitorStreamProcessor.create_workflow(params)

        toa_data = np.array([10e6, 25e6, 35e6, 75e6])
        processor.accumulate({"det1": toa_data}, start_time=1000, end_time=2000)
        result = processor.finalize()

        # Both always present, equal when TOA range disabled
        assert result["counts_total"].value == 4
        assert result["counts_in_toa_range"].value == 4


@pytest.fixture
def test_instrument():
    """Create a test instrument for testing."""
    return Instrument(name="test_instrument")


def test_make_beam_monitor_instrument(test_instrument):
    """Test register_monitor_workflow_specs function."""
    source_names = ["source1", "source2"]

    handle = register_monitor_workflow_specs(test_instrument, source_names)
    handle.attach_factory()(MonitorStreamProcessor.create_workflow)

    factory = test_instrument.workflow_factory

    # Currently there is only one workflow registered
    assert len(factory) == 1
    id, spec = next(iter(factory.items()))
    assert id == spec.get_id()
    assert spec.name == "monitor_histogram"

    processor = factory.create(
        source_name='source1', config=WorkflowConfig(identifier=id, params={})
    )
    assert isinstance(processor, MonitorStreamProcessor)


class TestMonitorHandlerFactory:
    @pytest.fixture
    def mock_instrument(self, test_instrument):
        """Create a mock instrument for testing."""
        handle = register_monitor_workflow_specs(test_instrument, ["source1"])
        handle.attach_factory()(MonitorStreamProcessor.create_workflow)
        return test_instrument

    @pytest.fixture
    def factory(self, mock_instrument):
        """Create MonitorHandlerFactory instance."""
        return MonitorHandlerFactory(instrument=mock_instrument)

    def test_make_preprocessor_monitor_counts(self, factory):
        """Test preprocessor creation for monitor counts."""
        stream_id = StreamId(kind=StreamKind.MONITOR_COUNTS, name="test")

        preprocessor = factory.make_preprocessor(stream_id)

        assert isinstance(preprocessor, Cumulative)
        # Check that clear_on_get is True
        assert preprocessor._clear_on_get is True

    def test_make_preprocessor_monitor_events(self, factory):
        """Test preprocessor creation for monitor events."""
        stream_id = StreamId(kind=StreamKind.MONITOR_EVENTS, name="test")

        preprocessor = factory.make_preprocessor(stream_id)

        assert isinstance(preprocessor, CollectTOA)

    def test_make_preprocessor_other_kind(self, factory):
        """Test preprocessor creation for unsupported stream kind."""
        # Assuming there's another StreamKind that's not supported
        stream_id = StreamId(kind=StreamKind.DETECTOR_EVENTS, name="test")

        preprocessor = factory.make_preprocessor(stream_id)

        assert preprocessor is None

    def test_make_preprocessor_multiple_calls(self, factory):
        """Test that multiple calls create separate instances."""
        stream_id1 = StreamId(kind=StreamKind.MONITOR_COUNTS, name="test1")
        stream_id2 = StreamId(kind=StreamKind.MONITOR_COUNTS, name="test2")

        preprocessor1 = factory.make_preprocessor(stream_id1)
        preprocessor2 = factory.make_preprocessor(stream_id2)

        assert preprocessor1 is not preprocessor2
        assert type(preprocessor1) is type(preprocessor2)
