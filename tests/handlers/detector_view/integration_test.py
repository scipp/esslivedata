# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration tests for detector view workflow via StreamProcessor."""

import numpy as np
import pytest
from ess.reduce.nexus.types import NeXusData, SampleRun
from scippnexus import NXdetector

from ess.livedata.config.models import ROI, Interval, RectangleROI

from .utils import make_fake_ungrouped_nexus_data, make_test_factory, make_test_params


class TestIntegrationWithStreamProcessor:
    """Integration tests using the full StreamProcessorWorkflow via factory."""

    def test_window_outputs_have_time_coords(self):
        """Test that window outputs have time, start_time, end_time coords.

        The factory configures current, counts_total, counts_in_toa_range, and
        roi_spectra_current as window outputs. These should have time coords;
        cumulative outputs should NOT have them (added later by Job).
        """
        factory = make_test_factory(y_size=4, x_size=4)
        workflow = factory.make_workflow('detector', params=make_test_params())

        events = make_fake_ungrouped_nexus_data(
            y_size=4, x_size=4, n_events_per_pixel=10
        )

        # Add an ROI so roi_spectra outputs are non-empty
        roi = RectangleROI(
            x=Interval(min=0, max=2, unit=None), y=Interval(min=0, max=2, unit=None)
        )
        rectangle_request = ROI.to_concatenated_data_array({0: roi})

        workflow.accumulate(
            {
                'detector': NeXusData[NXdetector, SampleRun](events),
                'roi_rectangle': rectangle_request,
            },
            start_time=1000,
            end_time=2000,
        )
        result = workflow.finalize()

        # Window outputs should have time coords
        for key in ('current', 'counts_total', 'counts_in_toa_range'):
            assert 'time' in result[key].coords, f"{key} missing 'time' coord"
            assert 'start_time' in result[key].coords, f"{key} missing 'start_time'"
            assert 'end_time' in result[key].coords, f"{key} missing 'end_time'"
            assert result[key].coords['time'].value == 1000
            assert result[key].coords['start_time'].value == 1000
            assert result[key].coords['end_time'].value == 2000
            assert result[key].coords['time'].unit == 'ns'

        # ROI spectra current should also have time coords
        assert 'time' in result['roi_spectra_current'].coords
        assert result['roi_spectra_current'].coords['time'].value == 1000

        # Cumulative outputs should NOT have time coords at this level.
        # The Job layer (not tested here) adds start_time/end_time spanning
        # the full job duration to any output that doesn't already have them.
        assert 'time' not in result['cumulative'].coords
        assert 'start_time' not in result['cumulative'].coords
        assert 'time' not in result['roi_spectra_cumulative'].coords
        assert 'time' not in result['counts_total_cumulative'].coords
        assert 'start_time' not in result['counts_total_cumulative'].coords
        assert 'time' not in result['counts_in_toa_range_cumulative'].coords
        assert 'start_time' not in result['counts_in_toa_range_cumulative'].coords

    def test_full_workflow_accumulate_and_finalize(self):
        """Test the full workflow with accumulate and finalize."""
        # Use factory to create workflow (same code path as production)
        factory = make_test_factory(y_size=4, x_size=4)
        workflow = factory.make_workflow('detector', params=make_test_params())

        # Create fake ungrouped events (format expected by GenericNeXusWorkflow)
        events = make_fake_ungrouped_nexus_data(
            y_size=4, x_size=4, n_events_per_pixel=10
        )

        # Accumulate (provide as NeXusData, which factory expects)
        workflow.accumulate(
            {'detector': NeXusData[NXdetector, SampleRun](events)},
            start_time=1000,
            end_time=2000,
        )

        # Finalize
        result = workflow.finalize()

        assert 'cumulative' in result
        assert 'current' in result
        assert 'counts_total' in result
        assert 'counts_in_toa_range' in result
        assert 'counts_total_cumulative' in result
        assert 'counts_in_toa_range_cumulative' in result

        # Verify output shapes
        assert result['cumulative'].dims == ('y', 'x')
        assert result['cumulative'].sizes == {'y': 4, 'x': 4}

    def test_cumulative_accumulates_current_resets(self):
        """Test that cumulative accumulates and current resets after finalize."""
        # Use factory to create workflow (same code path as production)
        factory = make_test_factory(y_size=4, x_size=4)
        workflow = factory.make_workflow('detector', params=make_test_params())

        # First batch - use ungrouped format
        events1 = make_fake_ungrouped_nexus_data(
            y_size=4, x_size=4, n_events_per_pixel=10
        )
        workflow.accumulate(
            {'detector': NeXusData[NXdetector, SampleRun](events1)},
            start_time=1000,
            end_time=2000,
        )
        result1 = workflow.finalize()

        cumulative1 = result1['cumulative'].sum().value
        current1 = result1['current'].sum().value
        counts_total_cum1 = result1['counts_total_cumulative'].value
        counts_total_cur1 = result1['counts_total'].value

        # After first finalize, cumulative == current (same data)
        assert cumulative1 == current1
        assert counts_total_cum1 == counts_total_cur1

        # Second batch - use ungrouped format
        events2 = make_fake_ungrouped_nexus_data(
            y_size=4, x_size=4, n_events_per_pixel=10
        )
        workflow.accumulate(
            {'detector': NeXusData[NXdetector, SampleRun](events2)},
            start_time=2000,
            end_time=3000,
        )
        result2 = workflow.finalize()

        cumulative2 = result2['cumulative'].sum().value
        current2 = result2['current'].sum().value
        counts_total_cum2 = result2['counts_total_cumulative'].value
        counts_total_cur2 = result2['counts_total'].value

        # Cumulative should have doubled (events1 + events2)
        assert cumulative2 == pytest.approx(cumulative1 * 2, rel=0.1)
        assert counts_total_cum2 == pytest.approx(counts_total_cum1 * 2, rel=0.1)

        # Current should be approximately the same as first batch (only events2)
        assert current2 == pytest.approx(current1, rel=0.1)
        assert counts_total_cur2 == pytest.approx(counts_total_cur1, rel=0.1)


class TestROISpectraIntegration:
    """Integration tests for ROI spectra with StreamProcessor via factory."""

    def test_roi_spectra_via_context_keys(self):
        """Test ROI spectra extraction via context_keys in StreamProcessorWorkflow."""
        # Use factory to create workflow (same code path as production)
        factory = make_test_factory(y_size=4, x_size=4)
        workflow = factory.make_workflow('detector', params=make_test_params())

        # Create fake ungrouped events (format expected by GenericNeXusWorkflow)
        events = make_fake_ungrouped_nexus_data(
            y_size=4, x_size=4, n_events_per_pixel=10
        )

        # First, accumulate events without ROI
        workflow.accumulate(
            {'detector': NeXusData[NXdetector, SampleRun](events)},
            start_time=1000,
            end_time=2000,
        )
        result1 = workflow.finalize()

        # Should have empty ROI spectra (no ROIs configured)
        assert result1['roi_spectra_cumulative'].sizes['roi'] == 0
        assert result1['roi_spectra_current'].sizes['roi'] == 0

        # Now add an ROI via context_keys and accumulate more events
        roi = RectangleROI(
            x=Interval(min=0, max=2, unit=None), y=Interval(min=0, max=2, unit=None)
        )
        rectangle_request = ROI.to_concatenated_data_array({0: roi})

        workflow.accumulate(
            {
                'detector': NeXusData[NXdetector, SampleRun](events),
                'roi_rectangle': rectangle_request,
            },
            start_time=2000,
            end_time=3000,
        )
        result2 = workflow.finalize()

        # Should now have ROI spectra
        assert result2['roi_spectra_cumulative'].sizes['roi'] == 1
        assert result2['roi_spectra_current'].sizes['roi'] == 1

        # Cumulative should include both batches
        # Current should only include second batch
        cumulative_sum = result2['roi_spectra_cumulative'].sum().value
        current_sum = result2['roi_spectra_current'].sum().value

        # Cumulative should be ~2x current (two batches)
        assert cumulative_sum > current_sum

    def test_roi_change_recomputes_from_accumulated_histogram(self):
        """Test that changing ROI recomputes spectra from full accumulated data."""
        # Use factory to create workflow (same code path as production)
        factory = make_test_factory(y_size=4, x_size=4)
        workflow = factory.make_workflow('detector', params=make_test_params())

        # Create ungrouped events with some reproducibility
        np.random.seed(42)
        events = make_fake_ungrouped_nexus_data(
            y_size=4, x_size=4, n_events_per_pixel=100
        )

        # Set initial ROI (small region)
        roi_small = RectangleROI(
            x=Interval(min=0, max=1, unit=None), y=Interval(min=0, max=1, unit=None)
        )
        rectangle_request_small = ROI.to_concatenated_data_array({0: roi_small})

        workflow.accumulate(
            {
                'detector': NeXusData[NXdetector, SampleRun](events),
                'roi_rectangle': rectangle_request_small,
            },
            start_time=1000,
            end_time=2000,
        )
        result1 = workflow.finalize()
        small_roi_sum = result1['roi_spectra_cumulative'].sum().value

        # Change ROI to larger region
        roi_large = RectangleROI(
            x=Interval(min=0, max=4, unit=None), y=Interval(min=0, max=4, unit=None)
        )
        rectangle_request_large = ROI.to_concatenated_data_array({0: roi_large})

        # Accumulate same events again with new ROI
        # (workflow requires dynamic data for accumulation)
        workflow.accumulate(
            {
                'detector': NeXusData[NXdetector, SampleRun](events),
                'roi_rectangle': rectangle_request_large,
            },
            start_time=2000,
            end_time=3000,
        )
        result2 = workflow.finalize()
        large_roi_sum = result2['roi_spectra_cumulative'].sum().value

        # Larger ROI should capture more counts from the accumulated data.
        # Even though we added the same events twice, the key point is that
        # the large ROI captures more of the full accumulated histogram.
        # small_roi covers 1/16 of detector (1x1 out of 4x4)
        # large_roi covers full detector (4x4 out of 4x4)
        assert large_roi_sum > small_roi_sum

    def test_roi_readback_is_always_published(self):
        """Test that ROI readback is published on every finalize when ROI is set.

        The current implementation always publishes readbacks when ROIs are
        configured, regardless of whether they changed. This simplifies the
        implementation and ensures downstream consumers always have current state.
        """
        factory = make_test_factory(y_size=4, x_size=4)
        workflow = factory.make_workflow('detector', params=make_test_params())

        events = make_fake_ungrouped_nexus_data(
            y_size=4, x_size=4, n_events_per_pixel=10
        )

        roi = RectangleROI(
            x=Interval(min=1.0, max=3.0, unit=None),
            y=Interval(min=1.0, max=3.0, unit=None),
        )
        rectangle_request = ROI.to_concatenated_data_array({0: roi})

        # First accumulate with ROI
        workflow.accumulate(
            {
                'detector': NeXusData[NXdetector, SampleRun](events),
                'roi_rectangle': rectangle_request,
            },
            start_time=1000,
            end_time=2000,
        )
        result1 = workflow.finalize()

        # Readback should be published
        assert 'roi_rectangle' in result1

        # Verify readback round-trips correctly
        readback = result1['roi_rectangle']
        recovered_rois = ROI.from_concatenated_data_array(readback)
        assert 0 in recovered_rois
        assert recovered_rois[0].x.min == 1.0
        assert recovered_rois[0].x.max == 3.0

        # Second accumulate with SAME ROI (unchanged)
        workflow.accumulate(
            {
                'detector': NeXusData[NXdetector, SampleRun](events),
                'roi_rectangle': rectangle_request,
            },
            start_time=2000,
            end_time=3000,
        )
        result2 = workflow.finalize()

        # Readback is still published (current behavior: always publish)
        assert 'roi_rectangle' in result2

    def test_stacked_spectra_sorted_by_roi_index(self):
        """Test that ROI spectra are sorted by index in the output.

        When multiple ROIs are configured with non-contiguous indices,
        the stacked output should have them sorted by index.
        """
        factory = make_test_factory(y_size=4, x_size=4)
        workflow = factory.make_workflow('detector', params=make_test_params())

        events = make_fake_ungrouped_nexus_data(
            y_size=4, x_size=4, n_events_per_pixel=10
        )

        # Create ROIs with indices 5 and 2 (out of order)
        roi_5 = RectangleROI(
            x=Interval(min=0, max=2, unit=None), y=Interval(min=0, max=2, unit=None)
        )
        roi_2 = RectangleROI(
            x=Interval(min=2, max=4, unit=None), y=Interval(min=2, max=4, unit=None)
        )
        rectangle_request = ROI.to_concatenated_data_array({5: roi_5, 2: roi_2})

        workflow.accumulate(
            {
                'detector': NeXusData[NXdetector, SampleRun](events),
                'roi_rectangle': rectangle_request,
            },
            start_time=1000,
            end_time=2000,
        )
        result = workflow.finalize()

        stacked = result['roi_spectra_current']

        # Should have 2 ROIs
        assert stacked.sizes['roi'] == 2

        # ROI indices should be sorted: [2, 5] not [5, 2]
        roi_indices = list(stacked.coords['roi'].values)
        assert roi_indices == [
            2,
            5,
        ], f"Expected sorted indices [2, 5], got {roi_indices}"


class TestUnitHandling:
    """Tests for unit handling in detector view workflow."""

    def test_toa_edges_with_microsecond_units(self):
        """Test that TOA edges in microseconds work correctly.

        The workflow should handle unit conversion internally and produce
        output with the user-specified unit.
        """
        from ess.livedata.handlers.detector_view_specs import DetectorViewParams
        from ess.livedata.parameter_models import TimeUnit, TOAEdges

        # Create params with microsecond TOA edges
        # Events have event_time_offset in nanoseconds (0-71ms = 0-71000 us)
        params = DetectorViewParams(
            toa_edges=TOAEdges(start=0.0, stop=80000.0, num_bins=10, unit=TimeUnit.US)
        )

        factory = make_test_factory(y_size=4, x_size=4)
        workflow = factory.make_workflow('detector', params=params)

        events = make_fake_ungrouped_nexus_data(
            y_size=4, x_size=4, n_events_per_pixel=10
        )

        workflow.accumulate(
            {'detector': NeXusData[NXdetector, SampleRun](events)},
            start_time=1000,
            end_time=2000,
        )
        result = workflow.finalize()

        # Output should have events binned
        assert result['current'].sum().value > 0

        # Verify the output has the expected histogram dimension
        # (time_of_arrival from the edges, projected down to 2D for detector image)
        assert result['current'].dims == ('y', 'x')

    def test_roi_spectra_output_structure(self):
        """Test that ROI spectra output has correct structure and units.

        The output should be a 2D DataArray with dims (roi, toa_dim) and
        have the correct coordinates and unit.
        """
        factory = make_test_factory(y_size=4, x_size=4)
        workflow = factory.make_workflow('detector', params=make_test_params())

        events = make_fake_ungrouped_nexus_data(
            y_size=4, x_size=4, n_events_per_pixel=10
        )

        roi = RectangleROI(
            x=Interval(min=0, max=2, unit=None), y=Interval(min=0, max=2, unit=None)
        )
        rectangle_request = ROI.to_concatenated_data_array({0: roi})

        workflow.accumulate(
            {
                'detector': NeXusData[NXdetector, SampleRun](events),
                'roi_rectangle': rectangle_request,
            },
            start_time=1000,
            end_time=2000,
        )
        result = workflow.finalize()

        spectra = result['roi_spectra_current']

        # Should be 2D: (roi, spectral_dim)
        assert spectra.ndim == 2
        assert 'roi' in spectra.dims

        # Should have roi coordinate
        assert 'roi' in spectra.coords

        # Second dimension should be time_of_arrival (the spectral dimension)
        assert 'time_of_arrival' in spectra.dims

        # Data should have a unit (dimensionless from histogram)
        assert spectra.data.unit is not None
