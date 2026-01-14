# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration tests for detector view workflow via StreamProcessor."""

import numpy as np
import pytest
from scippnexus import NXdetector

from ess.livedata.config.models import ROI, Interval, RectangleROI
from ess.reduce.nexus.types import NeXusData, SampleRun

from .utils import make_fake_ungrouped_nexus_data, make_test_factory, make_test_params


@pytest.mark.slow
class TestIntegrationWithStreamProcessor:
    """Integration tests using the full StreamProcessorWorkflow via factory."""

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

        # After first finalize, cumulative == current (same data)
        assert cumulative1 == current1

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

        # Cumulative should have doubled (events1 + events2)
        assert cumulative2 == pytest.approx(cumulative1 * 2, rel=0.1)

        # Current should be approximately the same as first batch (only events2)
        assert current2 == pytest.approx(current1, rel=0.1)


@pytest.mark.slow
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
