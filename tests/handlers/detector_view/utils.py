# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Test utilities for detector view tests."""

import numpy as np
import scipp as sc

from ess.livedata.handlers.detector_view.data_source import DetectorNumberSource
from ess.livedata.handlers.detector_view.factory import DetectorViewFactory
from ess.livedata.handlers.detector_view.types import LogicalViewConfig


def make_fake_nexus_detector_data(
    *,
    y_size: int = 4,
    x_size: int = 4,
    n_events_per_pixel: int = 10,
    event_time_offsets: np.ndarray | None = None,
) -> sc.DataArray:
    """Create fake detector data similar to what GenericNeXusWorkflow produces.

    GenericNeXusWorkflow produces binned event data grouped by detector_number,
    with events containing event_time_offset coordinates.

    This format is used by tests that work with RawDetector (already grouped).

    Parameters
    ----------
    y_size:
        Number of pixels in y dimension.
    x_size:
        Number of pixels in x dimension.
    n_events_per_pixel:
        Number of events per pixel (ignored if event_time_offsets is provided).
    event_time_offsets:
        Optional array of event_time_offset values in nanoseconds. If provided,
        must have exactly ``y_size * x_size * n`` elements for some integer n,
        and ``n_events_per_pixel`` is inferred from the array length.
    """
    total_pixels = y_size * x_size

    if event_time_offsets is not None:
        total_events = len(event_time_offsets)
        if total_events % total_pixels != 0:
            raise ValueError(
                f"event_time_offsets length ({total_events}) must be divisible "
                f"by total pixels ({total_pixels})"
            )
        n_events_per_pixel = total_events // total_pixels
        eto_values = event_time_offsets
    else:
        rng = np.random.default_rng(42)
        total_events = total_pixels * n_events_per_pixel
        eto_values = rng.uniform(0, 71_000_000, total_events)

    # Create event table with event_time_offset as coordinate
    # Data values are weights (typically 1.0 for each event)
    events = sc.DataArray(
        data=sc.ones(dims=['event'], shape=[total_events]),
        coords={
            'event_time_offset': sc.array(dims=['event'], values=eto_values, unit='ns'),
        },
    )

    # Create bin indices for each pixel
    begin = sc.arange(
        'detector_number', 0, total_pixels * n_events_per_pixel, n_events_per_pixel
    )
    begin.unit = None
    end = begin + sc.scalar(n_events_per_pixel, unit=None)

    # Bin the events by detector_number - wrap in DataArray
    binned_var = sc.bins(begin=begin, end=end, dim='event', data=events)

    # Create DataArray with detector_number coordinate
    binned = sc.DataArray(
        data=binned_var,
        coords={
            'detector_number': sc.arange(
                'detector_number', 1, total_pixels + 1, unit=None
            )
        },
    )

    return binned


def make_fake_ungrouped_nexus_data(
    *, y_size: int = 4, x_size: int = 4, n_events_per_pixel: int = 10
) -> sc.DataArray:
    """Create fake ungrouped NeXusData for testing with GenericNeXusWorkflow.

    GenericNeXusWorkflow expects ungrouped event data with event_id coordinate
    (detector pixel ID per event). This format is what the preprocessor outputs
    before events are grouped by detector pixel.

    This is different from make_fake_nexus_detector_data which produces already-
    grouped data in RawDetector format.
    """
    rng = np.random.default_rng(42)

    total_pixels = y_size * x_size
    total_events = total_pixels * n_events_per_pixel

    # Create event_time_offset values in nanoseconds (0-71ms range)
    eto_values = rng.uniform(0, 71_000_000, total_events)

    # Create event_id (detector pixel ID) for each event
    # Events are distributed evenly across pixels
    event_ids = np.repeat(np.arange(1, total_pixels + 1), n_events_per_pixel)

    # Create ungrouped event table with event_id coordinate
    # This is the format expected by assemble_detector_data / group_event_data
    events = sc.DataArray(
        data=sc.ones(dims=['event'], shape=[total_events]),
        coords={
            'event_time_offset': sc.array(dims=['event'], values=eto_values, unit='ns'),
            'event_id': sc.array(dims=['event'], values=event_ids, unit=None),
        },
    )

    # Wrap in bins with a pulse dimension (format expected by group_event_data)
    # group_event_data expects begin/end to be 1D arrays, not scalars
    begin = sc.array(dims=['pulse'], values=[0], dtype='int64', unit=None)
    end = sc.array(dims=['pulse'], values=[total_events], dtype='int64', unit=None)
    binned = sc.DataArray(data=sc.bins(begin=begin, end=end, dim='event', data=events))

    return binned


def make_logical_transform(y_size: int, x_size: int):
    """Create a logical transform that folds detector_number to (y, x)."""

    def transform(da: sc.DataArray) -> sc.DataArray:
        return da.fold(dim='detector_number', sizes={'y': y_size, 'x': x_size})

    return transform


def make_fake_empty_detector(y_size: int, x_size: int) -> sc.DataArray:
    """Create a fake EmptyDetector for testing logical projection.

    EmptyDetector is the detector structure without events, used to determine
    output dimensions before any events arrive.
    """
    total_pixels = y_size * x_size
    # Empty bins structure - same as make_fake_nexus_detector_data but with 0 events
    begin = sc.zeros(dims=['detector_number'], shape=[total_pixels], dtype='int64')
    begin.unit = None
    end = begin.copy()  # Same as begin = no events

    # Create empty event table
    events = sc.DataArray(
        data=sc.empty(dims=['event'], shape=[0], dtype='float32', unit='counts'),
        coords={
            'event_time_offset': sc.empty(dims=['event'], shape=[0], unit='ns'),
        },
    )

    binned_var = sc.bins(begin=begin, end=end, dim='event', data=events)

    return sc.DataArray(
        data=binned_var,
        coords={
            'detector_number': sc.arange(
                'detector_number', 1, total_pixels + 1, unit=None
            )
        },
    )


def make_fake_detector_number(y_size: int, x_size: int) -> sc.Variable:
    """Create detector_number Variable for testing with DetectorNumberSource."""
    total_pixels = y_size * x_size
    return sc.arange('detector_number', 1, total_pixels + 1, unit=None)


def make_test_factory(y_size: int = 4, x_size: int = 4) -> DetectorViewFactory:
    """Create a DetectorViewScilineFactory configured for testing.

    Uses DetectorNumberSource for fast, file-less workflow creation.
    """
    detector_number = make_fake_detector_number(y_size, x_size)

    def logical_transform(da: sc.DataArray, source_name: str) -> sc.DataArray:
        return da.fold(dim='detector_number', sizes={'y': y_size, 'x': x_size})

    return DetectorViewFactory(
        data_source=DetectorNumberSource(detector_number),
        view_config=LogicalViewConfig(transform=logical_transform),
    )


def make_test_params():
    """Create minimal DetectorViewParams for testing."""
    from ess.livedata.handlers.detector_view_specs import DetectorViewParams

    return DetectorViewParams()
