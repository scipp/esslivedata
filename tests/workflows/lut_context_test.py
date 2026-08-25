# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for consuming a streamed wavelength lookup table as workflow context.

The producer flattens essreduce's ``LookupTable`` dataclass onto one
``DataArray`` so it fits da00; the consumer rebuilds it. These tests drive the
producer's own flattening rather than a hand-built array, so a change to either
side that breaks the pair fails here.
"""

from __future__ import annotations

import pytest
import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import SampleRun
from ess.reduce.unwrap import LookupTable
from ess.reduce.unwrap.types import LookupTable as LookupTableType

from ess.livedata.workflows.lut_context import (
    detector_lookup_table,
    monitor_lookup_table,
)
from ess.livedata.workflows.wavelength_lut_workflow import _flatten_table
from ess.livedata.workflows.wavelength_lut_workflow_specs import lut_stream_name


@pytest.fixture
def table() -> LookupTableType:
    array = sc.DataArray(
        sc.array(
            dims=['distance', 'event_time_offset'],
            values=[[1.0, 2.0], [3.0, 4.0]],
            unit='angstrom',
        ),
        coords={
            'distance': sc.array(dims=['distance'], values=[10.0, 11.0], unit='m'),
            'event_time_offset': sc.array(
                dims=['event_time_offset'], values=[0.0, 1.0], unit='ms'
            ),
        },
    )
    return LookupTable[SampleRun, snx.NXdetector](
        array=array,
        pulse_period=sc.scalar(1 / 14, unit='s'),
        pulse_stride=2,
        distance_resolution=sc.scalar(0.1, unit='m'),
        time_resolution=sc.scalar(250.0, unit='us'),
    )


@pytest.fixture
def wire(table: LookupTableType) -> sc.DataArray:
    """The table as it goes on the wire, flattened by the producer."""
    return _flatten_table(table)


def test_round_trip_restores_the_dataclass(
    wire: sc.DataArray, table: LookupTableType
) -> None:
    restored = detector_lookup_table(wire)

    assert sc.identical(restored.array, table.array)
    assert sc.identical(restored.pulse_period, table.pulse_period)
    assert restored.pulse_stride == table.pulse_stride
    assert sc.identical(restored.distance_resolution, table.distance_resolution)
    assert sc.identical(restored.time_resolution, table.time_resolution)


def test_round_trip_drops_the_scalar_field_coords_from_the_array(
    wire: sc.DataArray,
) -> None:
    # They ride along on the wire but are dataclass fields, not table axes;
    # leaving them on the array would confuse anything inspecting its coords.
    restored = detector_lookup_table(wire)

    assert set(restored.array.coords) == {'distance', 'event_time_offset'}


def test_detector_and_monitor_reassemble_to_distinct_keys(
    wire: sc.DataArray,
) -> None:
    """``Component`` is what lets one job hold both tables at once."""
    assert isinstance(detector_lookup_table(wire), LookupTable)
    assert isinstance(monitor_lookup_table(wire), LookupTable)


def test_missing_scalar_field_coord_fails_loud(wire: sc.DataArray) -> None:
    truncated = wire.drop_coords(['pulse_stride'])

    with pytest.raises(ValueError, match='pulse_stride'):
        detector_lookup_table(truncated)


def test_stream_name_is_prefixed_and_per_component() -> None:
    # Prefixed to stay collision-free against device and motion stream names,
    # and greppable.
    assert lut_stream_name('mantle_detector') == 'wavelength_lut/mantle_detector'
    assert lut_stream_name('monitor_cave') == 'wavelength_lut/monitor_cave'
