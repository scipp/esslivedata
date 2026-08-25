# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for consuming a streamed wavelength lookup table as workflow context.

The producer concatenates one block per group of components and flattens
essreduce's ``LookupTable`` dataclass onto the result so it fits da00; the
consumer selects its block and rebuilds the dataclass. These tests drive the
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
from ess.livedata.workflows.wavelength_lut_workflow import _flatten_blocks
from ess.livedata.workflows.wavelength_lut_workflow_specs import LUT_STREAM_NAMES


def _block(*, start: float, rows: int, resolution: float = 0.1) -> LookupTableType:
    """A uniform block of ``rows`` rows, values counting up from ``start``."""
    distance = sc.linspace(
        'distance', start, start + (rows - 1) * resolution, rows, unit='m'
    )
    array = sc.DataArray(
        sc.broadcast(distance, sizes={'distance': rows, 'event_time_offset': 2}).to(
            unit='angstrom', copy=True
        ),
        coords={
            'distance': distance,
            'event_time_offset': sc.array(
                dims=['event_time_offset'], values=[0.0, 1.0], unit='ms'
            ),
        },
    )
    return LookupTable[SampleRun, snx.NXdetector](
        array=array,
        pulse_period=sc.scalar(1 / 14, unit='s'),
        pulse_stride=2,
        distance_resolution=sc.scalar(resolution, unit='m'),
        time_resolution=sc.scalar(250.0, unit='us'),
    )


@pytest.fixture
def table() -> LookupTableType:
    return _block(start=10.0, rows=4)


@pytest.fixture
def wire(table: LookupTableType) -> sc.DataArray:
    """A single-block table as it goes on the wire, flattened by the producer."""
    return _flatten_blocks([table])


@pytest.fixture
def two_block_wire() -> sc.DataArray:
    """A monitor-shaped table: two blocks, far apart in flight path."""
    return _flatten_blocks([_block(start=10.0, rows=4), _block(start=70.0, rows=4)])


def _ltotal(value: float) -> sc.Variable:
    return sc.scalar(value, unit='m')


def test_round_trip_restores_the_dataclass(
    wire: sc.DataArray, table: LookupTableType
) -> None:
    restored = detector_lookup_table(wire, _ltotal(10.15))

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
    restored = detector_lookup_table(wire, _ltotal(10.15))

    assert set(restored.array.coords) == {'distance', 'event_time_offset'}


def test_detector_and_monitor_reassemble_to_distinct_keys(
    wire: sc.DataArray,
) -> None:
    """``Component`` is what lets one job hold both tables at once."""
    assert isinstance(detector_lookup_table(wire, _ltotal(10.15)), LookupTable)
    assert isinstance(monitor_lookup_table(wire, _ltotal(10.15)), LookupTable)


def test_missing_scalar_field_coord_fails_loud(wire: sc.DataArray) -> None:
    truncated = wire.drop_coords(['pulse_stride'])

    with pytest.raises(ValueError, match='pulse_stride'):
        detector_lookup_table(truncated, _ltotal(10.15))


class TestBlockSelection:
    """A consumer must receive its own block, never the concatenation.

    essreduce's interpolator locates a row by assuming a uniform distance axis,
    so handing it a table with a gap in it reads the wrong row without saying so.
    """

    @pytest.mark.parametrize(
        ('ltotal', 'expected_start'), [(10.15, 10.0), (70.15, 70.0)]
    )
    def test_selects_the_block_covering_the_flight_path(
        self, two_block_wire: sc.DataArray, ltotal: float, expected_start: float
    ) -> None:
        restored = monitor_lookup_table(two_block_wire, _ltotal(ltotal))

        distance = restored.array.coords['distance']
        assert distance[0].value == pytest.approx(expected_start)
        assert len(distance) == 4

    def test_selected_block_is_uniform(self, two_block_wire: sc.DataArray) -> None:
        restored = monitor_lookup_table(two_block_wire, _ltotal(70.15))

        steps = (
            restored.array.coords['distance'].values[1:]
            - (restored.array.coords['distance'].values[:-1])
        )
        assert steps == pytest.approx(restored.distance_resolution.value)

    def test_selects_by_midpoint_so_a_pixel_off_the_end_still_resolves(
        self, wire: sc.DataArray
    ) -> None:
        # Ltotal reaching past the table is a NaN in the lookup by design; it
        # must not cost the job its whole table.
        ltotal = sc.array(dims=['pixel'], values=[10.05, 10.5], unit='m')

        restored = detector_lookup_table(wire, ltotal)

        assert len(restored.array.coords['distance']) == 4

    def test_flight_path_in_the_gap_fails_loud(
        self, two_block_wire: sc.DataArray
    ) -> None:
        with pytest.raises(ValueError, match='No block'):
            monitor_lookup_table(two_block_wire, _ltotal(40.0))


def test_stream_names_are_prefixed_and_per_group() -> None:
    # Prefixed to stay collision-free against device and motion stream names,
    # and greppable.
    assert LUT_STREAM_NAMES == {
        'detector_lut': 'wavelength_lut/detectors',
        'monitor_lut': 'wavelength_lut/monitors',
    }
