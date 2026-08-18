# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for per-component lookup-table flight-path ranges.

Uses the real DREAM geometry artifact rather than a synthetic file: the point
of the derivation is that it agrees with what essreduce computes at lookup
time, which a hand-built file would not exercise.
"""

from __future__ import annotations

import pytest
import scipp as sc

from ess.livedata.config.stream import MotionEnvelope
from ess.livedata.preprocessors.detector_data import get_nexus_geometry_filename
from ess.livedata.workflows.lut_ranges import (
    LtotalRangeError,
    component_ltotal_range,
    component_ltotal_ranges,
)

#: LOKI's carriage envelope, mirroring the instrument declaration.
LOKI_MOTION = {
    '/entry/instrument/detector_carriage/value': MotionEnvelope(
        nominal=sc.scalar(0.0, unit='mm'), travel=sc.scalar(15.0, unit='m')
    )
}


@pytest.fixture(scope='module')
def dream_geometry() -> str:
    return get_nexus_geometry_filename('dream')


@pytest.fixture(scope='module')
def loki_geometry() -> str:
    return get_nexus_geometry_filename('loki')


@pytest.mark.slow
def test_detector_range_brackets_its_pixels(dream_geometry: str) -> None:
    start, stop = component_ltotal_range(
        dream_geometry, 'mantle_detector', is_monitor=False
    )

    # Scattering geometry: source to sample (~76.5 m) plus the scattered leg.
    assert start < sc.scalar(77.6, unit='m')
    assert stop > sc.scalar(78.6, unit='m')


@pytest.mark.slow
def test_monitor_range_is_padded_around_a_single_distance(
    dream_geometry: str,
) -> None:
    # A monitor sits at one distance, so without padding the range would be
    # degenerate and the table would have no usable width.
    start, stop = component_ltotal_range(
        dream_geometry, 'monitor_bunker', is_monitor=True
    )

    assert start < stop
    assert sc.isclose(stop - start, sc.scalar(0.2, unit='m'))


@pytest.mark.slow
def test_monitor_uses_straight_line_not_scattering_geometry(
    dream_geometry: str,
) -> None:
    """The two Ltotal definitions differ, and picking the wrong one puts the
    table tens of metres from where the consumer queries it."""
    straight, _ = component_ltotal_range(
        dream_geometry, 'monitor_bunker', is_monitor=True
    )

    # The bunker monitor is upstream of the sample (~76.5 m away), so its
    # straight-line Ltotal must be far shorter than any scattering path.
    assert straight < sc.scalar(10.0, unit='m')


@pytest.mark.slow
def test_travel_envelope_extends_the_range_downstream_only(
    loki_geometry: str,
) -> None:
    """LOKI's rear bank rides the carriage: parked at the declared nominal it
    sits ~28.6 m out, and the 15 m of travel extends the range downstream only,
    since the carriage moves away from the sample."""
    start, stop = component_ltotal_range(
        loki_geometry, 'loki_detector_0', is_monitor=False, motion=LOKI_MOTION
    )

    assert start < sc.scalar(28.7, unit='m')
    assert stop > start + sc.scalar(15.0, unit='m')


@pytest.mark.slow
def test_component_riding_a_declared_axis_becomes_placeable(
    loki_geometry: str,
) -> None:
    # Without the declaration there is no position at all; with it, the
    # artifact's geometry resolves.
    with pytest.raises(LtotalRangeError, match='MotionEnvelope'):
        component_ltotal_range(loki_geometry, 'loki_detector_0', is_monitor=False)

    component_ltotal_range(
        loki_geometry, 'loki_detector_0', is_monitor=False, motion=LOKI_MOTION
    )


@pytest.mark.slow
def test_ranges_are_narrow_enough_to_be_worth_splitting(
    dream_geometry: str,
) -> None:
    """The whole point of one table per component: at the default 0.1 m
    resolution each spans a handful of rows, where one instrument-wide table
    spanning source to detector would need hundreds."""
    ranges = component_ltotal_ranges(
        dream_geometry,
        detectors=['mantle_detector', 'sans_detector'],
        monitors=['monitor_bunker', 'monitor_cave'],
    )

    assert set(ranges) == {
        'mantle_detector',
        'sans_detector',
        'monitor_bunker',
        'monitor_cave',
    }
    for start, stop in ranges.values():
        assert (stop - start) < sc.scalar(2.0, unit='m')


@pytest.mark.slow
def test_undeclared_live_axis_fails_loud(loki_geometry: str) -> None:
    """``beam_monitor_m4`` rides its own axis, which nothing declares. Failing
    loud is the point: the alternative is a table silently placed at the wrong
    distance."""
    with pytest.raises(LtotalRangeError, match='beam_monitor_m4'):
        component_ltotal_range(
            loki_geometry, 'beam_monitor_m4', is_monitor=True, motion=LOKI_MOTION
        )
