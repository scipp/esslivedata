# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for per-component lookup-table flight-path ranges.

Uses the real DREAM and LOKI geometry artifacts rather than synthetic files:
the point of the derivation is that it agrees with what essreduce computes at
lookup time, which a hand-built file would not exercise. The one exception is
the live rotation axis, which no registered artifact has.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import Filename, NeXusName, SampleRun
from ess.reduce.unwrap import GenericUnwrapWorkflow
from ess.reduce.unwrap.types import DetectorLtotal

from ess.livedata.config.stream import AxisRange
from ess.livedata.preprocessors.detector_data import get_nexus_geometry_filename
from ess.livedata.workflows.lut_ranges import (
    LtotalRangeError,
    component_ltotal_range,
)

#: LOKI's carriage range, plus the sample axes every LOKI detector's Ltotal
#: passes through, mirroring the instrument declaration.
LOKI_AXES = {
    '/entry/instrument/detector_carriage/value': AxisRange(
        lower=sc.scalar(0.0, unit='m'), upper=sc.scalar(15.0, unit='m')
    ),
    **{
        f'/entry/instrument/{axis}/value': AxisRange(
            lower=sc.scalar(-0.5, unit='m'), upper=sc.scalar(0.5, unit='m')
        )
        for axis in (
            'linear_sample_changer',
            'sample_stack_x',
            'sample_stack_y',
            'sample_stack_z',
        )
    },
}

#: Live axes of the synthetic sample, as ``name -> (direction, AxisRange)``.
SAMPLE_AXES = {
    'x': (
        [1.0, 0.0, 0.0],
        AxisRange(lower=sc.scalar(-0.5, unit='m'), upper=sc.scalar(0.5, unit='m')),
    ),
    'z': (
        [0.0, 0.0, 1.0],
        AxisRange(lower=sc.scalar(-300.0, unit='mm'), upper=sc.scalar(0.3, unit='m')),
    ),
}


def _write_sample_geometry(path: str, sample_position: dict[str, float] | None) -> None:
    """Source, a sample on the :data:`SAMPLE_AXES`, and a static detector.

    With ``sample_position`` ``None`` the sample axes are live, stored as the
    artifact writer stores an f144-driven transform: an NXlog with the transform
    attributes and no values. Otherwise each axis is a static transform at the
    given value, in metres.
    """
    with h5py.File(path, 'w') as f:
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        instrument = entry.create_group('instrument')
        instrument.attrs['NX_class'] = 'NXinstrument'

        source = instrument.create_group('source')
        source.attrs['NX_class'] = 'NXsource'
        source.create_dataset('depends_on', data='transformations/distance')
        source_transforms = source.create_group('transformations')
        source_transforms.attrs['NX_class'] = 'NXtransformations'
        distance = source_transforms.create_dataset('distance', data=-20.0)
        distance.attrs.update(
            units='m',
            transformation_type='translation',
            vector=np.array([0.0, 0.0, 1.0]),
            depends_on='.',
        )

        sample = entry.create_group('sample')
        sample.attrs['NX_class'] = 'NXsample'
        sample_transforms = sample.create_group('transformations')
        sample_transforms.attrs['NX_class'] = 'NXtransformations'
        names = list(SAMPLE_AXES)
        sample.create_dataset('depends_on', data=f'transformations/{names[0]}')
        for name, next_name in zip(names, [*names[1:], None], strict=True):
            vector, _ = SAMPLE_AXES[name]
            attrs = {
                'transformation_type': 'translation',
                'vector': np.array(vector),
                'depends_on': '.' if next_name is None else next_name,
            }
            if sample_position is None:
                axis = sample_transforms.create_group(name)
                axis.attrs.update(NX_class='NXlog', writer_module='f144', **attrs)
                values = axis.create_dataset('value', shape=(0,), dtype='float64')
                values.attrs['units'] = 'm'
                time = axis.create_dataset('time', shape=(0,), dtype='uint64')
                time.attrs.update(units='ns', start='1970-01-01T00:00:00Z')
            else:
                axis = sample_transforms.create_dataset(
                    name, data=sample_position[name]
                )
                axis.attrs.update(units='m', **attrs)

        detector = instrument.create_group('detector')
        detector.attrs['NX_class'] = 'NXdetector'
        detector.create_dataset('detector_number', data=np.arange(4))
        for name, values in (
            ('x_pixel_offset', [-1.0, 1.0, -1.0, 1.0]),
            ('y_pixel_offset', [-1.0, -1.0, 1.0, 1.0]),
            ('z_pixel_offset', [5.0, 5.0, 5.0, 5.0]),
        ):
            offsets = detector.create_dataset(name, data=np.array(values))
            offsets.attrs['units'] = 'm'
        detector.create_dataset('depends_on', data='.')


@pytest.fixture(scope='module')
def dream_geometry() -> str:
    return get_nexus_geometry_filename('dream')


@pytest.fixture(scope='module')
def loki_geometry() -> str:
    return get_nexus_geometry_filename('loki')


@pytest.fixture(scope='module')
def live_rotation_geometry(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A detector riding a live rotation axis.

    Synthetic, unlike every other fixture here, because no registered artifact
    has one: the guard against them exists precisely so the first instrument
    that does fails loudly instead of silently getting a table too narrow for
    its swing.
    """
    path = tmp_path_factory.mktemp('geometry') / 'live_rotation.nxs'
    with h5py.File(path, 'w') as f:
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        instrument = entry.create_group('instrument')
        instrument.attrs['NX_class'] = 'NXinstrument'
        stage = instrument.create_group('stage')
        stage.attrs['NX_class'] = 'NXlog'
        # An f144-driven transform, as the artifact writer stores it: an NXlog
        # with the transform attributes and no values at all.
        axis = stage.create_group('value')
        axis.attrs.update(
            NX_class='NXlog',
            transformation_type='rotation',
            vector=np.array([1.0, 0.0, 0.0]),
            depends_on='.',
            writer_module='f144',
        )
        axis.create_dataset('value', shape=(0,), dtype='float64').attrs['units'] = 'deg'
        time = axis.create_dataset('time', shape=(0,), dtype='uint64')
        time.attrs.update(units='ns', start='1970-01-01T00:00:00Z')
        detector = instrument.create_group('detector')
        detector.attrs['NX_class'] = 'NXdetector'
        detector.create_dataset('detector_number', data=np.arange(4))
        for name, values in (
            ('x_pixel_offset', [0.0, 1.0, 0.0, 1.0]),
            ('y_pixel_offset', [0.0, 0.0, 1.0, 1.0]),
            ('z_pixel_offset', [10.0, 10.0, 10.0, 10.0]),
        ):
            offsets = detector.create_dataset(name, data=np.array(values))
            offsets.attrs['units'] = 'm'
        detector.create_dataset('depends_on', data='/entry/instrument/stage/value')
    return str(path)


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
def test_axis_range_spans_both_ends_of_the_travel(loki_geometry: str) -> None:
    """LOKI's rear bank rides the carriage: at the axis lower bound it sits
    ~28.6 m out, and the range reaches the 15 m the carriage travels away from
    the sample without the caller saying which way along the beam that is."""
    start, stop = component_ltotal_range(
        loki_geometry, 'loki_detector_0', is_monitor=False, axis_ranges=LOKI_AXES
    )

    assert start < sc.scalar(28.7, unit='m')
    assert stop > start + sc.scalar(15.0, unit='m')


@pytest.mark.slow
def test_axis_range_is_direction_agnostic(loki_geometry: str) -> None:
    """Swapping the bounds describes the same interval of axis values, so the
    same span comes back: the transform, not the declaration order, decides
    which end of the flight path grows."""
    swapped = {
        **LOKI_AXES,
        '/entry/instrument/detector_carriage/value': AxisRange(
            lower=sc.scalar(15.0, unit='m'), upper=sc.scalar(0.0, unit='m')
        ),
    }

    assert component_ltotal_range(
        loki_geometry, 'loki_detector_0', is_monitor=False, axis_ranges=swapped
    ) == component_ltotal_range(
        loki_geometry, 'loki_detector_0', is_monitor=False, axis_ranges=LOKI_AXES
    )


@pytest.mark.slow
def test_component_riding_a_declared_axis_becomes_placeable(
    loki_geometry: str,
) -> None:
    # Without the declaration there is no position at all; with it, the
    # artifact's geometry resolves.
    with pytest.raises(LtotalRangeError, match='AxisRange'):
        component_ltotal_range(loki_geometry, 'loki_detector_0', is_monitor=False)

    component_ltotal_range(
        loki_geometry, 'loki_detector_0', is_monitor=False, axis_ranges=LOKI_AXES
    )


@pytest.mark.slow
def test_ranges_are_narrow_enough_to_be_worth_blocking(dream_geometry: str) -> None:
    """Why the table is blocks rather than one grid: every component spans a
    handful of rows at the default 0.1 m resolution, while the beamline they sit
    on spans seventy metres."""
    ranges = [
        component_ltotal_range(dream_geometry, name, is_monitor=is_monitor)
        for name, is_monitor in (
            ('mantle_detector', False),
            ('sans_detector', False),
            ('monitor_bunker', True),
            ('monitor_cave', True),
        )
    ]

    for start, stop in ranges:
        assert (stop - start) < sc.scalar(2.0, unit='m')
    assert max(stop for _, stop in ranges) - min(
        start for start, _ in ranges
    ) > sc.scalar(70.0, unit='m')


@pytest.mark.slow
def test_undeclared_live_axis_fails_loud(loki_geometry: str) -> None:
    """``beam_monitor_m4`` rides its own axis, which nothing declares. Failing
    loud is the point: the alternative is a table silently placed at the wrong
    distance."""
    with pytest.raises(LtotalRangeError, match='beam_monitor_m4'):
        component_ltotal_range(
            loki_geometry, 'beam_monitor_m4', is_monitor=True, axis_ranges=LOKI_AXES
        )


def test_live_rotation_axis_is_refused(live_rotation_geometry: str) -> None:
    """Ltotal is not monotonic in an angle, so its extremes need not lie at the
    declared bounds. Refusing costs the component its table, which is handled;
    accepting would give it one that is wrong at the ends of the swing."""
    axis_ranges = {
        '/entry/instrument/stage/value': AxisRange(
            lower=sc.scalar(0.0, unit='deg'), upper=sc.scalar(90.0, unit='deg')
        )
    }

    with pytest.raises(LtotalRangeError, match='rotation'):
        component_ltotal_range(
            live_rotation_geometry,
            'detector',
            is_monitor=False,
            axis_ranges=axis_ranges,
        )


def _detector_ltotal(path: str) -> sc.Variable:
    workflow = GenericUnwrapWorkflow(run_types=[SampleRun], monitor_types=[])
    workflow[Filename[SampleRun]] = path
    workflow[NeXusName[snx.NXdetector]] = 'detector'
    return workflow.compute(DetectorLtotal[SampleRun]).to(unit='m')


@pytest.mark.parametrize(
    'sample_position',
    [
        {'x': 0.0, 'z': 0.0},
        {'x': -0.5, 'z': -0.3},
        {'x': 0.5, 'z': 0.3},
        {'x': -0.5, 'z': 0.3},
        {'x': 0.2, 'z': -0.1},
    ],
)
def test_detector_range_covers_every_sample_position_in_the_box(
    tmp_path, sample_position: dict[str, float]
) -> None:
    """A detector's Ltotal passes through the sample, so the range must hold
    wherever the sample sits within its declared axes, not only at one point."""
    live = str(tmp_path / 'live.nxs')
    _write_sample_geometry(live, sample_position=None)
    placed = str(tmp_path / 'placed.nxs')
    _write_sample_geometry(placed, sample_position=sample_position)
    axis_ranges = {
        f'/entry/sample/transformations/{name}': axis
        for name, (_, axis) in SAMPLE_AXES.items()
    }

    start, stop = component_ltotal_range(
        live, 'detector', is_monitor=False, axis_ranges=axis_ranges
    )

    ltotal = _detector_ltotal(placed)
    assert start <= ltotal.min()
    assert ltotal.max() <= stop


def test_undeclared_sample_axis_fails_loud(tmp_path) -> None:
    live = str(tmp_path / 'live.nxs')
    _write_sample_geometry(live, sample_position=None)

    with pytest.raises(LtotalRangeError, match='AxisRange'):
        component_ltotal_range(live, 'detector', is_monitor=False)
