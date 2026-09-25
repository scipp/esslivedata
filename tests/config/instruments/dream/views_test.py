# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for DREAM logical detector view transforms."""

import h5py
import numpy as np
import pytest
import scipp as sc

from ess.livedata.config.instruments.dream.views import (
    get_mantle_front_layer,
    get_strip_view,
    get_wire_view,
    logical_detector_number,
)
from ess.livedata.preprocessors.detector_data import get_nexus_geometry_filename

BANKS = [
    'mantle_detector',
    'endcap_backward_detector',
    'endcap_forward_detector',
    'high_resolution_detector',
    'sans_detector',
]
ENDCAPS = ['endcap_backward_detector', 'endcap_forward_detector']
CUBOID_BANKS = ['high_resolution_detector', 'sans_detector']


@pytest.fixture(scope='module')
def geometry() -> dict[str, sc.DataArray]:
    """Pixel positions per bank in mm, sorted by ``detector_number``.

    The positions are lab-frame (``depends_on`` is ``.``) with the beam along +z.
    """
    banks = {}
    with h5py.File(get_nexus_geometry_filename('dream-no-shape')) as f:
        for bank in BANKS:
            group = f['entry/instrument'][bank]
            assert group['x_pixel_offset'].attrs['units'] == 'mm'
            position = np.stack(
                [group[f'{c}_pixel_offset'][()] for c in 'xyz'], axis=-1
            )
            banks[bank] = sc.DataArray(
                sc.vectors(dims=['pixel'], values=position, unit='mm'),
                coords={
                    'detector_number': sc.array(
                        dims=['pixel'], values=group['detector_number'][()]
                    )
                },
            )
    return banks


def _logical_positions(geometry: dict[str, sc.DataArray], bank: str) -> sc.Variable:
    number = logical_detector_number(bank)
    bank_geometry = geometry[bank]
    index = np.searchsorted(
        bank_geometry.coords['detector_number'].values, number.values
    )
    return sc.vectors(
        dims=number.dims, values=bank_geometry.data.values[index], unit='mm'
    )


def _steps(position: sc.Variable, dim: str) -> sc.Variable:
    return position[dim, 1:] - position[dim, :-1]


def _raw(bank: str) -> sc.DataArray:
    """Pixels sorted by ``detector_number``, with the id as the data value."""
    number = np.sort(logical_detector_number(bank).values.ravel())
    return sc.DataArray(sc.array(dims=['detector_number'], values=number))


@pytest.mark.parametrize('bank', BANKS)
def test_logical_ids_are_the_pixels_of_the_geometry(
    geometry: dict[str, sc.DataArray], bank: str
) -> None:
    logical = np.sort(logical_detector_number(bank).values.ravel())
    np.testing.assert_array_equal(
        logical, geometry[bank].coords['detector_number'].values
    )


def test_mantle_strips_run_along_the_beam(
    geometry: dict[str, sc.DataArray],
) -> None:
    position = _logical_positions(geometry, 'mantle_detector')
    strip_step = _steps(position, 'strip')
    assert sc.all(strip_step.fields.z > sc.scalar(5.0, unit='mm')).value
    for dim in ('counter', 'cassette', 'module'):
        z_step = _steps(position, dim).fields.z
        assert sc.all(abs(z_step) < sc.scalar(1e-3, unit='mm')).value


def test_mantle_wires_run_into_the_depth(geometry: dict[str, sc.DataArray]) -> None:
    position = _logical_positions(geometry, 'mantle_detector')
    radius = sc.sqrt(position.fields.x**2 + position.fields.y**2)
    assert sc.all(_steps(radius, 'wire') > sc.scalar(10.0, unit='mm')).value


@pytest.mark.parametrize('bank', ENDCAPS)
def test_endcap_strips_run_away_from_the_sample(
    geometry: dict[str, sc.DataArray], bank: str
) -> None:
    position = _logical_positions(geometry, bank)
    assert sc.all(_steps(sc.norm(position), 'strip') > sc.scalar(0.0, unit='mm')).value


@pytest.mark.parametrize('bank', ENDCAPS)
def test_endcap_wires_run_away_from_the_beam_axis(
    geometry: dict[str, sc.DataArray], bank: str
) -> None:
    position = _logical_positions(geometry, bank)
    radius = sc.sqrt(position.fields.x**2 + position.fields.y**2)
    assert sc.all(_steps(radius, 'wire') > sc.scalar(0.0, unit='mm')).value


@pytest.mark.parametrize('bank', CUBOID_BANKS)
def test_cuboid_strips_run_away_from_the_sample(
    geometry: dict[str, sc.DataArray], bank: str
) -> None:
    position = _logical_positions(geometry, bank)
    assert sc.all(_steps(sc.norm(position), 'strip') > sc.scalar(9.0, unit='mm')).value


@pytest.mark.parametrize('bank', CUBOID_BANKS)
def test_cuboid_wires_and_counters_are_perpendicular_in_every_cuboid(
    geometry: dict[str, sc.DataArray], bank: str
) -> None:
    # Wires have a uniform pitch of 10.62 mm, while the pitch across cassettes and
    # counters alternates between 10.1 and 10.3 mm. A wrong cuboid rotation mixes
    # the two, so each step would be along the wrong axis.
    position = _logical_positions(geometry, bank)
    wire_step = _steps(position, 'wire')
    counter_step = _steps(position, 'counter')
    assert sc.allclose(
        sc.norm(wire_step),
        sc.full_like(sc.norm(wire_step), 10.62),
        atol=sc.scalar(0.05, unit='mm'),
    )
    assert sc.all(sc.norm(counter_step) < sc.scalar(10.5, unit='mm')).value
    wire_step = wire_step['counter', 0]['wire', 0]
    counter_step = counter_step['wire', 0]
    cosine = sc.dot(wire_step, counter_step) / (
        sc.norm(wire_step) * sc.norm(counter_step)
    )
    assert sc.all(abs(cosine) < sc.scalar(0.01)).value


@pytest.mark.parametrize('bank', CUBOID_BANKS)
def test_cuboid_rotations_form_a_pinwheel(
    geometry: dict[str, sc.DataArray], bank: str
) -> None:
    # The cuboids of the four quadrants are rotated by 0, 90, 180 and 270 degrees,
    # so the wires of every cuboid circulate the same way around the beam axis. A
    # rotation must not mirror a cuboid, so the (wire, cassette) axes have the same
    # handedness in every cuboid.
    position = _logical_positions(geometry, bank)
    centre = position.mean(['cassette', 'counter', 'wire', 'strip'])
    wire = _steps(position, 'wire').mean(['cassette', 'counter', 'wire', 'strip'])
    cassette = _steps(position, 'cassette').mean(
        ['cassette', 'counter', 'wire', 'strip']
    )
    z = sc.vector([0.0, 0.0, 1.0])
    circulation = np.sign(sc.dot(sc.cross(centre, wire), z).values)
    handedness = np.sign(sc.dot(sc.cross(wire, cassette), z).values)
    assert len(set(circulation)) == 1
    assert len(set(handedness)) == 1


@pytest.mark.parametrize('bank', BANKS)
def test_transforms_gather_pixels_into_logical_order(bank: str) -> None:
    # With the detector number as data, the transform output must reproduce the
    # ICD layout.
    wire_view = get_wire_view(_raw(bank), bank)
    strip_view = get_strip_view(_raw(bank), bank)
    number = logical_detector_number(bank)
    expected_wire = number.transpose(
        ['strip', 'wire', *(d for d in number.dims if d not in ('strip', 'wire'))]
    ).values.reshape(wire_view.shape)
    np.testing.assert_array_equal(wire_view.values, expected_wire)
    expected_strip = number.transpose(
        [
            'counter',
            'wire',
            'strip',
            *(d for d in number.dims if d not in ('counter', 'wire', 'strip')),
        ]
    ).values.reshape(strip_view.shape)
    np.testing.assert_array_equal(strip_view.values, expected_strip)


@pytest.mark.parametrize(
    ('bank', 'n_wire', 'n_strip'),
    [
        ('mantle_detector', 1920, 7680),
        ('endcap_backward_detector', 9856, 4928),
        ('endcap_forward_detector', 4480, 2240),
        ('high_resolution_detector', 8448, 8448),
        ('sans_detector', 9216, 9216),
    ],
)
def test_views_have_one_pixel_per_wire_and_per_strip(
    bank: str, n_wire: int, n_strip: int
) -> None:
    wire_view = get_wire_view(_raw(bank), bank).sum('strip')
    strip_view = get_strip_view(_raw(bank), bank).sum(['counter', 'wire'])
    assert wire_view.ndim == 2
    assert wire_view.data.size == n_wire
    assert strip_view.ndim == 2
    assert strip_view.data.size == n_strip


def test_endcap_wire_view_uses_icd_wire_numbering() -> None:
    bank = 'endcap_backward_detector'
    view = get_wire_view(_raw(bank), bank)
    # ICD: y = 16 strip + 15 - wire, so wire 0 of strip 0 is in pixel row y = 15.
    first = view['strip', 0]['wire', 0]['sector/cassette/counter', 0]
    assert first.value == 616 * 15 + 1 + 71680


def test_mantle_front_layer_has_one_pixel_per_front_voxel() -> None:
    bank = 'mantle_detector'
    view = get_mantle_front_layer(_raw(bank), bank)
    assert view.sizes == {'module/cassette/counter': 60, 'strip': 256}


def test_transform_rejects_pixel_count_mismatch() -> None:
    da = _raw('sans_detector')['detector_number', :-1]
    with pytest.raises(ValueError, match='voxels'):
        get_wire_view(da, 'sans_detector')
