# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Tests for TBL spec registration."""

import numpy as np
import pytest
import scipp as sc
import scippnexus as snx

from ess.livedata.config.instrument import Instrument, instrument_registry
from ess.livedata.config.instruments import get_config
from ess.livedata.config.instruments.tbl import specs
from ess.livedata.config.instruments.tbl.specs import GEOMETRY_FILE_Z_POSITIONS
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.preprocessors.detector_data import get_nexus_geometry_filename
from ess.livedata.preprocessors.downsample_pixel_ids import DownsamplePixelIds
from ess.livedata.preprocessors.group_by_pixel import GroupByPixel
from ess.livedata.preprocessors.to_nxevent_data import DetectorEvents, ToNXevent_data


def _park_live_axes(component: sc.DataGroup) -> None:
    """Hold every stream-driven transform of the component's chain at zero.

    The artifact stores such a transform as an empty log, which leaves the
    component without a position. The beam monitor rides a live x axis, so
    zero leaves its z position as the file states it.
    """
    chain = component.get('depends_on')
    if not isinstance(chain, snx.TransformationChain):
        return
    for transform in chain.transformations.values():
        if isinstance(transform.value, sc.DataArray) and 'time' in transform.value.dims:
            transform.value = sc.scalar(0.0, unit=transform.value.unit)


# The geometry file carries no data, so scippnexus falls back to loading each
# component as a plain DataGroup. Positions still resolve from depends_on.
@pytest.mark.filterwarnings('ignore:Failed to load:UserWarning')
def test_positions_shown_in_descriptions_match_geometry_file() -> None:
    """The view descriptions warn users with these positions; they must state
    what wavelength mode actually computes with."""
    with snx.File(get_nexus_geometry_filename('tbl')) as f:
        instrument = f['entry/instrument']
        for name, z in GEOMETRY_FILE_Z_POSITIONS.items():
            component = instrument[name][()]
            _park_live_axes(component)
            position = snx.compute_positions(component)['position']
            assert position.fields.z.to(unit='m').mean().value == pytest.approx(
                z, abs=1e-6
            ), name


@pytest.fixture(scope='module')
def tbl() -> Instrument:
    get_config('tbl')
    instrument = instrument_registry['tbl']
    instrument.load_factories()
    return instrument


def test_timepix3_downsampling_follows_the_file_numbering(tbl: Instrument) -> None:
    """The Timepix3 file counts from 1 with the first dimension fastest,
    ``detector_number[i, j] = 1 + i + 4096 * j``."""
    downsampling = tbl.get_downsampling('timepix3')
    assert downsampling.first_id == 1
    assert downsampling.grid.sizes == {
        'dim_0': specs.TIMEPIX3_IMAGE_RESOLUTION,
        'dim_1': specs.TIMEPIX3_IMAGE_RESOLUTION,
    }


def test_timepix3_ingest_puts_each_pixel_in_the_cell_covering_it(
    tbl: Instrument,
) -> None:
    """A pixel at ``(i, j)`` of the file's grid lands in cell
    ``(i // block, j // block)`` of the reduced image."""
    with snx.File(get_nexus_geometry_filename('tbl')) as f:
        declared = f['entry/instrument/timepix3/detector_number'][()]
    side = declared.sizes['dim_0']
    resolution = specs.TIMEPIX3_IMAGE_RESOLUTION
    block = side // resolution
    rng = np.random.default_rng(seed=1234)
    i, j = rng.integers(0, side, size=(2, 50_000))
    events = DetectorEvents(
        time_of_arrival=np.arange(i.size, dtype='int64'),
        unit='ns',
        pixel_id=declared.values[i, j].astype('int64'),
    )

    downsampling = tbl.get_downsampling('timepix3')
    accumulator = DownsamplePixelIds(
        GroupByPixel(ToNXevent_data(), downsampling.grid), downsampling
    )
    accumulator.add(Timestamp.from_ns(0), events)
    reduced = (
        accumulator.get()
        .bins.size()
        .fold('detector_number', sizes=downsampling.grid.sizes)
        .transpose(declared.dims)
    )

    expected = np.zeros((resolution, resolution), dtype='int64')
    np.add.at(expected, (i // block, j // block), 1)
    np.testing.assert_array_equal(reduced.values, expected)
