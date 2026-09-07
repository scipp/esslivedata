# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Tests for NMX's reduced-resolution panel ingest."""

import numpy as np
import pytest
import scipp as sc

from ess.livedata.config.instrument import Instrument, instrument_registry
from ess.livedata.config.instruments import get_config
from ess.livedata.config.instruments.nmx.specs import (
    PANEL_IMAGE_RESOLUTION,
    PANEL_RESOLUTION,
)
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.preprocessors.downsample_pixel_ids import DownsamplePixelIds
from ess.livedata.preprocessors.group_by_pixel import GroupByPixel
from ess.livedata.preprocessors.to_nxevent_data import DetectorEvents, ToNXevent_data


@pytest.fixture(scope='module')
def instrument() -> Instrument:
    get_config('nmx')  # Register
    nmx = instrument_registry['nmx']
    nmx.load_factories()  # Panel detector_number is computed in setup_factories
    return nmx


@pytest.mark.parametrize('panel', range(3))
def test_panel_is_ingested_on_a_named_grid_from_its_own_id_base(
    instrument: Instrument, panel: int
) -> None:
    downsampling = instrument.get_downsampling(f'detector_panel_{panel}')
    # Panels are enumerated 1-based, one contiguous block each. Reading the
    # base off the wrong panel would not merely shift the image, it would put
    # every id past the panel and drop it.
    assert downsampling.first_id == panel * PANEL_RESOLUTION**2 + 1
    # The coarse grid keeps the axis names the panel declares, so the view
    # transform has nothing left to rename.
    assert downsampling.grid.sizes == {
        'x': PANEL_IMAGE_RESOLUTION,
        'y': PANEL_IMAGE_RESOLUTION,
    }
    # The readout is fixed, so the stride is stated rather than inferred, and
    # agrees with the grid the panel declares.
    assert downsampling.source_resolution == PANEL_RESOLUTION


def test_ingest_yields_the_block_sum_of_the_full_resolution_image(
    instrument: Instrument,
) -> None:
    """The reduced image is what folding the full-resolution one would give.

    Exercises the real configuration rather than a synthetic grid, on the
    panel whose ids start well above zero.
    """
    downsampling = instrument.get_downsampling('detector_panel_1')
    declared = sc.arange(
        'detector_number',
        downsampling.first_id,
        downsampling.first_id + PANEL_RESOLUTION**2,
        unit=None,
    )
    rng = np.random.default_rng(seed=1234)
    pixel_id = rng.choice(declared.values, size=50_000).astype('int64')
    events = DetectorEvents(
        time_of_arrival=np.arange(pixel_id.size, dtype='int64'),
        unit='ns',
        pixel_id=pixel_id,
    )

    accumulator = DownsamplePixelIds(
        GroupByPixel(ToNXevent_data(), downsampling.grid.flatten(to='detector_number')),
        downsampling,
    )
    accumulator.add(Timestamp.from_ns(0), events)
    reduced = accumulator.get().bins.size()

    reference = GroupByPixel(ToNXevent_data(), declared)
    reference.add(Timestamp.from_ns(0), events)
    block = PANEL_RESOLUTION // PANEL_IMAGE_RESOLUTION
    expected = (
        reference.get()
        .bins.size()
        .fold('detector_number', sizes={'x': PANEL_RESOLUTION, 'y': PANEL_RESOLUTION})
        .fold('x', sizes={'x': PANEL_IMAGE_RESOLUTION, 'x_bin': block})
        .fold('y', sizes={'y': PANEL_IMAGE_RESOLUTION, 'y_bin': block})
        .sum(['x_bin', 'y_bin'])
    )

    assert sc.identical(
        reduced.fold(
            'detector_number',
            sizes={'x': PANEL_IMAGE_RESOLUTION, 'y': PANEL_IMAGE_RESOLUTION},
        ).data,
        expected.data,
    )
    assert reduced.data.sum().value == pixel_id.size
