# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the TBL detector views.

The transforms in ``tbl.views`` are coupled by dimension name: the view
transform names the spatial dims, the spectrum transform reduces them. These
tests run both through a real workflow so a rename on either side fails loudly.
"""

import numpy as np
import pytest
import scipp as sc
from ess.reduce.nexus.types import RawDetector, SampleRun

from ess.livedata.config.instrument import instrument_registry
from ess.livedata.config.instruments import get_config
from ess.livedata.config.instruments.tbl.specs import Timepix3DetectorViewParams
from ess.livedata.config.instruments.tbl.views import (
    MAX_IMAGE_SIZE,
    fold_timepix3_image,
    get_he3_detector_view,
    get_he3_spectrum,
    get_multiblade_spectrum,
    get_multiblade_view,
)
from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.workflows.detector_view.data_source import DetectorNumberSource
from ess.livedata.workflows.detector_view.factory import DetectorViewFactory
from ess.livedata.workflows.detector_view.types import LogicalViewConfig
from ess.livedata.workflows.detector_view_specs import (
    SpectrumViewSpec,
    make_detector_view_params,
)

# Shapes of the real TBL detectors, from geometry-tbl-2026-07-01.nxs.
MULTIBLADE_SIZES = {'blade': 14, 'wire': 32, 'strip': 64}
HE3_SIZES = {'dim_0': 4, 'dim_1': 100}

ROI_AUX_NAMES = {'roi_rectangle': 'roi_rectangle', 'roi_polygon': 'roi_polygon'}


@pytest.fixture(scope='module')
def tbl_instrument():
    get_config('tbl')  # Registers the specs.
    return instrument_registry['tbl']


def _make_events(detector_number: sc.Variable, n_per_pixel: int) -> sc.DataArray:
    """Bin ``n_per_pixel`` events per pixel, in NXevent_data layout."""
    rng = np.random.default_rng(42)
    ids = np.sort(detector_number.values.ravel())
    total = len(ids) * n_per_pixel
    events = sc.DataArray(
        data=sc.ones(dims=['event'], shape=[total]),
        coords={
            'event_time_offset': sc.array(
                dims=['event'], values=rng.uniform(0, 71e6, total), unit='ns'
            ),
            'event_id': sc.array(
                dims=['event'], values=np.repeat(ids, n_per_pixel), unit=None
            ),
        },
    )
    begin = sc.arange('detector_number', 0, total, n_per_pixel)
    begin.unit = None
    return sc.DataArray(
        data=sc.bins(
            begin=begin,
            end=begin + sc.scalar(n_per_pixel, unit=None),
            dim='event',
            data=events,
        ),
        coords={'detector_number': sc.array(dims=['detector_number'], values=ids)},
    )


def _run_view(
    detector_number: sc.Variable,
    view_transform,
    spectrum_spec: SpectrumViewSpec | None = None,
    *,
    reduction_dim: list[str] | None = None,
    n_per_pixel: int = 2,
    roi_support: bool = True,
) -> dict[str, sc.DataArray]:
    """Build and run one update of a logical view, returning its outputs."""
    factory = DetectorViewFactory(
        data_source=DetectorNumberSource(detector_number),
        view_config=LogicalViewConfig(
            transform=view_transform,
            reduction_dim=reduction_dim,
            spectrum_view=spectrum_spec,
            roi_support=roi_support,
        ),
    )
    workflow = factory.make_workflow(
        'detector',
        params=make_detector_view_params(spectrum_view=spectrum_spec)(),
        aux_source_names=ROI_AUX_NAMES,
    )
    workflow.build()
    workflow.accumulate(
        {
            'detector': RawDetector[SampleRun](
                _make_events(detector_number, n_per_pixel)
            )
        },
        start_time=Timestamp.from_ns(1000),
        end_time=Timestamp.from_ns(2000),
    )
    return workflow.finalize()


@pytest.mark.parametrize(
    ('view_name', 'expected_spatial_dims'),
    [
        ('multiblade_detector_view', ('blade', 'wire')),
        ('he3_detector_view', ('tube', 'pixel')),
    ],
)
def test_spectrum_view_is_a_registered_output(
    tbl_instrument, view_name, expected_spatial_dims
):
    spec = tbl_instrument.workflow_factory[
        WorkflowId(instrument='tbl', name=view_name, version=1)
    ]
    template = spec.outputs().spectrum_view
    # The spectral dim is a placeholder until the workflow runs; the spatial dims
    # come straight from the registered ``output_dims``.
    assert template.dims[:-1] == expected_spatial_dims


@pytest.mark.parametrize('view_name', ['tbl_detector_timepix3', 'ngem_detector_view'])
def test_views_without_spectrum_support_have_no_spectrum_output(
    tbl_instrument, view_name
):
    spec = tbl_instrument.workflow_factory[
        WorkflowId(instrument='tbl', name=view_name, version=1)
    ]
    assert 'spectrum_view' not in spec.outputs.model_fields


def test_multiblade_view_has_no_roi_outputs(tbl_instrument):
    """ROI geometries are 2D; the multiblade screen is (blade, wire, strip)."""
    spec = tbl_instrument.workflow_factory[
        WorkflowId(instrument='tbl', name='multiblade_detector_view', version=1)
    ]
    assert 'roi_spectra_cumulative' not in spec.outputs.model_fields
    assert 'roi_rectangle' not in spec.outputs.model_fields


def test_he3_view_keeps_roi_outputs(tbl_instrument):
    spec = tbl_instrument.workflow_factory[
        WorkflowId(instrument='tbl', name='he3_detector_view', version=1)
    ]
    assert 'roi_spectra_cumulative' in spec.outputs.model_fields
    assert 'roi_rectangle' in spec.outputs.model_fields


def test_multiblade_spectrum_sums_strips():
    n_pixels = int(np.prod(list(MULTIBLADE_SIZES.values())))
    detector_number = sc.arange('detector_number', 1, n_pixels + 1, unit=None)
    result = _run_view(
        detector_number,
        get_multiblade_view,
        SpectrumViewSpec(
            transform=get_multiblade_spectrum, output_dims=['blade', 'wire']
        ),
        roi_support=False,
    )

    spectrum = result['spectrum_view']
    assert spectrum.dims == ('blade', 'wire', 'time_of_arrival')
    assert spectrum.sizes['blade'] == MULTIBLADE_SIZES['blade']
    assert spectrum.sizes['wire'] == MULTIBLADE_SIZES['wire']
    # Summing strips redistributes but does not lose counts.
    assert sc.isclose(spectrum.sum().data, result['cumulative'].sum().data).value


def test_he3_spectrum_keeps_every_pixel():
    n_pixels = HE3_SIZES['dim_0'] * HE3_SIZES['dim_1']
    detector_number = sc.arange('dim_0', 1, n_pixels + 1, unit=None).fold(
        'dim_0', sizes=HE3_SIZES
    )
    result = _run_view(
        detector_number,
        get_he3_detector_view,
        SpectrumViewSpec(transform=get_he3_spectrum, output_dims=['tube', 'pixel']),
    )

    spectrum = result['spectrum_view']
    assert spectrum.dims == ('tube', 'pixel', 'time_of_arrival')
    assert spectrum.sizes['tube'] == HE3_SIZES['dim_0']
    assert spectrum.sizes['pixel'] == HE3_SIZES['dim_1']
    assert sc.isclose(spectrum.sum().data, result['cumulative'].sum().data).value


def test_he3_spectrum_does_not_alias_the_accumulator_buffer():
    """The published spectrum must survive the next update mutating the buffer."""
    histogram = sc.DataArray(
        sc.ones(dims=['tube', 'pixel', 'time_of_arrival'], shape=[4, 100, 8])
    )
    spectrum = get_he3_spectrum(histogram)
    histogram += histogram

    assert sc.identical(spectrum.data, sc.ones_like(spectrum.data))


def _timepix3_detector_number(grid: int) -> sc.Variable:
    """Full readout grid, enumerated the way the TBL geometry file does."""
    return sc.arange('dim_0', grid * grid, unit=None).fold(
        'dim_0', sizes={'dim_0': grid, 'dim_1': grid}
    )


def test_timepix3_resolution_defaults_to_the_full_grid():
    assert Timepix3DetectorViewParams().resolution.value == 4096


def test_fold_timepix3_image_enumerates_pixels_with_x_as_the_slow_axis():
    grid = 8
    resolution = 4
    da = sc.DataArray(_timepix3_detector_number(grid))

    folded = fold_timepix3_image(da, resolution)

    assert folded.sizes == {'x': 4, 'x_bin': 1, 'y': 4, 'y_bin': 1}
    expected = sc.arange('x', resolution * resolution, unit=None).fold(
        'x', sizes={'x': resolution, 'x_bin': 1, 'y': resolution, 'y_bin': 1}
    )
    assert sc.identical(folded.data, expected)


def test_fold_timepix3_image_downsamples_resolutions_above_the_image_limit():
    resolution = 2 * MAX_IMAGE_SIZE
    da = sc.DataArray(
        sc.ones(dims=['dim_0', 'dim_1'], shape=[resolution, resolution], unit='counts')
    )

    folded = fold_timepix3_image(da, resolution)

    assert folded.sizes == {
        'x': MAX_IMAGE_SIZE,
        'x_bin': 2,
        'y': MAX_IMAGE_SIZE,
        'y_bin': 2,
    }


def test_timepix3_view_drops_pixels_outside_the_selected_resolution():
    """A coarser mode reads out the leading pixels of the declared grid only."""
    grid = 64
    resolution = 16
    n_per_pixel = 2

    result = _run_view(
        _timepix3_detector_number(grid),
        lambda da, _source_name: fold_timepix3_image(da, resolution),
        reduction_dim=['x_bin', 'y_bin'],
        n_per_pixel=n_per_pixel,
    )

    image = result['cumulative']
    assert image.sizes == {'x': resolution, 'y': resolution}
    # Every pixel of the grid received events, but only those the mode reads out
    # are part of the image.
    assert image.sum().value == resolution * resolution * n_per_pixel
    assert sc.identical(
        image.data,
        sc.full(sizes=image.sizes, value=float(n_per_pixel), unit=image.unit),
    )


def test_timepix3_view_at_full_resolution_keeps_every_pixel():
    grid = 64
    n_per_pixel = 2

    result = _run_view(
        _timepix3_detector_number(grid),
        lambda da, _source_name: fold_timepix3_image(da, grid),
        reduction_dim=['x_bin', 'y_bin'],
        n_per_pixel=n_per_pixel,
    )

    image = result['cumulative']
    assert image.sizes == {'x': grid, 'y': grid}
    assert image.sum().value == grid * grid * n_per_pixel
