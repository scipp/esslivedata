# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
TBL instrument factory implementations.
"""

import scipp as sc

from ess.livedata.config import Instrument

from . import specs


def _fold_image(da: sc.DataArray) -> sc.DataArray:
    """Fold detector image dimensions for downsampling to 512x512."""
    # 4096x4096 is the actual panel size, but ess.livedata might not be able to keep
    # up with that so we downsample to 512x512.
    da = da.rename_dims({'x_pixel_offset': 'x', 'y_pixel_offset': 'y'})
    da = da.fold(dim='x', sizes={'x': 512, 'x_bin': -1})
    da = da.fold(dim='y', sizes={'y': 512, 'y_bin': -1})
    return da


def _fold_area_detector(da: sc.DataArray) -> sc.DataArray:
    """Fold detector image dimensions for downsampling to 512x512."""
    # 2048x2048 is the actual panel size, but ess.livedata might not be able to keep
    # up with that so we downsample to 512x512.
    da = da.rename_dims({'dim_0': 'x', 'dim_1': 'y'})
    da = da.fold(dim='x', sizes={'x': 512, 'x_bin': -1})
    da = da.fold(dim='y', sizes={'y': 512, 'y_bin': -1})
    return da


def setup_factories(instrument: Instrument) -> None:
    """Initialize TBL-specific factories and workflows."""
    from ess.livedata.handlers.area_detector_view import AreaDetectorView
    from ess.livedata.handlers.detector_data_handler import DetectorLogicalView

    # Timepix3 detector view (ev44 event detector)
    _timepix3_view = DetectorLogicalView(
        instrument=instrument,
        transform=_fold_image,
        reduction_dim=['x_bin', 'y_bin'],
    )
    specs.timepix3_view_handle.attach_factory()(_timepix3_view.make_view)

    _multiblade_view = DetectorLogicalView(
        instrument=instrument,
        transform=lambda da: da.fold(
            dim='detector_number', sizes={'blade': 14, 'wire': -1, 'strip': 64}
        ),
    )
    specs.multiblade_view_handle.attach_factory()(_multiblade_view.make_view)

    # Orca area detector view (ad00 image detector)
    specs.orca_view_handle.attach_factory()(
        AreaDetectorView.view_factory(
            transform=_fold_area_detector, reduction_dim=['x_bin', 'y_bin']
        )
    )
