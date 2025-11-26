# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ODIN instrument factory implementations.
"""

import scipp as sc

from ess.livedata.config import Instrument

from . import specs


def _fold_image(da: sc.DataArray) -> sc.DataArray:
    """Fold detector image dimensions for downsampling to 512x512."""
    # 4096x4096 is the actual panel size, but ess.livedata might not be able to keep
    # up with that so we downsample to 512x512.
    # The geometry file has generic dim_0/dim_1 names, so we rename to x/y.
    da = da.rename_dims({'dim_0': 'x', 'dim_1': 'y'})
    da = da.fold(dim='x', sizes={'x': 512, 'x_bin': -1})
    da = da.fold(dim='y', sizes={'y': 512, 'y_bin': -1})
    return da


def setup_factories(instrument: Instrument) -> None:
    """Initialize ODIN-specific factories and workflows."""
    from ess.livedata.handlers.detector_data_handler import (
        DetectorLogicalDownsampler,
    )

    # Configure detector with custom group name
    instrument.configure_detector(
        'timepix3', detector_group_name='event_mode_detectors'
    )

    # Detector view using LogicalView for proper ROI support
    _panel_0_view = DetectorLogicalDownsampler(
        instrument=instrument,
        transform=_fold_image,
        reduction_dim=['x_bin', 'y_bin'],
    )

    specs.panel_0_view_handle.attach_factory()(_panel_0_view.make_view)
