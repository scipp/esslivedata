# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
TBL instrument factory implementations.
"""

from ess.livedata.config import Instrument

from .._detectors import orca_fold, timepix3_fold
from . import specs


def setup_factories(instrument: Instrument) -> None:
    """Initialize TBL-specific factories and workflows."""
    from ess.livedata.handlers.area_detector_view import AreaDetectorView
    from ess.livedata.handlers.detector_data_handler import DetectorLogicalView

    # Timepix3 detector view (ev44 event detector)
    _timepix3_view = DetectorLogicalView(
        instrument=instrument,
        transform=timepix3_fold,
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
    _he3_detector_view = DetectorLogicalView(
        instrument=instrument,
        transform=lambda da: da.rename_dims(dim_0='tube', dim_1='pixel'),
    )
    specs.he3_detector_handle.attach_factory()(_he3_detector_view.make_view)

    _ngem_view = DetectorLogicalView(
        instrument=instrument, transform=lambda da: da, reduction_dim='dim_0'
    )
    specs.ngem_detector_handle.attach_factory()(_ngem_view.make_view)

    # Orca area detector view (ad00 image detector)
    specs.orca_view_handle.attach_factory()(
        AreaDetectorView.view_factory(
            transform=orca_fold, reduction_dim=['x_bin', 'y_bin']
        )
    )
