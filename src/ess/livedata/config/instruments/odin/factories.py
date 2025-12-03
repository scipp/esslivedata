# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ODIN instrument factory implementations.
"""

from ess.livedata.config import Instrument

from .._detectors import timepix3_fold
from . import specs


def setup_factories(instrument: Instrument) -> None:
    """Initialize ODIN-specific factories and workflows."""
    from ess.livedata.handlers.detector_data_handler import DetectorLogicalView

    # Configure detector with custom group name
    instrument.configure_detector(
        'timepix3', detector_group_name='event_mode_detectors'
    )

    # Detector view with downsampling and ROI support
    _panel_0_view = DetectorLogicalView(
        instrument=instrument,
        transform=timepix3_fold,
        reduction_dim=['x_bin', 'y_bin'],
    )

    specs.panel_0_view_handle.attach_factory()(_panel_0_view.make_view)
