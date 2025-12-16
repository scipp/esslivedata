# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ESTIA instrument factory implementations.
"""

from ess.estia.beamline import DETECTOR_BANK_SIZES
from ess.livedata.config import Instrument

from . import specs


def setup_factories(instrument: Instrument) -> None:
    """Initialize ESTIA-specific factories and workflows."""
    from ess.livedata.handlers.detector_data_handler import DetectorLogicalView

    _multiblade_view = DetectorLogicalView(
        instrument=instrument,
        transform=lambda da: da.fold(
            dim='detector_number', sizes=DETECTOR_BANK_SIZES['multiblade_detector']
        ),
    )
    specs.multiblade_view_handle.attach_factory()(_multiblade_view.make_view)
