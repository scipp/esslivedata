# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
NMX instrument factory implementations.
"""

from ess.livedata.config import Instrument

from . import specs


def setup_factories(instrument: Instrument) -> None:
    """Initialize NMX-specific factories and configure detectors."""
    # Lazy imports
    import scipp as sc

    from ess.livedata.handlers.detector_data_handler import (
        DetectorLogicalView,
        LogicalViewConfig,
    )
    from ess.livedata.handlers.detector_view_specs import DetectorViewParams

    # Configure detectors with computed arrays
    # TODO Unclear if this is transposed or not. Wait for updated files.
    dim = 'detector_number'
    sizes = {'x': 1280, 'y': 1280}
    for panel in range(3):
        instrument.configure_detector(
            f'detector_panel_{panel}',
            detector_number=sc.arange(
                'detector_number',
                panel * 1280**2 + 1,
                (panel + 1) * 1280**2 + 1,
                unit=None,
            ).fold(dim=dim, sizes=sizes),
        )

    # Create detector view configuration
    _nmx_panels_config = LogicalViewConfig(
        name='panel_xy',
        title='Detector counts',
        description='Detector counts per pixel.',
        source_names=instrument.detector_names,
    )
    _nmx_panels_view = DetectorLogicalView(
        instrument=instrument, config=_nmx_panels_config
    )

    @specs.panel_xy_view_handle.attach_factory()
    def _panel_xy_view_factory(source_name: str, params: DetectorViewParams):
        """Factory for panel_xy detector view."""
        return _nmx_panels_view.make_view(source_name, params=params)
