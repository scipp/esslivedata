# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
NMX instrument factory implementations.
"""

import scipp as sc

from ess.livedata.config import Instrument

from . import specs


def setup_factories(instrument: Instrument) -> None:
    """Initialize NMX-specific factories and configure detectors."""
    # Lazy imports
    from ess.livedata.handlers.detector_view import (
        DetectorViewFactory,
        InstrumentDetectorSource,
        LogicalViewConfig,
    )

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

    # Create detector view using Sciline-based factory (identity transform)
    _nmx_panels_view = DetectorViewFactory(
        data_source=InstrumentDetectorSource(instrument),
        view_config=LogicalViewConfig(),  # Identity transform
    )

    specs.panel_xy_view_handle.attach_factory()(_nmx_panels_view.make_workflow)

    # Monitor workflow factory (TOA-only)
    from ess.livedata.handlers.monitor_workflow import create_monitor_workflow
    from ess.livedata.handlers.monitor_workflow_specs import TOAOnlyMonitorDataParams

    @specs.monitor_handle.attach_factory()
    def _monitor_workflow_factory(source_name: str, params: TOAOnlyMonitorDataParams):
        """Factory for NMX monitor workflow (TOA-only)."""
        return create_monitor_workflow(
            source_name=source_name,
            edges=params.get_active_edges(),
            range_filter=params.get_active_range(),
            coordinate_mode='toa',
        )
