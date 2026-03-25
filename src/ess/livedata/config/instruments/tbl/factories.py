# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
TBL workflow workflow factory implementations.
"""

from ess.livedata.config import Instrument

from . import specs
from .views import fold_image


def setup_factories(instrument: Instrument) -> None:
    """Initialize TBL-specific factories and workflows."""
    from ess.livedata.handlers.area_detector_view import AreaDetectorView
    from ess.livedata.handlers.monitor_workflow import create_monitor_workflow
    from ess.livedata.handlers.monitor_workflow_specs import TOAOnlyMonitorDataParams

    specs.orca_view_handle.attach_factory()(
        AreaDetectorView.view_factory(
            transform=fold_image, reduction_dim=['x_bin', 'y_bin']
        )
    )

    @specs.monitor_handle.attach_factory()
    def _monitor_workflow_factory(source_name: str, params: TOAOnlyMonitorDataParams):
        return create_monitor_workflow(
            source_name=source_name,
            edges=params.get_active_edges(),
            range_filter=params.get_active_range(),
            coordinate_mode='toa',
        )
