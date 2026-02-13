# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ODIN instrument factory implementations.
"""

from ess.livedata.config import Instrument

from . import specs


def setup_factories(instrument: Instrument) -> None:
    """Initialize ODIN-specific factories and workflows."""
    # Configure detector with custom group name
    instrument.configure_detector(
        'timepix3', detector_group_name='event_mode_detectors'
    )

    # Monitor workflow factory (TOA-only)
    from ess.livedata.handlers.monitor_workflow import create_monitor_workflow
    from ess.livedata.handlers.monitor_workflow_specs import TOAOnlyMonitorDataParams

    @specs.monitor_handle.attach_factory()
    def _monitor_workflow_factory(source_name: str, params: TOAOnlyMonitorDataParams):
        """Factory for ODIN monitor workflow (TOA-only)."""
        return create_monitor_workflow(
            source_name=source_name,
            edges=params.get_active_edges(),
            range_filter=params.get_active_range(),
            coordinate_mode='toa',
        )
