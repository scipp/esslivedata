# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ESTIA instrument factory implementations.
"""

from ess.livedata.config import Instrument

from . import specs


def setup_factories(instrument: Instrument) -> None:
    """Initialize ESTIA-specific factories and workflows.

    The multiblade detector view (with its spectrum output) is wired via
    ``add_logical_view`` in ``specs.py``. The generic ``cbm`` monitor workflow
    factory is attached here.
    """
    from ess.livedata.handlers.monitor_workflow import create_monitor_workflow
    from ess.livedata.handlers.monitor_workflow_specs import TOAOnlyMonitorDataParams

    @specs.monitor_handle.attach_factory()
    def _monitor_workflow_factory(source_name: str, params: TOAOnlyMonitorDataParams):
        return create_monitor_workflow(
            source_name=source_name,
            edges=params.get_active_edges(),
            range_filter=params.get_active_range(),
            coordinate_mode='toa',
        )
