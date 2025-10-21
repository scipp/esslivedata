# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ODIN instrument spec registration (lightweight).

This module registers workflow specs WITHOUT heavy dependencies.
Frontend loads this module to access workflow specs.
Backend services must also import .streams for stream mapping configuration.
"""

from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.handlers.monitor_workflow_specs import register_monitor_workflow_specs

# Create instrument
instrument = Instrument(name='odin')

# Register lightweight workflows (no heavy dependencies)
monitor_workflow_handle = register_monitor_workflow_specs(
    instrument=instrument, source_names=['monitor1', 'monitor2']
)

# Register instrument
instrument_registry.register(instrument)

# Detector view spec registration (currently disabled)
# WARNING: Disabled until fixed
# from ess.livedata.handlers.detector_view_specs import DetectorViewParams
#
# panel_0_view_handle = instrument.register_spec(
#     namespace='detector_data',
#     name='odin_detector_xy',
#     version=1,
#     title='Timepix3 XY Detector Counts',
#     description='2D view of the Timepix3 detector counts',
#     source_names=['timepix3'],
#     params=DetectorViewParams,
# )
