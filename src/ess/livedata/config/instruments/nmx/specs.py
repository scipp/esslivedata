# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
NMX instrument spec registration.
"""

from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.handlers.detector_view_specs import (
    DetectorViewOutputs,
    DetectorViewParams,
)
from ess.livedata.handlers.monitor_workflow_specs import (
    TOAOnlyMonitorDataParams,
    register_monitor_workflow_specs,
)

# Detector panel names
detector_names = ['detector_panel_0', 'detector_panel_1', 'detector_panel_2']

# Create instrument with detectors and monitors
instrument = Instrument(
    name='nmx',
    detector_names=detector_names,
    monitors=['monitor1', 'monitor2'],
)

# Register instrument
instrument_registry.register(instrument)

# Register monitor workflow spec (TOA-only, no TOF lookup tables)
monitor_handle = register_monitor_workflow_specs(
    instrument, ['monitor1', 'monitor2'], params=TOAOnlyMonitorDataParams
)

# Register detector view spec for the panel_xy view
panel_xy_view_handle = instrument.register_spec(
    namespace='detector_data',
    name='panel_xy',
    version=1,
    title='Detector counts',
    description='Detector counts per pixel.',
    source_names=detector_names,
    params=DetectorViewParams,
    outputs=DetectorViewOutputs,
)
