# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
NMX instrument spec registration.
"""

from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.handlers.detector_view_specs import DetectorViewParams
from ess.livedata.handlers.monitor_workflow_specs import register_monitor_workflow_specs

# Create instrument
instrument = Instrument(name='nmx')

monitor_workflow_handle = register_monitor_workflow_specs(
    instrument=instrument, source_names=['monitor1', 'monitor2']
)

# Register instrument
instrument_registry.register(instrument)

# Register detector view spec for the panel_xy view (no factory yet)
# Note: The factory will be attached in factories.py
# Source names are the detector panel names (created in factories.py)
panel_xy_view_handle = instrument.register_spec(
    namespace='detector_data',
    name='panel_xy',
    version=1,
    title='Detector counts',
    description='Detector counts per pixel.',
    source_names=['detector_panel_0', 'detector_panel_1', 'detector_panel_2'],
    params=DetectorViewParams,
)
