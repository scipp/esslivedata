# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
LOKI instrument factory implementations (heavy).

This module contains factory implementations with heavy dependencies.
Only imported by backend services.
"""

import sciline
import sciline.typing
from scippnexus import NXdetector

import ess.loki.live  # noqa: F401
from ess import loki
from ess.livedata.config import instrument_registry
from ess.livedata.handlers.detector_data_handler import (
    DetectorProjection,
    get_nexus_geometry_filename,
)
from ess.livedata.handlers.detector_view_specs import (
    DetectorViewParams,
    ROIHistogramParams,
)
from ess.livedata.handlers.monitor_data_handler import attach_monitor_workflow_factory
from ess.livedata.handlers.stream_processor_workflow import StreamProcessorWorkflow
from ess.reduce.nexus.types import NeXusData, NeXusDetectorName, SampleRun
from ess.sans import types as sans_types
from ess.sans.types import (
    Filename,
    Incident,
    IofQ,
    Numerator,
    ReducedQ,
    Transmission,
)

from . import specs
from .specs import SansWorkflowParams

# Get instrument from registry (already registered by specs.py)
instrument = instrument_registry['loki']

# Add detectors
for bank in range(9):
    instrument.add_detector(f'loki_detector_{bank}')

# Created once outside workflow wrappers since this configures some files from pooch
# where a checksum is needed, which takes significant time.
_base_workflow = loki.live._configured_Larmor_AgBeh_workflow()
_base_workflow[Filename[SampleRun]] = get_nexus_geometry_filename('loki')

# Create XY projection with specific resolution for each detector
_xy_projection = DetectorProjection(
    instrument=instrument,
    projection='xy_plane',
    pixel_noise='cylindrical',
    resolution={
        'loki_detector_0': {'y': 12, 'x': 12},
        # First window frame
        'loki_detector_1': {'y': 3, 'x': 9},
        'loki_detector_2': {'y': 9, 'x': 3},
        'loki_detector_3': {'y': 3, 'x': 9},
        'loki_detector_4': {'y': 9, 'x': 3},
        # Second window frame
        'loki_detector_5': {'y': 3, 'x': 9},
        'loki_detector_6': {'y': 9, 'x': 3},
        'loki_detector_7': {'y': 3, 'x': 9},
        'loki_detector_8': {'y': 9, 'x': 3},
    },
    resolution_scale=12,
)


# Attach detector view factory
@specs.xy_projection_handles['xy_plane']['view'].attach_factory()
def _xy_projection_view_factory(source_name: str, params: DetectorViewParams):
    """Factory for XY projection detector view."""
    return _xy_projection.make_view(source_name, params=params)


# Attach ROI histogram factory
@specs.xy_projection_handles['xy_plane']['roi'].attach_factory()
def _xy_projection_roi_factory(source_name: str, params: ROIHistogramParams):
    """Factory for XY projection ROI histogram."""
    return _xy_projection.make_roi(source_name, params=params)


# Helper functions for SANS workflows
def _transmission_from_current_run(
    data: sans_types.CleanMonitor[SampleRun, sans_types.MonitorType],
) -> sans_types.CleanMonitor[
    sans_types.TransmissionRun[SampleRun], sans_types.MonitorType
]:
    return data


def _dynamic_keys(source_name: str) -> dict[str, sciline.typing.Key]:
    return {
        source_name: NeXusData[NXdetector, SampleRun],
        'incident_monitor': NeXusData[Incident, SampleRun],
        'transmission_monitor': NeXusData[Transmission, SampleRun],
    }


_accumulators = (
    ReducedQ[SampleRun, Numerator],
    sans_types.CleanMonitor[SampleRun, Incident],
    sans_types.CleanMonitor[SampleRun, Transmission],
)


# Attach I(Q) workflow factory
@specs.i_of_q_handle.attach_factory()
def _i_of_q_factory(source_name: str) -> StreamProcessorWorkflow:
    """Factory for basic I(Q) workflow."""
    wf = _base_workflow.copy()
    wf[NeXusDetectorName] = source_name
    return StreamProcessorWorkflow(
        wf,
        dynamic_keys=_dynamic_keys(source_name),
        target_keys={'i_of_q': IofQ[SampleRun]},
        accumulators=_accumulators,
    )


# Attach I(Q) with params workflow factory
@specs.i_of_q_with_params_handle.attach_factory()
def _i_of_q_with_params_factory(
    source_name: str, params: SansWorkflowParams
) -> StreamProcessorWorkflow:
    """Factory for I(Q) workflow with configurable parameters."""
    wf = _base_workflow.copy()
    wf[NeXusDetectorName] = source_name

    wf[sans_types.QBins] = params.q_edges.get_edges()
    wf[sans_types.WavelengthBins] = params.wavelength_edges.get_edges()

    if not params.options.use_transmission_run:
        target_keys = {
            'i_of_q': IofQ[SampleRun],
            'transmission_fraction': sans_types.TransmissionFraction[SampleRun],
        }
        wf.insert(_transmission_from_current_run)
    else:
        # Transmission fraction is static, do not display
        target_keys = {'i_of_q': IofQ[SampleRun]}
    return StreamProcessorWorkflow(
        wf,
        dynamic_keys=_dynamic_keys(source_name),
        target_keys=target_keys,
        accumulators=_accumulators,
    )


# Attach monitor workflow factory
attach_monitor_workflow_factory(specs.monitor_workflow_handle)
