# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
DREAM instrument factory implementations (heavy).

This module contains factory implementations with heavy dependencies.
Only imported by backend services.
"""

from typing import NewType

import scipp as sc
from scippnexus import NXdetector

import ess.powder.types  # noqa: F401
from ess import dream, powder
from ess.dream import DreamPowderWorkflow
from ess.livedata.config import instrument_registry
from ess.livedata.handlers.detector_data_handler import (
    DetectorLogicalView,
    DetectorProjection,
    LogicalViewConfig,
    get_nexus_geometry_filename,
)
from ess.livedata.handlers.detector_view_specs import DetectorViewParams
from ess.livedata.handlers.stream_processor_workflow import StreamProcessorWorkflow
from ess.reduce.nexus.types import (
    DetectorData,
    Filename,
    NeXusData,
    NeXusName,
    RunType,
    SampleRun,
    VanadiumRun,
)

from .specs import (
    PowderWorkflowParams,
    cylinder_roi_handle,
    cylinder_view_handle,
    mantle_front_layer_handle,
    mantle_wire_view_handle,
    powder_reduction_handle,
    powder_reduction_with_vanadium_handle,
    xy_roi_handle,
    xy_view_handle,
)

# Get instrument from registry (already registered by specs.py)
instrument = instrument_registry['dream']

# Add detectors
instrument.add_detector('mantle_detector')
instrument.add_detector('endcap_backward_detector')
instrument.add_detector('endcap_forward_detector')
instrument.add_detector('high_resolution_detector')
instrument.add_detector('sans_detector')

# Create detector projections
_cylinder_projection = DetectorProjection(
    instrument=instrument,
    projection='cylinder_mantle_z',
    pixel_noise=sc.scalar(4.0, unit='mm'),
    resolution={'mantle_detector': {'arc_length': 10, 'z': 40}},
    resolution_scale=8,
)

_xy_projection = DetectorProjection(
    instrument=instrument,
    projection='xy_plane',
    pixel_noise=sc.scalar(4.0, unit='mm'),
    resolution={
        'endcap_backward_detector': {'y': 30, 'x': 20},
        'endcap_forward_detector': {'y': 20, 'x': 20},
        'high_resolution_detector': {'y': 20, 'x': 20},
    },
    resolution_scale=8,
)

# Attach detector view factories using handles from specs
_cylinder_projection.attach_to_handles(
    view_handle=cylinder_view_handle, roi_handle=cylinder_roi_handle
)

_xy_projection.attach_to_handles(view_handle=xy_view_handle, roi_handle=xy_roi_handle)


# Bank sizes for mantle detector logical views
_bank_sizes = {
    'mantle_detector': {
        'wire': 32,
        'module': 5,
        'segment': 6,
        'strip': 256,
        'counter': 2,
    },
}


def _get_mantle_front_layer(da: sc.DataArray) -> sc.DataArray:
    """Transform function to extract mantle front layer."""
    return (
        da.fold(dim=da.dim, sizes=_bank_sizes['mantle_detector'])
        .transpose(('wire', 'module', 'segment', 'counter', 'strip'))['wire', 0]
        .flatten(('module', 'segment', 'counter'), to='mod/seg/cntr')
    )


def _get_wire_view(da: sc.DataArray) -> sc.DataArray:
    """Transform function to extract wire view."""
    return (
        da.fold(dim=da.dim, sizes=_bank_sizes['mantle_detector'])
        .sum('strip')
        .flatten(('module', 'segment', 'counter'), to='mod/seg/cntr')
        # Transpose so that wire is the "x" dimension for more natural plotting.
        .transpose()
    )


# Create logical views for mantle detector
_mantle_front_layer_config = LogicalViewConfig(
    name='mantle_front_layer',
    title='Mantle front layer',
    description='All voxels of the front layer of the mantle detector.',
    source_names=['mantle_detector'],
    transform=_get_mantle_front_layer,
)

_mantle_front_layer_view = DetectorLogicalView(
    instrument=instrument, config=_mantle_front_layer_config
)

_mantle_wire_view_config = LogicalViewConfig(
    name='mantle_wire_view',
    title='Mantle wire view',
    description='Sum over strips to show counts per wire in the mantle detector.',
    source_names=['mantle_detector'],
    transform=_get_wire_view,
)

_mantle_wire_view = DetectorLogicalView(
    instrument=instrument, config=_mantle_wire_view_config
)


# Attach logical view factories
# Note: Logical views only have view handles, no ROI handles
@mantle_front_layer_handle.attach_factory()
def _mantle_front_layer_factory(source_name: str, params: DetectorViewParams):
    """Factory for mantle front layer logical view."""
    return _mantle_front_layer_view.make_view(source_name, params=params)


@mantle_wire_view_handle.attach_factory()
def _mantle_wire_view_factory(source_name: str, params: DetectorViewParams):
    """Factory for mantle wire view logical view."""
    return _mantle_wire_view.make_view(source_name, params=params)


# Powder reduction workflow setup
_reduction_workflow = DreamPowderWorkflow(
    run_norm=powder.RunNormalization.proton_charge
)

_source_names = [
    'mantle_detector',
    'endcap_backward_detector',
    'endcap_forward_detector',
    'high_resolution_detector',
]

TotalCounts = NewType('TotalCounts', sc.DataArray)


def _total_counts(data: DetectorData[SampleRun]) -> TotalCounts:
    """Dummy provider for some plottable result of total counts."""
    return TotalCounts(
        data.nanhist(
            event_time_offset=sc.linspace(
                'event_time_offset', 0, 71_000_000, num=1000, unit='ns'
            ),
            dim=data.dims,
        )
    )


def _fake_proton_charge(
    data: powder.types.ReducedCountsDspacing[RunType],
) -> powder.types.AccumulatedProtonCharge[RunType]:
    """
    Fake approximate proton charge for consistent normalization during streaming.

    This is not meant for production, but as a workaround until monitor normalization is
    fixed and/or we have setup a proton-charge stream.
    """
    fake_charge = sc.values(data.data).sum()
    fake_charge.unit = 'counts/µAh'
    return powder.types.AccumulatedProtonCharge[RunType](fake_charge)


_reduction_workflow.insert(_total_counts)
_reduction_workflow.insert(_fake_proton_charge)
_reduction_workflow[powder.types.CalibrationData] = None
_reduction_workflow = powder.with_pixel_mask_filenames(_reduction_workflow, [])
_reduction_workflow[powder.types.UncertaintyBroadcastMode] = (
    powder.types.UncertaintyBroadcastMode.drop
)
_reduction_workflow[powder.types.KeepEvents[SampleRun]] = powder.types.KeepEvents[
    SampleRun
](False)

# dream-no-shape is a much smaller file without pixel_shape, which is not needed
# for data reduction.
_reduction_workflow[Filename[SampleRun]] = get_nexus_geometry_filename('dream-no-shape')


@powder_reduction_handle.attach_factory()
def _powder_workflow_factory(source_name: str, params: PowderWorkflowParams):
    """Factory for DREAM powder reduction workflow."""
    wf = _reduction_workflow.copy()
    wf[NeXusName[NXdetector]] = source_name
    # Convert string to enum - this is where we handle the heavy dependency
    wf[dream.InstrumentConfiguration] = getattr(
        dream.InstrumentConfiguration, params.instrument_configuration.value
    )
    wmin = params.wavelength_range.get_start()
    wmax = params.wavelength_range.get_stop()
    wf[powder.types.WavelengthMask] = lambda w: (w < wmin) | (w > wmax)
    wf[powder.types.TwoThetaBins] = params.two_theta_edges.get_edges()
    wf[powder.types.DspacingBins] = params.dspacing_edges.get_edges()
    return StreamProcessorWorkflow(
        wf,
        dynamic_keys={
            source_name: NeXusData[NXdetector, SampleRun],
            'cave_monitor': NeXusData[powder.types.CaveMonitor, SampleRun],
        },
        target_keys={
            'focussed_data_dspacing': powder.types.FocussedDataDspacing[SampleRun],
            'focussed_data_dspacing_two_theta': (
                powder.types.FocussedDataDspacingTwoTheta[SampleRun]
            ),
        },
        accumulators=(
            powder.types.ReducedCountsDspacing[SampleRun],
            powder.types.WavelengthMonitor[SampleRun, powder.types.CaveMonitor],
        ),
    )


@powder_reduction_with_vanadium_handle.attach_factory()
def _powder_workflow_with_vanadium_factory(
    source_name: str, params: PowderWorkflowParams
):
    """Factory for DREAM powder reduction workflow with vanadium normalization."""
    wf = _reduction_workflow.copy()
    wf[NeXusName[NXdetector]] = source_name
    wf[Filename[VanadiumRun]] = '268227_00024779_Vana_inc_BC_offset_240_deg_wlgth.hdf'
    # Convert string to enum - this is where we handle the heavy dependency
    wf[dream.InstrumentConfiguration] = getattr(
        dream.InstrumentConfiguration, params.instrument_configuration.value
    )
    wmin = params.wavelength_range.get_start()
    wmax = params.wavelength_range.get_stop()
    wf[powder.types.WavelengthMask] = lambda w: (w < wmin) | (w > wmax)
    wf[powder.types.TwoThetaBins] = params.two_theta_edges.get_edges()
    wf[powder.types.DspacingBins] = params.dspacing_edges.get_edges()
    return StreamProcessorWorkflow(
        wf,
        dynamic_keys={
            source_name: NeXusData[NXdetector, SampleRun],
            'cave_monitor': NeXusData[powder.types.CaveMonitor, SampleRun],
        },
        target_keys={
            'focussed_data_dspacing': powder.types.FocussedDataDspacing[SampleRun],
            'focussed_data_dspacing_two_theta': (
                powder.types.FocussedDataDspacingTwoTheta[SampleRun]
            ),
            'i_of_dspacing': powder.types.IofDspacing[SampleRun],
            'i_of_dspacing_two_theta': powder.types.IofDspacingTwoTheta[SampleRun],
        },
        accumulators=(
            powder.types.ReducedCountsDspacing[SampleRun],
            powder.types.WavelengthMonitor[SampleRun, powder.types.CaveMonitor],
        ),
    )
