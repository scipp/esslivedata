# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
LOKI instrument spec registration.
"""

from enum import StrEnum

import pydantic
import scipp as sc

from ess.livedata import parameter_models
from ess.livedata.config import Instrument, SourceMetadata, instrument_registry
from ess.livedata.config.workflow_spec import (
    MONITORS,
    AuxInput,
    AuxSources,
    WorkflowOutputsBase,
)
from ess.livedata.handlers.detector_view.types import TransformValueStream
from ess.livedata.handlers.detector_view_specs import (
    DetectorROIAuxSources,
    register_detector_view_spec,
)
from ess.livedata.handlers.monitor_workflow_specs import (
    MonitorDataParams,
    register_monitor_workflow_specs,
)

from .views import get_tube_view

#: Per-source bindings of NeXus transformation entries to live f144 streams.
#: Single source of truth shared between the spec (for routing via
#: ``DetectorROIAuxSources``) and the factory (for graph wiring via
#: ``DetectorViewFactory(dynamic_transforms=...)``). Only the rear bank has
#: a live carriage readback; other banks have no dynamic geometry.
LOKI_DYNAMIC_TRANSFORMS: dict[str, TransformValueStream] = {
    'loki_detector_0': TransformValueStream(
        transform_name='/entry/instrument/detector_carriage/value',
        aux_stream='detector_carriage',
    ),
}


class TransmissionMode(StrEnum):
    """Transmission correction mode for SANS reduction."""

    constant = 'constant'
    current_run = 'current_run'


class TransmissionOptions(pydantic.BaseModel):
    """Transmission correction configuration.

    The standard SANS transmission calculation normalizes sample monitors
    against an empty-beam measurement. In live mode only simplified modes
    are available that do not require a separate empty-beam run.
    """

    mode: TransmissionMode = pydantic.Field(
        title='Mode',
        description=(
            '"constant": no correction (transmission fraction = 1).'
            ' "current_run": estimate transmission from the ratio of'
            ' transmission to incident monitor in the sample run.'
        ),
        default=TransmissionMode.current_run,
    )


class BeamCenterXY(pydantic.BaseModel):
    """Beam center position in detector coordinates."""

    x: float = pydantic.Field(
        default=0.0,
        title='X',
        description='Beam center X coordinate.',
    )
    y: float = pydantic.Field(
        default=0.0,
        title='Y',
        description='Beam center Y coordinate.',
    )
    unit: parameter_models.LengthUnit = pydantic.Field(
        default=parameter_models.LengthUnit.METER,
        description='Unit of the beam center coordinates.',
    )

    def get_vector(self) -> sc.Variable:
        """Get the beam center as a 3D scipp vector."""
        return sc.vector([self.x, self.y, 0.0], unit=self.unit.value)


loki_aux_sources = AuxSources(
    {
        'incident_monitor': AuxInput(
            choices=('beam_monitor_m1',),
            default='beam_monitor_m1',
            title='Incident Monitor',
            description='Incident beam monitor for normalization.',
        ),
        'transmission_monitor': AuxInput(
            choices=('beam_monitor_m3',),
            default='beam_monitor_m3',
            title='Transmission Monitor',
            description='Transmission monitor for sample transmission calculation.',
        ),
    }
)


def _make_1d_q_template() -> sc.DataArray:
    """Create an empty 1D template for I(Q) output."""
    return sc.DataArray(
        sc.zeros(dims=['Q'], shape=[0], unit='dimensionless'),
        coords={'Q': sc.arange('Q', 0, unit='1/angstrom')},
    )


def _make_1d_wavelength_template() -> sc.DataArray:
    """Create an empty 1D template for wavelength-binned output."""
    return sc.DataArray(
        sc.zeros(dims=['wavelength'], shape=[0], unit='dimensionless'),
        coords={'wavelength': sc.arange('wavelength', 0, unit='angstrom')},
    )


class IofQOutputs(WorkflowOutputsBase):
    """Outputs for the I(Q) workflow."""

    i_of_q: sc.DataArray = pydantic.Field(
        default_factory=_make_1d_q_template,
        title='I(Q)',
        description='Scattered intensity as a function of momentum transfer Q.',
    )
    transmission_fraction: sc.DataArray = pydantic.Field(
        default_factory=_make_1d_wavelength_template,
        title='Transmission Fraction',
        description='Sample transmission fraction calculated from monitor ratio.',
    )


class SansWorkflowParams(pydantic.BaseModel):
    q_edges: parameter_models.QEdges = pydantic.Field(
        title='Q bins',
        description='Define the bin edges for binning in Q.',
        default=parameter_models.QEdges(
            start=0.01,
            stop=0.3,
            num_bins=20,
            unit=parameter_models.QUnit.INVERSE_ANGSTROM,
        ),
    )
    wavelength_edges: parameter_models.WavelengthEdges = pydantic.Field(
        title='Wavelength bins',
        description='Define the bin edges for binning in wavelength.',
        default=parameter_models.WavelengthEdges(
            start=1.0,
            stop=13.0,
            num_bins=100,
            unit=parameter_models.WavelengthUnit.ANGSTROM,
        ),
    )
    beam_center: BeamCenterXY = pydantic.Field(
        title='Beam center',
        description='Beam center position in the detector plane.',
        default_factory=BeamCenterXY,
    )
    transmission: TransmissionOptions = pydantic.Field(
        title='Transmission',
        description=(
            'Transmission correction settings.'
            ' The standard SANS transmission calculation normalizes sample'
            ' monitors against an empty-beam measurement;'
            ' this is not available in live mode.'
        ),
        default_factory=TransmissionOptions,
    )


# Detector names for LOKI
detector_names = [f'loki_detector_{bank}' for bank in range(9)]

# f144 log streams for LOKI.
# The detector carriage readback is the position dependency of loki_detector_0
# (depends_on -> /entry/instrument/detector_carriage/value in the NeXus file).
f144_log_streams = {
    'detector_carriage': {
        'source': 'LOKI-DtCar1:MC-LinX-01:Mtr.RBV',
        'topic': 'loki_motion',
        'units': 'mm',
    },
}

# Create instrument
instrument = Instrument(
    name='loki',
    detector_names=detector_names,
    monitors=[
        'beam_monitor_m0',
        'beam_monitor_m1',
        'beam_monitor_m2',
        'beam_monitor_m3',
        'beam_monitor_m4',
    ],
    f144_attribute_registry={
        name: {'units': info['units']} for name, info in f144_log_streams.items()
    },
    source_metadata={
        'loki_detector_0': SourceMetadata(title='Rear'),
        'loki_detector_1': SourceMetadata(title='Mid Top'),
        'loki_detector_2': SourceMetadata(title='Mid Left'),
        'loki_detector_3': SourceMetadata(title='Mid Bottom'),
        'loki_detector_4': SourceMetadata(title='Mid Right'),
        'loki_detector_5': SourceMetadata(title='Front Top'),
        'loki_detector_6': SourceMetadata(title='Front Left'),
        'loki_detector_7': SourceMetadata(title='Front Bottom'),
        'loki_detector_8': SourceMetadata(title='Front Right'),
        'beam_monitor_m0': SourceMetadata(
            title='Beam Monitor 0', description='Upstream, z = -16.8 m'
        ),
        'beam_monitor_m1': SourceMetadata(
            title='Beam Monitor 1', description='Upstream, z = -8.4 m'
        ),
        'beam_monitor_m2': SourceMetadata(
            title='Beam Monitor 2', description='Upstream, z = -2.04 m'
        ),
        'beam_monitor_m3': SourceMetadata(
            title='Beam Monitor 3', description='Downstream, z = +0.2 m'
        ),
        'beam_monitor_m4': SourceMetadata(
            title='Beam Monitor 4',
            description='Downstream, movable (on detector carriage)',
        ),
        'detector_carriage': SourceMetadata(
            title='Rear Detector Carriage',
            description='Rear detector carriage position w.r.t. its zero at z=5.098 m '
            'after the sample.',
        ),
    },
)

# Register instrument
instrument_registry.register(instrument)

# All monitors get the standard TOA histogram.
monitor_handle = register_monitor_workflow_specs(
    instrument, instrument.monitors, params=MonitorDataParams
)

# beam_monitor_m3 is pixellated (pixel IDs 4-8). Reuse the detector view workflow
# with an identity logical projection to get per-pixel counts with TOA histogramming.
# detector_number is hard-coded because current NeXus files do not contain
# pixel IDs for this monitor.
instrument.configure_pixellated_monitor(
    'beam_monitor_m3',
    detector_number=sc.array(dims=['event_id'], values=[4, 5, 6, 7, 8], unit=None),
)
instrument.add_logical_view(
    name='monitor_counts_per_pixel',
    title='Beam monitor: counts per pixel',
    description='Per-pixel event counts for pixellated beam monitors.',
    source_names=['beam_monitor_m3'],
    group=MONITORS,
    roi_support=False,
    output_ndim=1,
)

xy_projection_handle = register_detector_view_spec(
    instrument=instrument,
    projection='xy_plane',
    source_names=detector_names,
    aux_sources=DetectorROIAuxSources(dynamic_transforms=LOKI_DYNAMIC_TRANSFORMS),
)

# Register tube view for all detector banks
instrument.add_logical_view(
    name='tube_view',
    title='Tube View',
    description='Sum over straw and pixel dimensions to show layer x tube counts.',
    source_names=detector_names,
    transform=get_tube_view,
    output_ndim=2,
    reduction_dim=['straw', 'pixel'],
)

# Register I(Q) workflow spec
i_of_q_handle = instrument.register_spec(
    name='i_of_q',
    version=1,
    title='I(Q)',
    description=(
        'SANS I(Q) reduction for LOKI. Converts detector event data into'
        ' scattered intensity as a function of momentum transfer Q.'
        ' Direct-beam normalization (flat/efficiency correction) is not applied and '
        'currently the transmission does not take into a account and empty run.'
    ),
    source_names=detector_names,
    aux_sources=loki_aux_sources,
    outputs=IofQOutputs,
    params=SansWorkflowParams,
)
