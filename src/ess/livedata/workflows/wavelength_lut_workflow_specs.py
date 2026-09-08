# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Spec registration for the wavelength lookup-table workflow."""

from __future__ import annotations

import pydantic
import scipp as sc

from ..config.instrument import Instrument
from ..config.workflow_spec import REDUCTION, WorkflowOutputsBase
from ..parameter_models import LengthUnit, TimeUnit, parse_number_list
from .workflow_factory import SpecHandle

#: Logical primary stream name for the synthesized chopper-cascade tick. The
#: synthesizer emits a single ``LogData`` message with this stream name once
#: all chopper setpoints are reached (vacuously true for chopperless
#: instruments). The workflow uses presence of this signal as its trigger.
CHOPPER_CASCADE_SOURCE = 'chopper_cascade'

#: Workflow ``name`` in the spec.
WAVELENGTH_LUT_WORKFLOW_NAME = 'wavelength_lut'

#: Output key for the wavelength bands diagnostic. Field name on
#: :class:`WavelengthLutOutputs`.
WAVELENGTH_BANDS_OUTPUT = 'chopper_cascade_bands'


class Pulse(pydantic.BaseModel):
    """Source pulse properties.

    The unit of ``frequency`` is hardcoded to Hz: there is no realistic case
    where a neutron-source pulse frequency would be expressed in kHz/MHz, so
    we skip the dropdown and keep the model tight.
    """

    frequency: float = pydantic.Field(
        default=14.0, gt=0.0, description="Pulse frequency in Hz."
    )
    auto_stride: bool = pydantic.Field(
        default=True,
        description=(
            "Guess the pulse stride from the chopper rotation frequencies "
            "(choppers at zero frequency are ignored). Disable to set the "
            "stride manually below."
        ),
    )
    stride: int = pydantic.Field(
        default=1,
        ge=1,
        description="Pulse stride (used only when auto-detection is disabled).",
    )

    def get_period(self) -> sc.Variable:
        """Return the pulse period as a scipp scalar."""
        return sc.scalar(1.0 / self.frequency, unit='s')


class DistanceResolution(pydantic.BaseModel):
    """Resolution of the distance axis in the lookup table."""

    value: float = pydantic.Field(default=0.1, description="Distance bin resolution.")
    unit: LengthUnit = pydantic.Field(
        default=LengthUnit.METER, description="Unit of the distance resolution."
    )

    def get(self) -> sc.Variable:
        return sc.scalar(self.value, unit=self.unit.value)


class TimeResolution(pydantic.BaseModel):
    """Resolution of the event-time-offset axis in the lookup table."""

    value: float = pydantic.Field(default=250.0, description="Time bin resolution.")
    unit: TimeUnit = pydantic.Field(
        default=TimeUnit.MICROSECOND, description="Unit of the time resolution."
    )

    def get(self) -> sc.Variable:
        return sc.scalar(self.value, unit=self.unit.value)


class SourceOffset(pydantic.BaseModel):
    """Offset of the neutron source along the beam, relative to the file.

    The chopper cascade measures every chopper's flight distance from the source
    position loaded from the geometry artifact. This shifts that reference along
    the beam (the lab +Z axis): a positive offset moves the source downstream
    (toward the choppers), a negative offset upstream.
    """

    offset: float = pydantic.Field(
        default=0.0,
        description=(
            "Signed source offset along the beam; positive moves the source "
            "downstream (toward the choppers)."
        ),
    )
    unit: LengthUnit = pydantic.Field(
        default=LengthUnit.METER, description="Unit of the source offset."
    )

    def get(self) -> sc.Variable:
        """Return the offset as a beam-aligned (Z) displacement vector."""
        return sc.scalar(self.offset, unit=self.unit.value) * sc.vector([0.0, 0.0, 1.0])


class CascadeBands(pydantic.BaseModel):
    """Configuration of the *Chopper cascade bands* output.

    That diagnostic always draws one curve at the source and one at each chopper
    (at their exact distances). ``distances`` adds *further* curves at arbitrary
    beamline points, propagating the cascade forward to each (typically the
    monitor and detector positions, but any point of interest is valid); entered
    as a comma-separated list, since the parameter widget has no native list
    input. ``num_bins`` sets how finely every curve samples the
    event-time-offset axis.
    """

    distances: str = pydantic.Field(
        default='',
        description="Comma-separated distances from the source, e.g. '6.2, 9.8'.",
    )
    unit: LengthUnit = pydantic.Field(
        default=LengthUnit.METER, description="Unit of the cascade-band distances."
    )
    num_bins: int = pydantic.Field(
        default=1000,
        ge=1,
        description=(
            "Number of event-time-offset bins sampling each curve. Independent "
            "of the lookup-table time resolution: chosen fine enough to resolve "
            "narrow transmitted bands across the frame."
        ),
    )

    @pydantic.field_validator('distances')
    @classmethod
    def _check(cls, v: str) -> str:
        parse_number_list(v)  # raises on malformed input
        return v

    def get_distances(self) -> sc.Variable:
        return sc.array(
            dims=['distance'],
            values=parse_number_list(self.distances),
            unit=self.unit.value,
        )


class WavelengthLutParams(pydantic.BaseModel):
    """User-facing parameters for the wavelength lookup-table workflow."""

    pulse: Pulse = pydantic.Field(
        title='Source pulse',
        description='Source pulse frequency and stride.',
        default_factory=Pulse,
    )
    source: SourceOffset = pydantic.Field(
        title='Source',
        description='Offset of the neutron source along the beam.',
        default_factory=SourceOffset,
    )
    distance_resolution: DistanceResolution = pydantic.Field(
        title='Distance resolution',
        description='Resolution of the distance axis in the lookup table.',
        default_factory=DistanceResolution,
    )
    time_resolution: TimeResolution = pydantic.Field(
        title='Time resolution',
        description='Resolution of the event-time-offset axis in the lookup table.',
        default_factory=TimeResolution,
    )
    cascade_bands: CascadeBands = pydantic.Field(
        title='Chopper cascade bands',
        description=(
            'Settings for the "Chopper cascade bands" output. The source and '
            'every chopper always get a curve; add curves at further beamline '
            'distances (typically monitor and detector positions) and set their '
            'event-time-offset sampling here.'
        ),
        default_factory=CascadeBands,
    )


def _empty_table_template() -> sc.DataArray:
    """Empty placeholder used by the spec until the first computation."""
    return sc.DataArray(
        sc.zeros(dims=['distance', 'event_time_offset'], shape=[0, 0], unit='angstrom'),
        coords={
            'distance': sc.array(dims=['distance'], values=[], unit='m'),
            'event_time_offset': sc.array(
                dims=['event_time_offset'], values=[], unit='us'
            ),
        },
    )


#: Output field and context stream carrying the detectors' shared table.
DETECTOR_LUT_OUTPUT = 'detector_lut'

#: Output field and context stream carrying the monitors' shared table.
MONITOR_LUT_OUTPUT = 'monitor_lut'

#: Prefix for the context stream names, keeping the namespace greppable and
#: collision-free against device and motion stream names.
LUT_STREAM_PREFIX = 'wavelength_lut'

#: Context stream name per output. Two streams rather than one: a monitor job
#: has no use for the detectors' dense table, which is the large payload.
LUT_STREAM_NAMES = {
    DETECTOR_LUT_OUTPUT: f'{LUT_STREAM_PREFIX}/detectors',
    MONITOR_LUT_OUTPUT: f'{LUT_STREAM_PREFIX}/monitors',
}


class WavelengthLutOutputs(WorkflowOutputsBase):
    """Outputs of the lookup-table workflow.

    Two tables, not one per component: a table is a function of ``distance``
    and ``event_time_offset`` alone, so components sharing a stretch of
    beamline share a table (see
    :mod:`~ess.livedata.workflows.lut_blocks` for how the blocks are laid out).
    The outputs are therefore the same for every instrument.
    """

    detector_lut: sc.DataArray = pydantic.Field(
        default_factory=_empty_table_template,
        title='Wavelength lookup table: detectors',
        description=(
            'Wavelength as a function of distance and event-time-offset over the '
            'flight-path range of every detector, computed from the current '
            'chopper cascade. One dense block covering all banks.'
        ),
    )
    monitor_lut: sc.DataArray = pydantic.Field(
        default_factory=_empty_table_template,
        title='Wavelength lookup table: monitors',
        description=(
            'Wavelength as a function of distance and event-time-offset at each '
            'monitor, computed from the current chopper cascade. A few rows per '
            'monitor and nothing in between, so the distance axis is not '
            'uniform: plot with the "Overlay 1D" plotter, one curve per row.'
        ),
    )
    chopper_cascade_bands: sc.DataArray = pydantic.Field(
        default_factory=_empty_table_template,
        title='Chopper cascade bands',
        description=(
            'Wavelength band transmitted along the beamline: always one curve at '
            'the source and one at each chopper (at their exact distances), plus '
            'a curve at each distance configured in the "Cascade bands" '
            'parameter section. Plot with the "Overlay 1D" plotter: one curve per '
            'distance (identified by its value in metres). A curve that vanishes '
            '(all-NaN) marks where the beam is blocked. Unlike the lookup table, '
            'this resolves closely-spaced choppers regardless of distance '
            'resolution.'
        ),
    )


def register_wavelength_lut_workflow_spec(
    instrument: Instrument,
    *,
    params: type[WavelengthLutParams] = WavelengthLutParams,
) -> SpecHandle:
    """Register the wavelength lookup-table workflow spec for ``instrument``.

    Grouped under :data:`REDUCTION` for the UI — it is a reduction input,
    not a per-device timeseries — but hosted by the ``timeseries`` service
    (explicit ``service`` override below). Both are f144-driven and share
    the same source/preprocessor plumbing; in v1 the chopper PVs feeding
    the synthesizer are themselves f144 streams that the timeseries
    service can plot. The ``ChopperSynthesizer`` emitting the synthetic
    primary trigger is a temporary stand-in for an upstream-side
    ``chopper_cascade_reached`` f144 stream; once the producer publishes
    that directly, the wrapper drops out.

    The workflow's only ``source_name`` is the synthetic ``chopper_cascade``
    stream emitted by ``ChopperSynthesizer``. The factory must be attached
    later via the returned handle.
    """
    return instrument.register_spec(
        group=REDUCTION,
        service='timeseries',
        name=WAVELENGTH_LUT_WORKFLOW_NAME,
        version=1,
        title='Wavelength lookup table',
        description=(
            'Compute a wavelength lookup table from the current chopper-cascade '
            'configuration. Refires when chopper setpoints change.'
        ),
        source_names=[CHOPPER_CASCADE_SOURCE],
        params=params,
        outputs=WavelengthLutOutputs,
        context_outputs=LUT_STREAM_NAMES,
        reset_on_run_transition=False,
        # The LUT is recomputed from the retained chopper-setpoint history on
        # the next trigger regardless, so a reset achieves nothing.
        supports_reset=False,
    )
