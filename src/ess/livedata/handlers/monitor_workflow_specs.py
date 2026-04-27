# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Lightweight monitor workflow spec registration (no heavy dependencies)."""

from __future__ import annotations

import pydantic
import scipp as sc

from .. import parameter_models
from ..config.instrument import Instrument
from ..config.workflow_spec import AuxSources, WorkflowOutputsBase
from ..handlers.detector_view_specs import CoordinateMode, CoordinateModeSettings
from ..handlers.workflow_factory import SpecHandle


class TOAOnlyCoordinateModeSettings(pydantic.BaseModel):
    """
    Coordinate mode settings restricted to TOA only.

    Use this for instruments that don't have TOF lookup tables available.
    """

    mode: CoordinateMode = pydantic.Field(
        default='toa',
        description="Coordinate system for event data. Only TOA (time-of-arrival) "
        "is available for this instrument.",
    )

    @pydantic.field_validator('mode')
    @classmethod
    def _validate_toa_only(cls, v: CoordinateMode) -> CoordinateMode:
        if v != 'toa':
            raise ValueError(
                f"Only 'toa' mode is supported for this instrument, got '{v}'. "
                "TOF mode requires instrument-specific lookup tables."
            )
        return v


class TOAOnlyMonitorDataParams(pydantic.BaseModel):
    """
    Monitor data parameters restricted to TOA mode only.

    Use this for instruments that don't have TOF lookup tables available.
    """

    coordinate_mode: TOAOnlyCoordinateModeSettings = pydantic.Field(
        title="Coordinate Mode",
        description="Select coordinate system for monitor data. "
        "Only TOA mode is available for this instrument.",
        default_factory=TOAOnlyCoordinateModeSettings,
    )
    toa_edges: parameter_models.TOAEdges = pydantic.Field(
        title="Time of Arrival Edges",
        description="Time of arrival edges for histogramming.",
        default=parameter_models.TOAEdges(
            start=0.0,
            stop=1000.0 / 14,
            num_bins=100,
            unit=parameter_models.TimeUnit.MS,
        ),
    )
    toa_range: parameter_models.TOARange = pydantic.Field(
        title="Time of Arrival Range",
        description="Time of arrival range filter.",
        default=parameter_models.TOARange(),
    )

    def get_active_edges(self) -> sc.Variable:
        """Return the TOA edges."""
        return self.toa_edges.get_edges()

    def get_active_range(self) -> tuple[sc.Variable, sc.Variable] | None:
        """Return the TOA range if enabled."""
        return self.toa_range.range if self.toa_range.enabled else None


class MonitorDataParams(pydantic.BaseModel):
    """Parameters for monitor histogram workflow."""

    coordinate_mode: CoordinateModeSettings = pydantic.Field(
        title="Coordinate Mode",
        description="Select coordinate system for monitor data.",
        default_factory=CoordinateModeSettings,
    )
    # TOA (time-of-arrival) settings
    toa_edges: parameter_models.TOAEdges = pydantic.Field(
        title="Time of Arrival Edges",
        description="Time of arrival edges for histogramming in TOA mode.",
        default=parameter_models.TOAEdges(
            start=0.0,
            stop=1000.0 / 14,
            num_bins=100,
            unit=parameter_models.TimeUnit.MS,
        ),
    )
    toa_range: parameter_models.TOARange = pydantic.Field(
        title="Time of Arrival Range",
        description="Time of arrival range filter for TOA mode.",
        default=parameter_models.TOARange(),
    )
    # Wavelength settings
    wavelength_edges: parameter_models.WavelengthEdges = pydantic.Field(
        title="Wavelength Edges",
        description="Wavelength edges for histogramming in wavelength mode.",
        default=parameter_models.WavelengthEdges(
            start=1.0,
            stop=10.0,
            num_bins=100,
            unit=parameter_models.WavelengthUnit.ANGSTROM,
        ),
    )
    wavelength_range: parameter_models.WavelengthRangeFilter = pydantic.Field(
        title="Wavelength Range",
        description="Wavelength range filter for wavelength mode.",
        default=parameter_models.WavelengthRangeFilter(),
    )

    def get_active_edges(self) -> sc.Variable:
        """Return the edges for the currently selected coordinate mode."""
        match self.coordinate_mode.mode:
            case 'toa':
                return self.toa_edges.get_edges()
            case 'wavelength':
                return self.wavelength_edges.get_edges()

    def get_active_range(self) -> tuple[sc.Variable, sc.Variable] | None:
        """Return the range for the currently selected coordinate mode, if enabled."""
        match self.coordinate_mode.mode:
            case 'toa':
                return self.toa_range.range if self.toa_range.enabled else None
            case 'wavelength':
                return (
                    self.wavelength_range.range
                    if self.wavelength_range.enabled
                    else None
                )


class MonitorHistogramOutputs(WorkflowOutputsBase):
    """Outputs for the monitor histogram workflow."""

    # Field names are legacy identifiers kept for compatibility with existing
    # workflow templates and serialized configs. Titles are user-facing names.

    cumulative: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=['time_of_arrival'], shape=[0], unit='counts'),
            coords={'time_of_arrival': sc.arange('time_of_arrival', 0, unit='ms')},
        ),
        title='Histogram (cumulative)',
        description='Monitor histogram accumulated since the start of the run.',
    )
    current: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=['time_of_arrival'], shape=[0], unit='counts'),
            coords={
                'time_of_arrival': sc.arange('time_of_arrival', 0, unit='ms'),
                'time': sc.scalar(0, unit='ns'),
            },
        ),
        title='Histogram (current)',
        description=(
            'Monitor histogram for the latest update interval only. '
            'Resets each update interval.'
        ),
    )
    counts_total: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.scalar(0, unit='counts'),
            coords={'time': sc.scalar(0, unit='ns')},
        ),
        title='Total (current)',
        description=(
            'Total number of monitor events for the latest update interval only. '
            'Resets each update interval.'
        ),
    )
    counts_in_toa_range: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.scalar(0, unit='counts'),
            coords={'time': sc.scalar(0, unit='ns')},
        ),
        title='Total in interval (current)',
        description=(
            'Number of monitor events within the configured range filter '
            'for the latest update interval only. Resets each update interval.'
        ),
    )


def register_monitor_workflow_specs(
    instrument: Instrument,
    source_names: list[str],
    params: type[MonitorDataParams] = MonitorDataParams,
    aux_sources: AuxSources | None = None,
    extra_description: str | None = None,
) -> SpecHandle | None:
    """
    Register monitor workflow specs (lightweight, no heavy dependencies).

    Parameters
    ----------
    instrument
        The instrument to register the workflow specs for.
    source_names
        List of monitor names (source names) for which to register the workflow.
        If empty, returns None without registering.
    params
        Parameter model class for the workflow. Defaults to MonitorDataParams.
        Instruments can provide a subclass with additional fields (e.g., for
        instrument-specific configuration like chopper mode selection).
    aux_sources
        Optional auxiliary source specification for position or other dynamic data
        streams. Instruments with movable monitors can provide an AuxSources spec
        that maps logical names to f144 position streams.
    extra_description
        Optional text appended to the standard workflow description. Use this to
        document instrument-specific caveats (e.g. that a generic placeholder is
        in use until the real monitor configuration is known).

    Returns
    -------
    SpecHandle for later factory attachment, or None if no monitors.
    """
    if not source_names:
        return None

    description = (
        "Histogrammed and time-integrated beam monitor. The monitor "
        "is histogrammed or rebinned into specified time-of-arrival (TOA) bins."
    )
    if extra_description:
        description = f"{description}\n\n{extra_description}"

    return instrument.register_spec(
        namespace='monitor_data',
        name='monitor_histogram',
        version=1,
        title="Beam monitor",
        description=description,
        source_names=source_names,
        aux_sources=aux_sources,
        params=params,
        outputs=MonitorHistogramOutputs,
    )


def create_monitor_workflow_factory(source_name: str, params: MonitorDataParams):
    """
    Factory function for monitor workflow from MonitorDataParams.

    This is a wrapper around create_monitor_workflow that unpacks the params.
    Defined here so the params type hint can be properly resolved by the
    workflow factory registration system.
    """
    from .monitor_workflow import create_monitor_workflow

    mode = params.coordinate_mode.mode

    return create_monitor_workflow(
        source_name=source_name,
        edges=params.get_active_edges(),
        range_filter=params.get_active_range(),
        coordinate_mode=mode,
    )
