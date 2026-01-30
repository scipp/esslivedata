# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Lightweight monitor workflow spec registration (no heavy dependencies)."""

from __future__ import annotations

import pydantic
import scipp as sc

from .. import parameter_models
from ..config.instrument import Instrument
from ..config.workflow_spec import WorkflowOutputsBase
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
        return self.toa_range.range_ns if self.toa_range.enabled else None


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
    # TOF (time-of-flight) settings
    tof_edges: parameter_models.TOFEdges = pydantic.Field(
        title="Time of Flight Edges",
        description="Time of flight edges for histogramming in TOF mode.",
        default=parameter_models.TOFEdges(
            start=0.0,
            stop=1000.0 / 14,
            num_bins=100,
            unit=parameter_models.TimeUnit.MS,
        ),
    )
    tof_range: parameter_models.TOFRange = pydantic.Field(
        title="Time of Flight Range",
        description="Time of flight range filter for TOF mode.",
        default=parameter_models.TOFRange(),
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
            case 'tof':
                return self.tof_edges.get_edges()
            case 'wavelength':
                return self.wavelength_edges.get_edges()

    def get_active_range(self) -> tuple[sc.Variable, sc.Variable] | None:
        """Return the range for the currently selected coordinate mode, if enabled."""
        match self.coordinate_mode.mode:
            case 'toa':
                return self.toa_range.range_ns if self.toa_range.enabled else None
            case 'tof':
                return self.tof_range.range if self.tof_range.enabled else None
            case 'wavelength':
                return (
                    self.wavelength_range.range
                    if self.wavelength_range.enabled
                    else None
                )


class MonitorHistogramOutputs(WorkflowOutputsBase):
    """Outputs for the monitor histogram workflow."""

    cumulative: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=['time_of_arrival'], shape=[0], unit='counts'),
            coords={'time_of_arrival': sc.arange('time_of_arrival', 0, unit='ms')},
        ),
        title='Cumulative Counts',
        description='Time-integrated monitor counts accumulated over all time.',
    )
    current: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=['time_of_arrival'], shape=[0], unit='counts'),
            coords={
                'time_of_arrival': sc.arange('time_of_arrival', 0, unit='ms'),
                'time': sc.scalar(0, unit='ns'),
            },
        ),
        title='Current Counts',
        description='Monitor counts for the current time window since last update.',
    )
    counts_total: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.scalar(0, unit='counts'),
            coords={'time': sc.scalar(0, unit='ns')},
        ),
        title='Total counts',
        description='Total monitor counts in the current time window.',
    )
    counts_in_toa_range: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.scalar(0, unit='counts'),
            coords={'time': sc.scalar(0, unit='ns')},
        ),
        title='Counts in TOA Range',
        description='Number of monitor events within the configured TOA range filter.',
    )


def register_monitor_workflow_specs(
    instrument: Instrument,
    source_names: list[str],
    params: type[MonitorDataParams] = MonitorDataParams,
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

    Returns
    -------
    SpecHandle for later factory attachment, or None if no monitors.
    """
    if not source_names:
        return None

    return instrument.register_spec(
        namespace='monitor_data',
        name='monitor_histogram',
        version=1,
        title="Beam monitor data",
        description="Histogrammed and time-integrated beam monitor data. The monitor "
        "is histogrammed or rebinned into specified time-of-arrival (TOA) bins.",
        source_names=source_names,
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
    if mode == 'wavelength':
        raise NotImplementedError("wavelength mode not yet implemented for monitors")

    return create_monitor_workflow(
        source_name=source_name,
        edges=params.get_active_edges(),
        range_filter=params.get_active_range(),
        coordinate_mode=mode,
    )
