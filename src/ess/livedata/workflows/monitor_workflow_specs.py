# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Lightweight monitor workflow spec registration (no heavy dependencies)."""

from __future__ import annotations

import abc
from typing import ClassVar

import pydantic
import scipp as sc

from .. import parameter_models
from ..config.device_contract import COUNTS_TOTAL_DEVICE
from ..config.instrument import Instrument
from ..config.workflow_spec import (
    MONITORS,
    AuxSources,
    CumulativeOutput,
    OutputView,
    WindowOutput,
    WorkflowOutputsBase,
)
from .detector_view_specs import (
    CoordinateMode,
    CoordinateModeSettings,
    TOAOnlyCoordinateModeSettings,
    WavelengthOnlyCoordinateModeSettings,
)
from .workflow_factory import SpecHandle

_TOA_EDGES_DESCRIPTION = (
    "Time of arrival (TOA) is the time elapsed since the most recent source "
    "pulse. These edges define the histogram bins in TOA mode. The default "
    f"range spans one pulse period of the 14 Hz ESS source "
    f"(0 to {parameter_models.ESS_PULSE_PERIOD_MS} ms); events outside the "
    "range are excluded from the histogram."
)


class MonitorDataParamsBase(pydantic.BaseModel, abc.ABC):
    """Common interface for monitor histogram parameter models.

    Subclasses expose the edges and range filter for the coordinate mode they
    offer, narrowing :attr:`coordinate_mode` to the modes they support. This
    lets a single workflow factory (:func:`create_monitor_workflow_factory`)
    serve every monitor spec regardless of which coordinate modes it offers.
    """

    coordinate_mode: CoordinateModeSettings = pydantic.Field(
        title="Coordinate Mode",
        description="Select coordinate system for monitor data.",
        default_factory=CoordinateModeSettings,
    )

    @abc.abstractmethod
    def get_active_edges(self) -> sc.Variable:
        """Return the edges for the active coordinate mode."""

    @abc.abstractmethod
    def get_active_range(self) -> tuple[sc.Variable, sc.Variable] | None:
        """Return the range filter for the active coordinate mode, if enabled."""

    def get_coordinate_mode(self) -> CoordinateMode:
        """Return the active coordinate mode."""
        return self.coordinate_mode.mode


class TOAOnlyMonitorDataParams(MonitorDataParamsBase):
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
        description=_TOA_EDGES_DESCRIPTION,
        default=parameter_models.TOAEdges(
            start=0.0,
            stop=parameter_models.ESS_PULSE_PERIOD_MS,
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


class MonitorDataParams(MonitorDataParamsBase):
    """Parameters for monitor histogram workflow offering both coordinate modes."""

    # TOA (time-of-arrival) settings
    toa_edges: parameter_models.TOAEdges = pydantic.Field(
        title="Time of Arrival Edges",
        description=_TOA_EDGES_DESCRIPTION,
        default=parameter_models.TOAEdges(
            start=0.0,
            stop=parameter_models.ESS_PULSE_PERIOD_MS,
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


class WavelengthMonitorDataParams(MonitorDataParamsBase):
    """Monitor data parameters restricted to wavelength mode."""

    coordinate_mode: WavelengthOnlyCoordinateModeSettings = pydantic.Field(
        title="Coordinate Mode",
        description="Select coordinate system for monitor data. This workflow "
        "always converts to wavelength.",
        default_factory=WavelengthOnlyCoordinateModeSettings,
    )
    wavelength_edges: parameter_models.WavelengthEdges = pydantic.Field(
        title="Wavelength Edges",
        description="Wavelength edges for histogramming.",
        default=parameter_models.WavelengthEdges(
            start=1.0,
            stop=10.0,
            num_bins=100,
            unit=parameter_models.WavelengthUnit.ANGSTROM,
        ),
    )
    wavelength_range: parameter_models.WavelengthRangeFilter = pydantic.Field(
        title="Wavelength Range",
        description="Wavelength range filter.",
        default=parameter_models.WavelengthRangeFilter(),
    )

    def get_active_edges(self) -> sc.Variable:
        """Return the wavelength edges."""
        return self.wavelength_edges.get_edges()

    def get_active_range(self) -> tuple[sc.Variable, sc.Variable] | None:
        """Return the wavelength range if enabled."""
        return self.wavelength_range.range if self.wavelength_range.enabled else None


def _make_histogram_template(dim: str, unit: str) -> sc.DataArray:
    """Empty histogram template over the spectral coordinate of a coordinate mode."""
    return sc.DataArray(
        sc.zeros(dims=[dim], shape=[0], unit='counts'),
        coords={dim: sc.arange(dim, 0, unit=unit)},
    )


class MonitorHistogramOutputs(WorkflowOutputsBase):
    """Outputs for the monitor histogram workflow."""

    output_views: ClassVar[tuple[OutputView, ...]] = (
        OutputView(
            name='histogram',
            title='Histogram',
            fields=('cumulative', 'current'),
            description=(
                'Monitor histogram. With "since run start" shows accumulated '
                'counts; with "latest update" or a window, shows recent counts.'
            ),
            params=('coordinate_mode', 'toa_edges', 'wavelength_edges'),
        ),
        OutputView(
            name='total_counts',
            title='Total',
            fields=('counts_total_cumulative', 'counts_total'),
            description=(
                'Total number of monitor events. With "since run start" shows '
                'the accumulated total; with "latest update" or a window, shows '
                'recent counts.'
            ),
        ),
        OutputView(
            name='total_in_range',
            title='Total in range',
            fields=('counts_in_toa_range_cumulative', 'counts_in_toa_range'),
            description=(
                'Number of monitor events within the configured range filter. '
                'With "since run start" shows the accumulated total; with '
                '"latest update" or a window, shows recent counts.'
            ),
            params=('coordinate_mode', 'toa_range', 'wavelength_range'),
        ),
    )

    # Field names are kept stable as wire-format identifiers (ResultKey,
    # da00 serialisation) and are referenced by ``output_views`` above.

    cumulative: CumulativeOutput = pydantic.Field(
        default_factory=lambda: _make_histogram_template('time_of_arrival', 'ms'),
        title='Histogram',
        description=(
            'Monitor histogram accumulated since the start of the run. '
            'In wavelength mode accumulation restarts if the monitor moves; in '
            'TOA mode a move is not detected and counts accumulate across it.'
        ),
    )
    current: WindowOutput = pydantic.Field(
        default_factory=lambda: _make_histogram_template('time_of_arrival', 'ms'),
        title='Histogram update',
        description=(
            'Monitor histogram for the latest update interval only. '
            'Resets each update interval.'
        ),
    )
    counts_total: WindowOutput = pydantic.Field(
        default_factory=lambda: sc.DataArray(sc.scalar(0, unit='counts')),
        title='Total',
        description=(
            'Total number of monitor events for the latest update interval only. '
            'Resets each update interval.'
        ),
    )
    counts_in_toa_range: WindowOutput = pydantic.Field(
        default_factory=lambda: sc.DataArray(sc.scalar(0, unit='counts')),
        title='Total in range',
        description=(
            'Number of monitor events within the configured range filter '
            'for the latest update interval only. Resets each update interval.'
        ),
    )
    counts_total_cumulative: CumulativeOutput = pydantic.Field(
        default_factory=lambda: sc.DataArray(sc.scalar(0, unit='counts')),
        title='Total',
        description='Total number of monitor events accumulated since the start '
        'of the run.',
    )
    counts_in_toa_range_cumulative: CumulativeOutput = pydantic.Field(
        default_factory=lambda: sc.DataArray(sc.scalar(0, unit='counts')),
        title='Total in range',
        description='Number of monitor events within the configured range filter '
        'accumulated since the start of the run.',
    )


class WavelengthMonitorHistogramOutputs(MonitorHistogramOutputs):
    """Monitor histogram outputs binned in wavelength.

    Only the histogram templates differ from the TOA outputs: they declare the
    spectral coordinate the workflow actually produces, which the UI uses to
    pick a plotter. The scalar count outputs are coordinate-mode independent.
    """

    cumulative: CumulativeOutput = pydantic.Field(
        default_factory=lambda: _make_histogram_template('wavelength', 'angstrom'),
        title='Histogram',
        description=(
            'Monitor histogram accumulated since the start of the run. '
            'Accumulation restarts if the monitor moves.'
        ),
    )
    current: WindowOutput = pydantic.Field(
        default_factory=lambda: _make_histogram_template('wavelength', 'angstrom'),
        title='Histogram update',
        description=(
            'Monitor histogram for the latest update interval only. '
            'Resets each update interval.'
        ),
    )


def register_monitor_workflow_specs(
    instrument: Instrument,
    source_names: list[str],
    params: type[MonitorDataParamsBase] = MonitorDataParams,
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
        Parameter model class for the workflow, a MonitorDataParamsBase
        subclass. Defaults to MonitorDataParams. Instruments can provide the
        restricted TOAOnlyMonitorDataParams or a subclass with additional fields
        (e.g., for instrument-specific configuration like chopper mode selection).
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
        description = f"{description}<br><br>{extra_description}"

    return instrument.register_spec(
        group=MONITORS,
        name='monitor_histogram',
        version=1,
        title="Beam monitor",
        description=description,
        source_names=source_names,
        aux_sources=aux_sources,
        params=params,
        outputs=MonitorHistogramOutputs,
        # Every instrument's cumulative monitor total is a NICOS derived device.
        device_outputs=COUNTS_TOTAL_DEVICE,
    )


def register_monitor_wavelength_workflow_specs(
    instrument: Instrument,
    source_names: list[str],
    params: type[MonitorDataParamsBase] = WavelengthMonitorDataParams,
    aux_sources: AuxSources | None = None,
) -> SpecHandle | None:
    """
    Register the wavelength-mode monitor spec, separate from the TOA spec.

    A separate spec rather than a coordinate-mode parameter on the TOA spec:
    the wavelength path consumes a lookup table as gated context, and gating is
    resolved per ``(workflow_id, source_name)`` and never per parameter value,
    so a combined spec would gate its TOA jobs on a table they never read
    (ADR 0010). Register this only for instruments that can supply a table.

    Declares no ``device_outputs``: the NICOS monitor-total device is owned by
    the TOA spec registered by :func:`register_monitor_workflow_specs`, and a
    cumulative event count does not depend on the coordinate mode. Declaring it
    on both specs would render the same device name twice, which
    :class:`~ess.livedata.config.device_contract.DeviceContract` rejects.

    Parameters
    ----------
    instrument:
        The instrument to register the workflow spec for.
    source_names:
        Monitor names for which to register the workflow. If empty, returns
        None without registering.
    params:
        Parameter model class, a :class:`MonitorDataParamsBase` subclass
        restricted to wavelength. Instruments needing extra fields subclass
        :class:`WavelengthMonitorDataParams`.
    aux_sources:
        Optional auxiliary source specification.

    Returns
    -------
    :
        SpecHandle for later factory attachment, or None if no monitors.
    """
    if not source_names:
        return None

    return instrument.register_spec(
        group=MONITORS,
        name='monitor_histogram_wavelength',
        version=1,
        title="Beam monitor (wavelength)",
        description=(
            "Histogrammed and time-integrated beam monitor, converted to "
            "wavelength. The monitor is histogrammed or rebinned into "
            "specified wavelength bins."
        ),
        source_names=source_names,
        aux_sources=aux_sources,
        params=params,
        outputs=WavelengthMonitorHistogramOutputs,
    )


def create_monitor_workflow_factory(source_name: str, params: MonitorDataParamsBase):
    """
    Factory function for monitor workflow from monitor data parameters.

    Wraps :func:`create_monitor_workflow`, unpacking the params. It serves any
    spec whose params subclass :class:`MonitorDataParamsBase`, including the
    TOA-only restricted variant. Instruments needing TOF lookup tables for
    wavelength mode (DREAM, LOKI) provide their own factory instead.

    Defined here so the params type hint can be properly resolved by the
    workflow factory registration system.
    """
    from .monitor_workflow import create_monitor_workflow

    return create_monitor_workflow(
        source_name=source_name,
        edges=params.get_active_edges(),
        range_filter=params.get_active_range(),
        coordinate_mode=params.get_coordinate_mode(),
    )
