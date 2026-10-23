# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Lightweight monitor workflow spec registration (no heavy dependencies)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import pydantic
import scipp as sc

from .. import parameter_models
from ..config.device_contract import MONITOR_DEVICES
from ..config.instrument import Instrument
from ..config.spectral_coords import (
    spectral_edges_template_coords,
    spectral_range_template_coords,
)
from ..config.workflow_spec import (
    MONITORS,
    AuxSources,
    CumulativeOutput,
    OutputView,
    WindowOutput,
    WorkflowOutputsBase,
)
from .workflow_factory import SpecHandle


class MonitorDataParams(pydantic.BaseModel):
    """Parameters for the monitor histogram workflow offering both spectral axes."""

    spectral_axis: parameter_models.SpectralAxis = pydantic.Field(
        title="Spectral Axis",
        description="Axis along which the monitor histograms events.",
        default_factory=parameter_models.TOASpectralAxis,
    )


class TOAOnlyMonitorDataParams(MonitorDataParams):
    """
    Monitor data parameters restricted to the time-of-arrival axis.

    Use this where wavelength cannot work: it needs each monitor's flight
    path, taken from the instrument's geometry file.
    """

    spectral_axis: parameter_models.TOASpectralAxis = pydantic.Field(
        title="Spectral Axis",
        description="Axis along which the monitor histograms events. Only time "
        "of arrival (TOA) is available for this instrument.",
        default_factory=parameter_models.TOASpectralAxis,
    )


def _make_scalar_template() -> sc.DataArray:
    """Create an empty template for a total summed over the spectral axis."""
    return sc.DataArray(
        sc.scalar(0, unit='counts'), coords=spectral_range_template_coords()
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
            params=('spectral_axis',),
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
            params=('spectral_axis',),
        ),
    )

    # Field names are kept stable as wire-format identifiers (ResultKey,
    # da00 serialisation) and are referenced by ``output_views`` above.

    cumulative: CumulativeOutput = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=['time_of_arrival'], shape=[0], unit='counts'),
            coords=spectral_edges_template_coords(),
        ),
        title='Histogram',
        description=(
            'Monitor histogram accumulated since the start of the run. '
            'On the wavelength axis accumulation restarts if the monitor moves; '
            'on the TOA axis a move is not detected and counts accumulate across '
            'it.'
        ),
    )
    current: WindowOutput = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=['time_of_arrival'], shape=[0], unit='counts'),
            coords=spectral_edges_template_coords(),
        ),
        title='Histogram update',
        description=(
            'Monitor histogram for the latest update interval only. '
            'Resets each update interval.'
        ),
    )
    counts_total: WindowOutput = pydantic.Field(
        default_factory=_make_scalar_template,
        title='Total',
        description=(
            'Total number of monitor events for the latest update interval only. '
            'Resets each update interval.'
        ),
    )
    counts_in_toa_range: WindowOutput = pydantic.Field(
        default_factory=_make_scalar_template,
        title='Total in range',
        description=(
            'Number of monitor events within the configured range filter '
            'for the latest update interval only. Resets each update interval.'
        ),
    )
    counts_total_cumulative: CumulativeOutput = pydantic.Field(
        default_factory=_make_scalar_template,
        title='Total',
        description='Total number of monitor events accumulated since the start '
        'of the run.',
    )
    counts_in_toa_range_cumulative: CumulativeOutput = pydantic.Field(
        default_factory=_make_scalar_template,
        title='Total in range',
        description='Number of monitor events within the configured range filter '
        'accumulated since the start of the run.',
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
        Parameter model class for the workflow, MonitorDataParams or a
        subclass. Instruments whose monitors cannot offer the wavelength axis
        provide TOAOnlyMonitorDataParams.
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
        # Every instrument's cumulative monitor total and histogram are NICOS
        # derived devices.
        device_outputs=MONITOR_DEVICES,
    )


def make_monitor_workflow_factory(instrument: Instrument) -> Callable[..., Any]:
    """
    Return the shared monitor workflow factory for ``instrument``.

    The factory wraps :func:`create_monitor_workflow`, unpacking the params. It
    serves any spec whose params subclass :class:`MonitorDataParams`,
    including the TOA-only restricted variant. On the wavelength axis it passes
    the instrument's geometry file, which the monitor needs for ``Ltotal``; the
    TOA axis loads no file.

    The returned function is defined here so its params type hint can be
    resolved by the workflow factory registration system.
    """

    def factory(source_name: str, params: MonitorDataParams) -> Any:
        from .monitor_workflow import create_monitor_workflow

        axis = params.spectral_axis
        return create_monitor_workflow(
            source_name=source_name,
            edges=axis.edges.get_edges(),
            range_filter=None if axis.range_filter is None else axis.range_filter.range,
            coordinate_mode=axis.kind,
            geometry_filename=(
                instrument.nexus_file if axis.kind == 'wavelength' else None
            ),
        )

    return factory
