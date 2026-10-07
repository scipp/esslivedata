# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Export of buffered dashboard data to files.

An export can contain only what :class:`DataService` holds; the dashboard keeps
no other record. Two kinds of export exist, chosen by the type of the params:

- :class:`CurrentValueExportParams`: the latest update of an output, or its
  value accumulated since the run started. Unlike plots, no aggregation over a
  time window is offered: it needs buffered history, which outputs other than
  0-D time series have only while a plot asks for it, and then only for the
  plot's window.
- :class:`HistoryExportParams`: the buffered history of a 0-D time-series
  output, as stored: one point per update, each with its own ``time`` and,
  for per-window outputs, ``start_time``. Optionally with correlation axes: each
  axis is added as a coord holding the axis value in effect at every point (see
  :func:`correlate`), and is also exported as its own time series.

History is buffered only for keys some subscriber asks history of. For 0-D
time series, :class:`TimeseriesRetention` asks for it permanently, so their
history can be exported whether or not they are plotted.

Time coords (``time``, ``start_time``, ``end_time``) are exported as UTC
datetimes. Export names are source names with ``/`` replaced, since HDF5 reads
``/`` as a path separator; the original names are kept in the file's metadata.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

import pydantic
import scipp as sc

from ess.livedata.config.workflow_spec import (
    DataKey,
    Windowing,
    WorkflowId,
    WorkflowSpec,
    find_timeseries_outputs,
)

from .correlation import correlate
from .data_roles import PRIMARY
from .data_service import DataService, DataServiceSubscriber
from .data_subscriber import DataSubscriber
from .extractors import FullHistoryExtractor, LatestValueExtractor, UpdateExtractor
from .nexus_writer import write_nexus
from .plot_orchestrator import (
    DataSourceConfig,
    build_subscription_keys,
    resolve_data_sources,
)
from .plot_params import WindowModeMixin
from .save_filename import build_save_filename

MAX_EXPORT_BYTES = 50 * 1024 * 1024
"""Largest file offered for download.

The file reaches the browser base64-encoded in a websocket message, which is
built on the IOLoop all sessions share; a larger one would stall every session.
"""

_TIME_COORDS = ('time', 'start_time', 'end_time')


class ExportFormat(enum.StrEnum):
    """File formats an export can be written in."""

    nexus = 'nexus'


_EXTENSIONS = {ExportFormat.nexus: 'nxs'}


class FileParams(pydantic.BaseModel):
    format: ExportFormat = pydantic.Field(
        default=ExportFormat.nexus,
        title='Format',
        description=(
            "NeXus: an NXentry with one NXdata group per exported source. Opens in "
            "any HDF5 or NeXus tool; scippnexus loads each group as a DataArray."
        ),
    )


class ExportedValue(enum.StrEnum):
    """Which value of an output a current-value export holds."""

    latest_update = 'latest_update'
    since_start = 'since_start'


class ValueParams(pydantic.BaseModel):
    value: ExportedValue = pydantic.Field(
        default=ExportedValue.latest_update,
        title='Value',
        description=(
            "'latest_update': the most recent update of the output. "
            "'since_start': the output accumulated since the run started."
        ),
    )


class CurrentValueExportParams(WindowModeMixin):
    """Params for exporting the current value of an output."""

    value: ValueParams = pydantic.Field(
        default_factory=ValueParams, title='Value', description='Value to export.'
    )
    file: FileParams = pydantic.Field(
        default_factory=FileParams, title='File', description='File options.'
    )

    def windowing(self) -> Windowing:
        return (
            'since_start'
            if self.value.value is ExportedValue.since_start
            else 'per_update'
        )

    @classmethod
    def hidden_fields(cls, options: frozenset[Windowing]) -> frozenset[str]:
        """Hide the choice unless the output offers both values."""
        return frozenset() if len(options) > 1 else frozenset({'value'})


class HistoryExportParams(pydantic.BaseModel):
    """Params for exporting the buffered history of a time-series output."""

    file: FileParams = pydantic.Field(
        default_factory=FileParams, title='File', description='File options.'
    )


type ExportParams = CurrentValueExportParams | HistoryExportParams


@dataclass(frozen=True)
class ExportRequest:
    """What to export: data sources keyed by data role, and how."""

    data_sources: Mapping[str, DataSourceConfig]
    params: ExportParams


@dataclass(frozen=True)
class ExportItem:
    """One exported array and the key it was read from."""

    key: DataKey
    data: sc.DataArray


class ExportError(Exception):
    """The request cannot be served from the data DataService holds."""


class _BufferedHistory(UpdateExtractor):
    """Extracts the buffer as stored, unlike the display-oriented extractors.

    :class:`FullHistoryExtractor` reduces per-update ``start_time`` to a scalar
    and shifts ``time`` into the local timezone, both for plots.
    """

    def get_required_timespan(self) -> float:
        return float('inf')

    def extract(self, data: sc.DataArray) -> sc.DataArray:
        return data


class TimeseriesRetention(DataServiceSubscriber[DataKey]):
    """Keeps the buffered history of the given keys, without consuming it.

    Registered once at startup for :func:`timeseries_keys`, so that a time
    series can be exported (or correlated) for as far back as its buffer
    reaches, not only from when a plot of it was opened.

    A buffer still starts over when its source's timestamps go backwards (see
    :class:`TemporalBuffer`), discarding what it held.
    """

    def __init__(self, keys: Iterable[DataKey]) -> None:
        self._extractors = {key: FullHistoryExtractor() for key in keys}
        super().__init__()

    @property
    def extractors(self) -> Mapping[DataKey, UpdateExtractor]:
        return self._extractors

    def on_updated(self, updated_keys: set[DataKey]) -> None:
        """Nothing to do: this subscriber exists for its retention alone."""


def timeseries_keys(registry: Mapping[WorkflowId, WorkflowSpec]) -> list[DataKey]:
    """Return the keys of all 0-D time-series outputs.

    Each view is keyed at its per-update field, which is the field correlation
    axes and history exports read.
    """
    return [
        DataKey(
            workflow_id=workflow_id,
            source_name=source_name,
            output_name=registry[workflow_id].field_for(view_name, 'per_update'),
        )
        for workflow_id, source_name, view_name in find_timeseries_outputs(registry)
    ]


def collect_export(
    request: ExportRequest,
    data_service: DataService[DataKey, sc.DataArray],
    registry: Mapping[WorkflowId, WorkflowSpec],
) -> dict[str, ExportItem]:
    """Read the requested data from ``data_service``.

    Parameters
    ----------
    request:
        The data sources and params to export.
    data_service:
        Where the data is read from. Reading does not register a subscriber, so
        it does not change what is buffered.
    registry:
        Resolves the requested output views to the keys they are stored under.

    Returns
    -------
    :
        Exported items keyed by a name unique within the export: the source
        name, extended by the output name where a source appears twice.

    Raises
    ------
    ExportError
        If an axis is also exported as data or as the other axis, if none of the
        requested sources or one of the correlation axes has data, or if no
        point of the data lies within the time range of the axes.
    """
    resolved = resolve_data_sources(request.data_sources, request.params, registry)
    keys_by_role, _ = build_subscription_keys(resolved)
    keys = [key for role_keys in keys_by_role.values() for key in role_keys]
    if len(set(keys)) < len(keys):
        raise ExportError(
            'A correlation axis must differ from the data and the other axis.'
        )
    extractor = (
        _BufferedHistory
        if isinstance(request.params, HistoryExportParams)
        else LatestValueExtractor
    )
    extractors = {key: extractor() for key in keys}
    subscriber = DataSubscriber(keys_by_role, extractors, on_update=lambda: None)
    snapshot = data_service.snapshot(subscriber)

    names = _unique_names(keys)
    by_role = {
        role: {key: snapshot[key] for key in role_keys if key in snapshot}
        for role, role_keys in keys_by_role.items()
    }
    primary = by_role.pop(PRIMARY)
    if not primary:
        raise ExportError('No data is buffered for the selected sources.')
    axes: dict[DataKey, sc.DataArray] = {}
    for role, role_data in by_role.items():
        if not role_data:
            (axis_key,) = keys_by_role[role]
            raise ExportError(
                f'No data is buffered for correlation axis {axis_key.source_name}.'
            )
        axes.update(role_data)
    if axes:
        primary = correlate(primary, {names[key]: da for key, da in axes.items()})
        if not primary:
            raise ExportError(
                'No data point lies after the first reading of every axis.'
            )
    return {
        names[key]: ExportItem(key=key, data=_with_datetimes(da))
        for key, da in {**primary, **axes}.items()
    }


def write_export(
    items: Mapping[str, ExportItem], *, file_format: ExportFormat, title: str
) -> bytes:
    """Write exported items to a file of the given format.

    Parameters
    ----------
    items:
        Exported items keyed by name, as returned by :func:`collect_export`.
    file_format:
        Format of the file.
    title:
        Title recorded in the file.

    Returns
    -------
    :
        The content of the file.
    """
    match file_format:
        case ExportFormat.nexus:
            return write_nexus(
                {name: item.data for name, item in items.items()},
                title=title,
                group_attrs={
                    name: {
                        'workflow': str(item.key.workflow_id),
                        'source_name': item.key.source_name,
                        'output_name': item.key.output_name,
                    }
                    for name, item in items.items()
                },
            )


def export_filename(
    *,
    instrument: str,
    source_titles: list[str],
    output_title: str,
    file_format: ExportFormat,
    created: datetime,
) -> str:
    """Return a descriptive filename for an export.

    Unlike plot screenshots, exports carry a timestamp: the same output is
    typically exported once per scan, and the names must not collide. It is in
    UTC, like the times in the file.
    """
    stem = build_save_filename(instrument, source_titles, [output_title])
    stamp = created.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')
    return f'{stem}_{stamp}.{_EXTENSIONS[file_format]}'


def _unique_names(keys: Iterable[DataKey]) -> dict[DataKey, str]:
    """Name each key after its source, adding the output where that is ambiguous.

    Names must be valid HDF5 names, so ``/`` is replaced.
    """
    keys = list(dict.fromkeys(keys))
    sources = [_hdf5_name(key.source_name) for key in keys]
    return {
        key: source
        if sources.count(source) == 1
        else f'{source}_{_hdf5_name(key.output_name)}'
        for key, source in zip(keys, sources, strict=True)
    }


def _hdf5_name(name: str) -> str:
    return name.replace('/', '_')


def _with_datetimes(da: sc.DataArray) -> sc.DataArray:
    """Convert integer time coords (time since the Unix epoch) to datetime64."""
    return da.assign_coords(
        {
            name: sc.epoch(unit=coord.unit) + coord
            for name in _TIME_COORDS
            if (coord := da.coords.get(name)) is not None
            and coord.dtype == sc.DType.int64
        }
    )
