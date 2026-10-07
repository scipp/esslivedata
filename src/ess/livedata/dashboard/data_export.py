# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Export of buffered dashboard data to files.

An export can contain only what :class:`DataService` holds; the dashboard keeps
no other record. Two kinds of export exist, chosen by the type of the params:

- :class:`CurrentValueExportParams`: the current value of an output, with the
  same window modes as plots (since start, latest update, or updates over a
  window aggregated). Aggregating needs the window's updates to be buffered.
  0-D time series always are; other outputs only while a plot in window mode
  asks for them, and then only for the plot's window. A window reaching
  further back than the buffer is refused rather than served partially.
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
from .plot_params import TimeWindowMixin
from .plotting_controller import create_extractors_from_params
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


class CurrentValueExportParams(TimeWindowMixin):
    """Params for exporting the current value of an output."""

    file: FileParams = pydantic.Field(
        default_factory=FileParams, title='File', description='File options.'
    )


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
        requested sources or one of the correlation axes has data, if the buffer
        of a source does not reach back over the requested window, or if no
        point of the data lies within the time range of the axes.
    """
    params = request.params
    resolved = resolve_data_sources(request.data_sources, params, registry)
    keys_by_role, temporality = build_subscription_keys(resolved)
    keys = [key for role_keys in keys_by_role.values() for key in role_keys]
    if len(set(keys)) < len(keys):
        raise ExportError(
            'A correlation axis must differ from the data and the other axis.'
        )
    aggregating: Mapping[DataKey, UpdateExtractor] = {}
    if (
        isinstance(params, CurrentValueExportParams)
        and params.time_window.aggregates_updates()
    ):
        try:
            aggregating = create_extractors_from_params(
                keys, params.time_window, temporality
            )
        except ValueError as error:  # Aggregation over a cumulative field.
            raise ExportError(str(error)) from None
    extractors = {
        key: LatestValueExtractor()
        if isinstance(params, CurrentValueExportParams) and not aggregating
        else _BufferedHistory()
        for key in keys
    }
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
    if aggregating:
        window = params.time_window.window_duration_seconds
        for key, history in primary.items():
            _check_covers(history, window, key.source_name)
        primary = {key: aggregating[key].extract(da) for key, da in primary.items()}
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


def _check_covers(history: sc.DataArray, seconds: float, source_name: str) -> None:
    """Raise unless the buffered updates reach back ``seconds`` from the latest.

    An update covers ``[start_time, time]``. Without a ``start_time``, the
    oldest update is taken to cover one median interval between updates. Half
    such an interval is tolerated, as the aggregating extractor tolerates it
    when selecting the updates in the window.
    """
    if 'time' not in history.dims:
        raise ExportError(
            f'Only the latest update is buffered for {source_name}. Open a plot '
            f'of it in window mode with a window of at least {seconds:g} s, or '
            'export the latest update.'
        )
    time = history.coords['time']
    interval = (
        sc.median(time[1:] - time[:-1])
        if len(time) > 1
        else sc.scalar(0, unit=time.unit)
    )
    start_time = history.coords.get('start_time')
    if start_time is None:
        oldest = time[0] - interval
    else:
        # 0-D when every update reported the same start, e.g. a run start.
        oldest = start_time[0] if start_time.ndim else start_time
    covered = (time[-1] - oldest + 0.5 * interval).to(unit='s', dtype='float64')
    if covered.value < seconds:
        raise ExportError(
            f'Only {covered.value:.0f} s are buffered for {source_name}, less than '
            f'the {seconds:g} s window. Open a plot of it in window mode with a '
            'window of at least this length, or choose a shorter window.'
        )


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
