# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Plotter rendering 0D scalar data as a table."""

from __future__ import annotations

from typing import Any, ClassVar

import holoviews as hv
import numpy as np
import param
import scipp as sc
from bokeh.models import NumberFormatter, ScientificFormatter
from holoviews.core.util import dimension_sanitizer
from holoviews.plotting.bokeh.tabular import TablePlot

from ess.livedata.config.workflow_spec import ResultKey

from .data_roles import PRIMARY
from .plot_params import PlotParamsTable, TableNotation
from .plots import (
    Plotter,
    TitleResolver,
    _compute_time_bounds,
    _normalize_to_rate,
)
from .range_hook import Axis


class _FormattedTablePlot(TablePlot):
    """HoloViews bokeh ``Table`` plot whose value columns carry a custom formatter.

    HoloViews hardcodes the numeric column formatter and rebuilds every column on
    each streaming update (``update_frame``) without re-running ``hooks``, so a
    post-render hook is reverted on the first data update. Overriding column
    construction makes the formatter persist across updates.
    """

    value_formatter = param.Callable(
        default=None,
        doc="Factory returning a fresh Bokeh formatter for each value column.",
    )

    def _get_columns(self, element, data):
        columns = super()._get_columns(element, data)
        if self.value_formatter is None or not element.kdims:
            return columns
        index_field = dimension_sanitizer(element.kdims[0].name)
        for column in columns:
            if column.field != index_field:
                column.formatter = self.value_formatter()
        return columns


hv.Store.register({hv.Table: _FormattedTablePlot}, 'bokeh')


class TablePlotter(Plotter):
    """Plotter rendering 0D scalar data as a table.

    Each source name becomes a row; each output name becomes a value column.
    Unlike most plotters this composes all datasets into a single ``hv.Table``
    rather than overlaying per-dataset elements: HoloViews tables cannot be
    meaningfully overlaid (an overlay renders as separate stacked widgets), so
    multiple columns are produced within one table instead.
    """

    AUTOSCALE_AXES: ClassVar[frozenset[Axis]] = frozenset()

    @property
    def is_overlayable(self) -> bool:
        """Tables render as DataTable widgets that cannot be overlaid."""
        return False

    def __init__(
        self,
        *,
        notation: TableNotation = TableNotation.auto,
        precision: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._notation = notation
        self._precision = precision

    @classmethod
    def from_params(cls, params: PlotParamsTable):
        """Create TablePlotter from PlotParamsTable."""
        return cls(
            aspect_params=params.plot_aspect,
            normalize_to_rate=params.rate.normalize_to_rate,
            notation=params.format.notation,
            precision=params.format.precision,
        )

    def compute(
        self,
        data: dict[str, dict[ResultKey, sc.DataArray]],
        *,
        title_resolver: TitleResolver | None = None,
        **kwargs,
    ) -> None:
        primary = data.get(PRIMARY, {})
        if self._normalize_to_rate:
            primary = {key: _normalize_to_rate(da) for key, da in primary.items()}
        self._range_targets = {}
        resolver = title_resolver or TitleResolver()
        try:
            result = self._build_table(primary, resolver)
        except Exception as e:
            result = hv.Text(0.5, 0.5, f"Error: {e}").opts(
                text_align='center', text_baseline='middle', **self._sizing_opts
            )
        self._time_bounds = _compute_time_bounds(primary)
        self._set_cached_state(result)

    def _build_table(
        self,
        data: dict[ResultKey, sc.DataArray],
        resolver: TitleResolver,
    ) -> hv.Element:
        """Pivot ``(source, output) -> value`` into a single table element."""
        if len(data) == 0:
            return hv.Text(0.5, 0.5, 'No data').opts(
                text_align='center', text_baseline='middle', **self._sizing_opts
            )

        rows: list[str] = []
        columns: dict[str, hv.Dimension] = {}
        cells: dict[tuple[str, str], float] = {}
        for data_key, da in data.items():
            if da.ndim != 0:
                raise ValueError(f"Expected 0D data, got {da.ndim}D")
            row = resolver.source(data_key.job_id.source_name)
            if row not in rows:
                rows.append(row)
            output_name = data_key.output_name or 'values'
            if output_name not in columns:
                unit = str(da.unit) if da.unit is not None else None
                label = resolver.get_axis_label(data_key.output_name) or output_name
                columns[output_name] = hv.Dimension(output_name, label=label, unit=unit)
            value = float(da.value)
            if self._notation is TableNotation.decimal and self._precision < 0:
                # Negative precision rounds the magnitude (e.g. -3 -> thousands);
                # numbro format strings cannot express this, so round the value.
                value = float(np.round(value, self._precision))
            cells[(row, output_name)] = value

        table_data: dict[str, list[Any]] = {'source': rows}
        for output_name in columns:
            table_data[output_name] = [
                cells.get((row, output_name), float('nan')) for row in rows
            ]
        return hv.Table(
            table_data,
            kdims=[hv.Dimension('source', label='Source')],
            vdims=list(columns.values()),
        )

    def _value_formatter(self):
        """Bokeh column formatter for value columns, per configured notation.

        Right-aligned so consistent precision lines up the decimal point (Bokeh
        ``DataTable`` has no decimal-separator alignment). NaN/missing cells
        render as ``-`` (the formatter default). Negative precision is a
        magnitude-rounding hint for decimal notation only (applied to the data,
        see :meth:`_build_table`); here it floors to zero decimal places.
        """
        decimals = max(self._precision, 0)
        fixed_fmt = '0.' + '0' * decimals if decimals else '0'
        if self._notation is TableNotation.decimal:
            return NumberFormatter(format=fixed_fmt, text_align='right')
        if self._notation is TableNotation.compact:
            # numbro 'a' abbreviates with lowercase k/m/b/t suffixes.
            return NumberFormatter(format=fixed_fmt + 'a', text_align='right')
        if self._notation is TableNotation.scientific:
            # Empty fixed-point window (limits equal) forces scientific notation
            # for every magnitude.
            return ScientificFormatter(
                precision=decimals,
                power_limit_low=0,
                power_limit_high=0,
                text_align='right',
            )
        return ScientificFormatter(precision=decimals, text_align='right')

    def style_opts(self) -> list[hv.Options]:
        # HoloViews' Table plot exposes only fixed width/height (no responsive
        # sizing); container sizing is left to the enclosing Panel pane.
        # ``value_formatter`` is a plot option of _FormattedTablePlot; it must
        # persist across streaming updates, which a post-render hook cannot.
        return [
            hv.opts.Table(index_position=None, value_formatter=self._value_formatter),
            *super().style_opts(),
        ]
