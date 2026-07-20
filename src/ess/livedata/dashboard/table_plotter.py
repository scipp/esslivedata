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
from holoviews.plotting.bokeh.tabular import TablePlot

from ess.livedata.config.workflow_spec import DataKey

from .plot_params import PlotParamsTable, TableNotation
from .plots import Plotter, TitleResolver
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
        if self.value_formatter is None:
            return columns
        # Only numeric value columns get the formatter; string columns (source
        # index, per-row unit) keep HoloViews' default StringFormatter.
        for column in columns:
            if np.asarray(data[column.field]).dtype.kind in 'iuf':
                column.formatter = self.value_formatter()
        return columns


hv.Store.register({hv.Table: _FormattedTablePlot}, 'bokeh')


class TablePlotter(Plotter):
    """Plotter rendering 0D scalar data as a table.

    Each source name becomes a row, with a single value column. Unlike most
    plotters this composes all datasets into a single ``hv.Table`` rather than
    overlaying per-dataset elements: HoloViews tables cannot be meaningfully
    overlaid (an overlay renders as separate stacked widgets).
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

    def _build_result(
        self,
        data: dict[DataKey, sc.DataArray],
        resolver: TitleResolver,
        **kwargs,
    ) -> hv.Element:
        """Build a table: one row per source, sharing a value column.

        All datasets reaching a single layer carry the same ``output_name`` (the
        layer's primary view), so the table has a single value column. Rows may
        nonetheless carry different units (e.g. device data forwarded by the
        ``timeseries`` service: a motor angle and a temperature share an output
        but not a unit). With a single shared unit it goes in the value-column
        header; with mixed units a dedicated ``Unit`` column is added instead.
        """
        if len(data) == 0:
            return hv.Text(0.5, 0.5, 'No data').opts(
                text_align='center', text_baseline='middle', **self._sizing_opts
            )

        rows: list[str] = []
        values: list[float] = []
        units: list[str] = []
        output_name = 'values'
        label: str | None = None
        for data_key, da in data.items():
            if da.ndim != 0:
                raise ValueError(f"Expected 0D data, got {da.ndim}D")
            output_name = data_key.output_name or 'values'
            label = resolver.get_axis_label(data_key.output_name) or output_name
            rows.append(resolver.source(data_key.source_name))
            units.append('' if da.unit is None else str(da.unit))
            value = float(da.value)
            if self._notation is TableNotation.decimal and self._precision < 0:
                # Negative precision rounds the magnitude (e.g. -3 -> thousands);
                # numbro format strings cannot express this, so round the value.
                value = float(np.round(value, self._precision))
            values.append(value)

        kdims = [hv.Dimension('source', label='Source')]
        if len(set(units)) > 1:
            value_dim = hv.Dimension(output_name, label=label)
            unit_dim = hv.Dimension('unit', label='Unit')
            return hv.Table(
                {'source': rows, value_dim.name: values, unit_dim.name: units},
                kdims=kdims,
                vdims=[value_dim, unit_dim],
            )

        unit = next(iter(units)) or None
        value_dim = hv.Dimension(output_name, label=label, unit=unit)
        return hv.Table(
            {'source': rows, value_dim.name: values},
            kdims=kdims,
            vdims=[value_dim],
        )

    def _value_formatter(self):
        """Bokeh column formatter for value columns, per configured notation.

        Right-aligned so consistent precision lines up the decimal point (Bokeh
        ``DataTable`` has no decimal-separator alignment). NaN values render as
        ``-`` (the formatter default). Negative precision is a
        magnitude-rounding hint for decimal notation only (applied to the data,
        see :meth:`_build_result`); here it floors to zero decimal places.
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
        # auto: ScientificFormatter's default power limits show decimal within
        # 1e-3..1e5 and switch to scientific for magnitudes outside that window.
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
