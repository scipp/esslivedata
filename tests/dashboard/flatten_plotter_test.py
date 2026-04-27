# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for FlattenPlotter and the dynamic param model."""

from __future__ import annotations

import uuid
from enum import IntEnum

import holoviews as hv
import pydantic
import pytest
import scipp as sc
from bokeh.models import CustomJSTickFormatter, LinearAxis

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard.flatten_plotter import (
    FlattenAxisConfig,
    FlattenParams,
    FlattenPlotter,
    make_flatten_params,
)
from ess.livedata.dashboard.plot_params import PlotScale


class _FakeFigure:
    def __init__(self) -> None:
        self.layouts: list[tuple[LinearAxis, str]] = []

    def add_layout(self, axis: LinearAxis, side: str) -> None:
        self.layouts.append((axis, side))


class _FakePlot:
    def __init__(self) -> None:
        self.handles: dict = {'plot': _FakeFigure()}


def _run_inner_axis_hook(img: hv.Image) -> _FakeFigure:
    """Invoke the secondary-axis hook against a fake bokeh figure and return it."""
    [hook] = img.opts.get('plot').kwargs['hooks']
    plot = _FakePlot()
    hook(plot, None)
    return plot.handles['plot']


def _outer_ticks_opt(img: hv.Image, transpose: bool) -> list[tuple[float, str]]:
    """Outer-axis ticks set on the primary (HoloViews xticks/yticks opt)."""
    key = 'xticks' if transpose else 'yticks'
    return img.opts.get('plot').kwargs[key]


hv.extension('bokeh')


@pytest.fixture
def data_key() -> ResultKey:
    return ResultKey(
        workflow_id=WorkflowId(instrument='inst', namespace='ns', name='wf', version=1),
        job_id=JobId(source_name='src', job_number=uuid.uuid4()),
        output_name='out',
    )


@pytest.fixture
def data_abc() -> sc.DataArray:
    """3D data with dims (a, b, c) and 1-D coords on each."""
    da = sc.DataArray(
        sc.arange('z', 0, 60, dtype='float64').fold(
            dim='z', sizes={'a': 3, 'b': 4, 'c': 5}
        ),
        coords={
            'a': sc.linspace('a', 0.0, 1.0, num=3, unit='m'),
            'b': sc.linspace('b', 0.0, 1.0, num=4, unit='s'),
            'c': sc.linspace('c', 0.0, 1.0, num=5, unit='K'),
        },
    )
    da.data.unit = 'counts'
    return da


def _make_params(
    dims: tuple[str, ...],
    *,
    keep_dim: str | int,
    flatten_transposed: bool = False,
    transpose: bool = False,
) -> FlattenParams:
    """Build params, accepting ``keep_dim`` as a name (resolved to position) or int."""
    Cls = make_flatten_params(dims)
    keep_dim_index = dims.index(keep_dim) if isinstance(keep_dim, str) else keep_dim
    p = Cls(
        flatten={
            'keep_dim': keep_dim_index,
            'flatten_transposed': flatten_transposed,
            'transpose': transpose,
        }
    )
    p.plot_scale.color_scale = PlotScale.linear
    return p


def _axis_cls(params_cls: type[FlattenParams]) -> type[FlattenAxisConfig]:
    return params_cls.model_fields['flatten'].annotation


class TestMakeFlattenParams:
    def test_narrows_keep_dim_to_intenum_with_dim_names_as_members(self) -> None:
        Cls = make_flatten_params(('a', 'b', 'c'))
        annotation = _axis_cls(Cls).model_fields['keep_dim'].annotation
        assert issubclass(annotation, IntEnum)
        # Member names render as dropdown labels; values are positions.
        assert {m.name: m.value for m in annotation} == {'a': 0, 'b': 1, 'c': 2}

    def test_does_not_expose_outer_dim_choice(self) -> None:
        # The flatten ordering is controlled by a single boolean checkbox so
        # the user is exposed to (potentially placeholder) dim names only via
        # the keep_dim dropdown.
        Cls = make_flatten_params(('a', 'b', 'c'))
        fields = _axis_cls(Cls).model_fields
        assert 'flatten_outer' not in fields
        assert fields['flatten_transposed'].annotation is bool

    def test_rejects_keep_dim_out_of_range(self) -> None:
        Cls = make_flatten_params(('a', 'b', 'c'))
        with pytest.raises(pydantic.ValidationError):
            Cls(flatten={'keep_dim': 99})

    def test_saved_keep_dim_round_trips_as_integer(self) -> None:
        # Integer storage is the whole point: the saved value is invariant
        # under per-instrument dim renames, and validates against the base
        # class on restore without needing template_dims.
        Cls = make_flatten_params(('a', 'b', 'c'))
        narrowed = Cls(flatten={'keep_dim': 1})
        dumped = narrowed.model_dump()
        assert dumped['flatten']['keep_dim'] == 1
        # Restore path: validate the dumped dict against the base class.
        restored = FlattenParams(**dumped)
        assert restored.flatten.keep_dim == 1

    def test_non_3d_returns_base_class_unmodified(self) -> None:
        # 2D / 4D etc. fall back to the unparameterized base; the plotter then
        # rejects mismatches at plot-time rather than at config-time.
        assert make_flatten_params(('x', 'y')) is FlattenParams
        assert make_flatten_params(('a', 'b', 'c', 'd')) is FlattenParams

    def test_renders_in_model_widget(self) -> None:
        """Every top-level field must itself be a BaseModel (one tab each).

        Regression test: the dashboard's ModelWidget builds tabs by calling
        ``model_fields`` on each field annotation. Plain Literal/bool fields
        crash this.
        """
        from ess.livedata.dashboard.widgets.model_widget import ModelWidget

        Cls = make_flatten_params(('a', 'b', 'c'))
        widget = ModelWidget(Cls)
        tab_names = {entry[0] for entry in widget.param_group_tabs}
        assert 'flatten' in tab_names


class TestFlattenPlotterPlot:
    # data_abc has dims ('a', 'b', 'c'); for keep_dim='b' the natural-order
    # flatten pair is (outer='a', inner='c'). flatten_transposed swaps that.

    def test_returns_2d_image(self, data_abc, data_key) -> None:
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        assert isinstance(img, hv.Image)

    def test_kept_dim_is_x_when_not_transposed(self, data_abc, data_key) -> None:
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        # HoloViews Image: kdims[0] is x, kdims[1] is y
        assert img.kdims[0].name == 'b'
        assert img.kdims[1].name == 'a·c'

    def test_kept_dim_is_y_when_transposed(self, data_abc, data_key) -> None:
        params = _make_params(('a', 'b', 'c'), keep_dim='b', transpose=True)
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        assert img.kdims[1].name == 'b'
        assert img.kdims[0].name == 'a·c'

    def test_flatten_transposed_swaps_outer_and_inner(self, data_abc, data_key) -> None:
        params = _make_params(('a', 'b', 'c'), keep_dim='b', flatten_transposed=True)
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        assert img.kdims[1].name == 'c·a'

    def test_kept_dim_unit_is_preserved(self, data_abc, data_key) -> None:
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        b_dim = next(d for d in img.kdims if d.name == 'b')
        assert b_dim.unit == 's'

    def test_outer_ticks_at_group_centres_on_primary_axis(
        self, data_abc, data_key
    ) -> None:
        # outer='a' size 3, inner='c' size 5 → centres at i*5 + 2 = 2, 7, 12.
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        ticks = _outer_ticks_opt(img, transpose=False)
        assert [pos for pos, _ in ticks] == [2.0, 7.0, 12.0]
        assert all(label.startswith('a=') for _, label in ticks)

    def test_inner_axis_hook_installs_secondary_with_inner_formatter(
        self, data_abc, data_key
    ) -> None:
        # keep_dim='b' → inner='c' size 5 → inner_size=5; secondary axis must
        # carry the zoom-aware inner formatter for the y axis (side='left').
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        fig = _run_inner_axis_hook(img)
        [(axis, side)] = fig.layouts
        assert side == 'left'
        assert isinstance(axis, LinearAxis)
        assert isinstance(axis.formatter, CustomJSTickFormatter)
        assert axis.formatter.args['inner_size'] == 5
        assert len(axis.formatter.args['values']) == 5
        # No axis_label on the secondary — it would float between the rows.
        assert axis.axis_label == ''

    def test_primary_axis_label_is_cleared(self, data_abc, data_key) -> None:
        # The synthetic flat-dim name (e.g. "a·c") is suppressed since each
        # tick row carries its own dim name.
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        assert img.opts.get('plot').kwargs['ylabel'] == ''

    def test_inner_axis_hook_is_idempotent(self, data_abc, data_key) -> None:
        # HoloViews may invoke hooks on every re-render; the secondary axis
        # must not accumulate.
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        [hook] = img.opts.get('plot').kwargs['hooks']
        plot = _FakePlot()
        hook(plot, None)
        hook(plot, None)
        assert len(plot.handles['plot'].layouts) == 1

    def test_transposed_uses_x_axis_and_below_side(self, data_abc, data_key) -> None:
        params = _make_params(('a', 'b', 'c'), keep_dim='b', transpose=True)
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        # Outer ticks on the primary x axis.
        assert _outer_ticks_opt(img, transpose=True)
        assert img.opts.get('plot').kwargs['xlabel'] == ''
        # Secondary axis attached to 'below'.
        fig = _run_inner_axis_hook(img)
        [(_axis, side)] = fig.layouts
        assert side == 'below'

    def test_flatten_transposed_changes_inner_size(self, data_abc, data_key) -> None:
        # outer='c' (size 5), inner='a' (size 3) → inner_size=3
        params = _make_params(('a', 'b', 'c'), keep_dim='b', flatten_transposed=True)
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        fig = _run_inner_axis_hook(img)
        [(axis, _side)] = fig.layouts
        assert axis.formatter.args['inner_size'] == 3

    def test_missing_inner_coord_falls_back_to_index_template(
        self, data_abc, data_key
    ) -> None:
        # Drop coord on the inner dim ('c' under natural order with keep='b').
        data = data_abc.copy(deep=False)
        del data.coords['c']
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data, data_key)
        fig = _run_inner_axis_hook(img)
        [(axis, _side)] = fig.layouts
        # Inner formatter falls back to "c[%d]".
        assert axis.formatter.args['values'] == []
        assert '%d' in axis.formatter.args['template']
        # Outer ('a') still has its real-coord labels on the primary axis.
        ticks = _outer_ticks_opt(img, transpose=False)
        assert all(label.startswith('a=') for _, label in ticks)

    def test_missing_outer_coord_falls_back_to_index_label(
        self, data_abc, data_key
    ) -> None:
        # Drop coord on the outer dim ('a' under natural order with keep='b').
        data = data_abc.copy(deep=False)
        del data.coords['a']
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data, data_key)
        ticks = _outer_ticks_opt(img, transpose=False)
        assert {label for _, label in ticks} == {f'a[{i}]' for i in range(3)}

    def test_bin_edge_coord_uses_midpoints(self, data_key) -> None:
        # 'a' (the outer under natural order with keep='b') has bin edges
        # (size 4 for dim size 3) → 3 outer-axis ticks for the 3 midpoints.
        data = sc.DataArray(
            sc.arange('z', 0, 60, dtype='float64').fold(
                dim='z', sizes={'a': 3, 'b': 4, 'c': 5}
            ),
            coords={
                'a': sc.linspace('a', 0.0, 1.0, num=4, unit='m'),
                'b': sc.linspace('b', 0.0, 1.0, num=4, unit='s'),
                'c': sc.linspace('c', 0.0, 1.0, num=5, unit='K'),
            },
        )
        data.data.unit = 'counts'
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data, data_key)
        ticks = _outer_ticks_opt(img, transpose=False)
        assert len(ticks) == 3

    def test_raises_for_non_3d_input(self, data_key) -> None:
        data = sc.DataArray(sc.zeros(dims=['x', 'y'], shape=[2, 2]))
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        with pytest.raises(ValueError, match='3D input'):
            plotter.plot(data, data_key)

    def test_runtime_dim_rename_is_transparent(self, data_abc, data_key) -> None:
        # Template dims (a, b, c) configured with keep_dim='b' (position 1);
        # runtime data is renamed to (a, d, c). Position 1 → 'd'. The
        # plotter never references the dim name, so renames are transparent.
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        data = data_abc.rename_dims({'b': 'd'})
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data, data_key)
        assert img.kdims[0].name == 'd'  # x = kept (position 1)
        assert img.kdims[1].name == 'a·c'

    def test_raises_for_keep_dim_out_of_range(self, data_abc, data_key) -> None:
        # Out-of-range positions are caught at plot time. Pydantic prevents
        # the narrowed model from constructing one, but the base model
        # accepts any int — and that's the path saved configs go through on
        # restore.
        params = FlattenParams(flatten={'keep_dim': 5})
        plotter = FlattenPlotter.from_params(params)
        with pytest.raises(ValueError, match='out of range'):
            plotter.plot(data_abc, data_key)


class TestFlattenPlotterCompute:
    def test_compute_caches_state(self, data_abc, data_key) -> None:
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        plotter.compute({'primary': {data_key: data_abc}})
        assert plotter.has_cached_state()

    def test_compute_renders_error_text_on_ndim_mismatch(self, data_key) -> None:
        # 2D data → ndim mismatch → base Plotter.compute() wraps the
        # ValueError in a Text element rather than crashing.
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        data_2d = sc.DataArray(sc.zeros(dims=['x', 'y'], shape=[2, 2]))
        plotter.compute({'primary': {data_key: data_2d}})
        assert plotter.get_cached_state() is not None
