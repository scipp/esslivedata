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
from bokeh.models import HoverTool

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard.flatten_plotter import (
    FlattenAxisConfig,
    FlattenParams,
    FlattenPlotter,
    make_flatten_params,
)
from ess.livedata.dashboard.plot_params import PlotScale


class _FakeToolbar:
    def __init__(self) -> None:
        # Mimic the default HoloViews-added HoverTool that the hook should drop.
        self.tools: list = [HoverTool()]


class _FakeFigure:
    def __init__(self) -> None:
        self.toolbar = _FakeToolbar()
        self.added_tools: list = []

    def add_tools(self, tool) -> None:
        self.added_tools.append(tool)
        self.toolbar.tools.append(tool)


class _FakePlot:
    def __init__(self) -> None:
        self.handles: dict = {'plot': _FakeFigure()}


def _run_hook(img: hv.Image) -> _FakeFigure:
    """Invoke the plotter's hook against a fake bokeh figure and return it."""
    [hook] = img.opts.get('plot').kwargs['hooks']
    plot = _FakePlot()
    hook(plot, None)
    return plot.handles['plot']


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


class TestFlattenPlotterHover:
    """Hover decomposes the flat axis into outer/inner labels."""

    def test_hook_replaces_default_hover_with_custom_one(
        self, data_abc, data_key
    ) -> None:
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        fig = _run_hook(img)
        hovers = [t for t in fig.toolbar.tools if isinstance(t, HoverTool)]
        # Exactly one hover survives — the custom flatten one.
        assert len(hovers) == 1
        assert hovers[0] in fig.added_tools

    def test_hover_tooltips_label_keep_outer_inner_and_value(
        self, data_abc, data_key
    ) -> None:
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        fig = _run_hook(img)
        [hover] = [t for t in fig.toolbar.tools if isinstance(t, HoverTool)]
        labels = [name for name, _ in hover.tooltips]
        # Keep dim is 'b' (unit 's'); natural-order outer='a', inner='c'.
        assert labels == ['b [s]', 'a', 'c', 'value']
        # Flat axis is y when not transposed; outer/inner both reference it.
        templates = dict(hover.tooltips)
        assert templates['b [s]'] == '$x'
        assert templates['a'] == '$y{outer}'
        assert templates['c'] == '$y{inner}'
        assert templates['value'] == '@image'
        assert '$y' in hover.formatters

    def test_hover_kept_dim_label_omits_unit_when_coord_missing(
        self, data_abc, data_key
    ) -> None:
        data = data_abc.copy(deep=False)
        del data.coords['b']
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data, data_key)
        fig = _run_hook(img)
        [hover] = [t for t in fig.toolbar.tools if isinstance(t, HoverTool)]
        labels = [name for name, _ in hover.tooltips]
        assert labels[0] == 'b'

    def test_hover_swaps_axes_when_transposed(self, data_abc, data_key) -> None:
        params = _make_params(('a', 'b', 'c'), keep_dim='b', transpose=True)
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        fig = _run_hook(img)
        [hover] = [t for t in fig.toolbar.tools if isinstance(t, HoverTool)]
        templates = dict(hover.tooltips)
        # Transposed: kept on y, flat on x. 'b' has unit 's'.
        assert templates['b [s]'] == '$y'
        assert templates['a'] == '$x{outer}'
        assert templates['c'] == '$x{inner}'
        assert '$x' in hover.formatters

    def test_hover_formatter_carries_outer_and_inner_coord_values(
        self, data_abc, data_key
    ) -> None:
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        fig = _run_hook(img)
        [hover] = [t for t in fig.toolbar.tools if isinstance(t, HoverTool)]
        fmt = hover.formatters['$y']
        assert fmt.args['inner_size'] == 5
        assert len(fmt.args['outer_values']) == 3  # a
        assert len(fmt.args['inner_values']) == 5  # c

    @pytest.mark.parametrize(
        ('outer_unit', 'inner_unit', 'expected_outer', 'expected_inner'),
        [
            ('m', 'K', '%s m', '%s K'),
            ('m', None, '%s m', '%s'),
            (None, 'K', '%s', '%s K'),
            (None, None, '%s', '%s'),
        ],
        ids=['both_units', 'outer_only', 'inner_only', 'no_units'],
    )
    def test_hover_formatter_template_reflects_coord_unit(
        self,
        data_key,
        outer_unit: str | None,
        inner_unit: str | None,
        expected_outer: str,
        expected_inner: str,
    ) -> None:
        da = sc.DataArray(
            sc.arange('z', 0, 60, dtype='float64').fold(
                dim='z', sizes={'a': 3, 'b': 4, 'c': 5}
            ),
            coords={
                'a': sc.array(dims=['a'], values=[0.0, 0.5, 1.0], unit=outer_unit),
                'b': sc.linspace('b', 0.0, 1.0, num=4, unit='s'),
                'c': sc.array(
                    dims=['c'], values=[0.0, 0.25, 0.5, 0.75, 1.0], unit=inner_unit
                ),
            },
        )
        da.data.unit = 'counts'
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(da, data_key)
        fig = _run_hook(img)
        [hover] = [t for t in fig.toolbar.tools if isinstance(t, HoverTool)]
        fmt = hover.formatters['$y']
        assert fmt.args['outer_template'] == expected_outer
        assert fmt.args['inner_template'] == expected_inner

    def test_hover_falls_back_to_index_when_coord_missing(
        self, data_abc, data_key
    ) -> None:
        data = data_abc.copy(deep=False)
        del data.coords['c']
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data, data_key)
        fig = _run_hook(img)
        [hover] = [t for t in fig.toolbar.tools if isinstance(t, HoverTool)]
        fmt = hover.formatters['$y']
        # Inner ('c') has no coord → empty values, bare-index template.
        assert fmt.args['inner_values'] == []
        assert fmt.args['inner_template'] == '%d'
        # Outer ('a') still carries real coord values.
        assert len(fmt.args['outer_values']) == 3

    def test_hook_is_idempotent(self, data_abc, data_key) -> None:
        # HoloViews may invoke hooks on every re-render; the hover must not
        # be installed multiple times.
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        [hook] = img.opts.get('plot').kwargs['hooks']
        plot = _FakePlot()
        hook(plot, None)
        hook(plot, None)
        fig = plot.handles['plot']
        assert len(fig.added_tools) == 1

    def test_flat_axis_label_is_not_suppressed(self, data_abc, data_key) -> None:
        # The synthetic flat-dim name (e.g. "a·c") flows through as the default
        # HoloViews axis label — no xlabel/ylabel opt overriding it.
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        plot_kwargs = img.opts.get('plot').kwargs
        assert 'xlabel' not in plot_kwargs
        assert 'ylabel' not in plot_kwargs


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
