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


def _make_data(sizes: dict[str, int], units: dict[str, str]) -> sc.DataArray:
    """Build a DataArray with C-order range data and 1-D coords on each dim."""
    n = 1
    for s in sizes.values():
        n *= s
    da = sc.DataArray(
        sc.arange('z', 0, n, dtype='float64').fold(dim='z', sizes=sizes),
        coords={
            d: sc.linspace(d, 0.0, 1.0, num=size, unit=units[d])
            for d, size in sizes.items()
        },
    )
    da.data.unit = 'counts'
    return da


@pytest.fixture
def data_ab() -> sc.DataArray:
    return _make_data({'a': 3, 'b': 4}, {'a': 'm', 'b': 's'})


@pytest.fixture
def data_abc() -> sc.DataArray:
    return _make_data({'a': 3, 'b': 4, 'c': 5}, {'a': 'm', 'b': 's', 'c': 'K'})


@pytest.fixture
def data_abcd() -> sc.DataArray:
    return _make_data(
        {'a': 2, 'b': 3, 'c': 4, 'd': 5},
        {'a': 'm', 'b': 's', 'c': 'K', 'd': 'J'},
    )


@pytest.fixture
def data_abcde() -> sc.DataArray:
    return _make_data(
        {'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6},
        {'a': 'm', 'b': 's', 'c': 'K', 'd': 'J', 'e': 'A'},
    )


def _make_params(
    dims: tuple[str, ...],
    *,
    axis_x: tuple[str | int, ...] | str | int,
    transpose_x_flatten: bool = False,
    transpose_y_flatten: bool = False,
    transpose: bool = False,
) -> FlattenParams:
    """Build params, accepting ``axis_x`` as dim name(s) or position(s)."""
    Cls = make_flatten_params(dims)
    if isinstance(axis_x, (str, int)):
        axis_x = (axis_x,)
    positions = {dims.index(d) if isinstance(d, str) else d for d in axis_x}
    p = Cls(
        flatten={
            'axis_x_dims': positions,
            'transpose_x_flatten': transpose_x_flatten,
            'transpose_y_flatten': transpose_y_flatten,
            'transpose': transpose,
        }
    )
    p.plot_scale.color_scale = PlotScale.linear
    return p


def _axis_cls(params_cls: type[FlattenParams]) -> type[FlattenAxisConfig]:
    return params_cls.model_fields['flatten'].annotation


_ARITY_PARAMS = pytest.mark.parametrize(
    'dims',
    [
        ('a', 'b'),
        ('a', 'b', 'c'),
        ('a', 'b', 'c', 'd'),
        ('a', 'b', 'c', 'd', 'e'),
    ],
    ids=['2d', '3d', '4d', '5d'],
)


class TestMakeFlattenParams:
    def test_narrows_axis_x_dims_to_set_of_intenum_with_dim_names(self) -> None:
        Cls = make_flatten_params(('a', 'b', 'c'))
        annotation = _axis_cls(Cls).model_fields['axis_x_dims'].annotation
        # set[Dim] — extract the inner Enum and verify member mapping.
        from typing import get_args

        (inner,) = get_args(annotation)
        assert issubclass(inner, IntEnum)
        assert {m.name: m.value for m in inner} == {'a': 0, 'b': 1, 'c': 2}

    def test_axis_config_exposes_partition_and_per_axis_flags(self) -> None:
        Cls = make_flatten_params(('a', 'b', 'c'))
        fields = _axis_cls(Cls).model_fields
        assert 'axis_x_dims' in fields
        assert fields['transpose_x_flatten'].annotation is bool
        assert fields['transpose_y_flatten'].annotation is bool
        assert fields['transpose'].annotation is bool

    @_ARITY_PARAMS
    def test_rejects_empty_axis_x_dims(self, dims) -> None:
        Cls = make_flatten_params(dims)
        with pytest.raises(pydantic.ValidationError):
            Cls(flatten={'axis_x_dims': set()})

    @_ARITY_PARAMS
    def test_rejects_axis_x_dims_covering_all_dims(self, dims) -> None:
        Cls = make_flatten_params(dims)
        with pytest.raises(pydantic.ValidationError):
            Cls(flatten={'axis_x_dims': set(range(len(dims)))})

    @_ARITY_PARAMS
    def test_rejects_axis_x_dims_with_unknown_position(self, dims) -> None:
        Cls = make_flatten_params(dims)
        with pytest.raises(pydantic.ValidationError):
            Cls(flatten={'axis_x_dims': {99}})

    def test_saved_axis_x_dims_round_trips_as_integers(self) -> None:
        # Integer storage is the whole point: invariant under per-instrument
        # dim renames, and validates against the base class on restore
        # without needing template_dims.
        Cls = make_flatten_params(('a', 'b', 'c'))
        narrowed = Cls(flatten={'axis_x_dims': {1}})
        dumped = narrowed.model_dump()
        assert dumped['flatten']['axis_x_dims'] == {1}
        restored = FlattenParams(**dumped)
        assert restored.flatten.axis_x_dims == {1}

    def test_returns_base_class_for_arity_below_two(self) -> None:
        # 0D / 1D have no meaningful partition; fall back to the un-narrowed
        # base. The plotter rejects mismatches at plot time.
        assert make_flatten_params(()) is FlattenParams
        assert make_flatten_params(('x',)) is FlattenParams

    def test_narrows_for_arity_two_or_more(self) -> None:
        for dims in [('x', 'y'), ('a', 'b', 'c'), ('a', 'b', 'c', 'd')]:
            assert make_flatten_params(dims) is not FlattenParams

    @_ARITY_PARAMS
    def test_renders_in_model_widget(self, dims) -> None:
        """Every top-level field must itself be a BaseModel (one tab each).

        Regression test: the dashboard's ModelWidget builds tabs by calling
        ``model_fields`` on each field annotation. Plain Literal/bool fields
        crash this.
        """
        from ess.livedata.dashboard.widgets.model_widget import ModelWidget

        Cls = make_flatten_params(dims)
        widget = ModelWidget(Cls)
        tab_names = {entry[0] for entry in widget.param_group_tabs}
        assert 'flatten' in tab_names


class TestFlattenPlotterPlot:
    # data_abc has dims ('a', 'b', 'c'); for axis_x='b' the natural-order
    # complement on Y is (a, c). transpose_y_flatten swaps that.

    def test_returns_2d_image(self, data_abc, data_key) -> None:
        params = _make_params(('a', 'b', 'c'), axis_x='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        assert isinstance(img, hv.Image)

    # HoloViews Image: kdims[0] is x, kdims[1] is y. The flat axis name uses
    # tuple notation for the dims that feed the flatten;
    # the global ``transpose`` swaps the resulting X and Y wholesale.
    @pytest.mark.parametrize(
        ('data_fixture', 'axis_x', 'flags', 'expected_x', 'expected_y'),
        [
            # 2D: degenerate, no flatten on either side.
            ('data_ab', 'b', {}, 'b', 'a'),
            # 3D: basic, transpose, per-axis flatten reversal.
            ('data_abc', 'b', {}, 'b', '(a,c)'),
            ('data_abc', 'b', {'transpose': True}, '(a,c)', 'b'),
            ('data_abc', 'b', {'transpose_y_flatten': True}, 'b', '(c,a)'),
            ('data_abc', ('a', 'c'), {'transpose_x_flatten': True}, '(c,a)', 'b'),
            # 4D: 2-2, 3-1, 1-3 partitions; K=3 reversal on each side; transpose.
            ('data_abcd', ('a', 'b'), {}, '(a,b)', '(c,d)'),
            ('data_abcd', ('a', 'b', 'c'), {}, '(a,b,c)', 'd'),
            ('data_abcd', 'a', {}, 'a', '(b,c,d)'),
            (
                'data_abcd',
                ('a', 'b', 'c'),
                {'transpose_x_flatten': True},
                '(c,b,a)',
                'd',
            ),
            ('data_abcd', 'a', {'transpose_y_flatten': True}, 'a', '(d,c,b)'),
            ('data_abcd', ('a', 'b'), {'transpose': True}, '(c,d)', '(a,b)'),
            # 5D: smoke for scaling beyond 4D.
            ('data_abcde', ('a', 'b'), {}, '(a,b)', '(c,d,e)'),
        ],
        ids=[
            '2d',
            '3d_basic',
            '3d_transpose',
            '3d_y_reversed',
            '3d_x_reversed',
            '4d_2_2',
            '4d_3_1',
            '4d_1_3',
            '4d_x_k3_reversed',
            '4d_y_k3_reversed',
            '4d_transpose',
            '5d_2_3',
        ],
    )
    def test_kdims_match_partition_and_flags(
        self,
        request,
        data_key,
        data_fixture,
        axis_x,
        flags,
        expected_x,
        expected_y,
    ) -> None:
        data = request.getfixturevalue(data_fixture)
        params = _make_params(data.dims, axis_x=axis_x, **flags)
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data, data_key)
        assert img.kdims[0].name == expected_x
        assert img.kdims[1].name == expected_y

    def test_axis_x_dim_unit_is_preserved(self, data_abc, data_key) -> None:
        params = _make_params(('a', 'b', 'c'), axis_x='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        b_dim = next(d for d in img.kdims if d.name == 'b')
        assert b_dim.unit == 's'

    def test_runtime_dim_rename_is_transparent(self, data_abc, data_key) -> None:
        # Template dims (a, b, c) configured with axis_x='b' (position 1);
        # runtime data renamed to (a, d, c). Position 1 → 'd'. The plotter
        # never references the dim name, so renames are transparent.
        params = _make_params(('a', 'b', 'c'), axis_x='b')
        data = data_abc.rename_dims({'b': 'd'})
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data, data_key)
        assert img.kdims[0].name == 'd'
        assert img.kdims[1].name == '(a,c)'

    def test_raises_when_axis_x_position_out_of_range_for_data(self, data_key) -> None:
        # Configured for 3D (axis_x position 2), data is 2D → position 2
        # is out of range. Out-of-range positions are caught at plot time;
        # the un-narrowed base FlattenParams accepts any int set, which is
        # the path saved configs go through on restore.
        data_2d = sc.DataArray(sc.zeros(dims=['x', 'y'], shape=[2, 2]))
        params = FlattenParams(flatten={'axis_x_dims': {2}})
        plotter = FlattenPlotter.from_params(params)
        with pytest.raises(ValueError, match='out of range'):
            plotter.plot(data_2d, data_key)

    def test_raises_when_axis_x_covers_all_dims_at_plot_time(
        self, data_abc, data_key
    ) -> None:
        # The narrowed model rejects this at config time; the base model
        # doesn't, so restore-then-plot must catch it.
        params = FlattenParams(flatten={'axis_x_dims': {0, 1, 2}})
        plotter = FlattenPlotter.from_params(params)
        with pytest.raises(ValueError, match='Y axis would be empty'):
            plotter.plot(data_abc, data_key)


class TestFlattenPlotterDirectConstruction:
    """`__init__`-level validation independent of params."""

    def _kwargs(self, **flatten_kwargs):
        # ImagePlotter requires several styling kwargs; lift them off a
        # default FlattenParams so each test only states the flatten kwargs
        # that matter to it.
        base = FlattenParams()
        return dict(
            grow_threshold=0.1,
            layout_params=base.layout,
            aspect_params=base.plot_aspect,
            scale_opts=base.plot_scale,
            tick_params=base.ticks,
            normalize_to_rate=False,
            **flatten_kwargs,
        )

    def test_rejects_empty_axis_x_dims(self) -> None:
        with pytest.raises(ValueError, match='at least one'):
            FlattenPlotter(**self._kwargs(axis_x_dims=frozenset()))

    def test_rejects_negative_position(self) -> None:
        with pytest.raises(ValueError, match='Negative position'):
            FlattenPlotter(**self._kwargs(axis_x_dims={-1}))


class TestFlattenPlotterHover:
    """Hover decomposes each image axis into per-dim labels."""

    def test_hook_replaces_default_hover_with_custom_one(
        self, data_abc, data_key
    ) -> None:
        params = _make_params(('a', 'b', 'c'), axis_x='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        fig = _run_hook(img)
        hovers = [t for t in fig.toolbar.tools if isinstance(t, HoverTool)]
        # Exactly one hover survives — the custom flatten one.
        assert len(hovers) == 1
        assert hovers[0] in fig.added_tools

    def test_hover_tooltips_one_row_per_dim_then_value(
        self, data_abc, data_key
    ) -> None:
        params = _make_params(('a', 'b', 'c'), axis_x='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        fig = _run_hook(img)
        [hover] = [t for t in fig.toolbar.tools if isinstance(t, HoverTool)]
        # x first, y next, value last. Per-dim format directive = dim name.
        # Each dim's row label carries its unit; values come back unit-less.
        labels = [name for name, _ in hover.tooltips]
        assert labels == ['b [s]', 'a [m]', 'c [K]', 'value']
        templates = dict(hover.tooltips)
        assert templates['b [s]'] == '$x{b}'
        assert templates['a [m]'] == '$y{a}'
        assert templates['c [K]'] == '$y{c}'
        assert templates['value'] == '@image'
        # Both axes always have a formatter — single-dim axes too.
        assert '$x' in hover.formatters
        assert '$y' in hover.formatters

    def test_hover_dim_label_omits_unit_when_coord_missing(
        self, data_abc, data_key
    ) -> None:
        data = data_abc.copy(deep=False)
        del data.coords['b']
        params = _make_params(('a', 'b', 'c'), axis_x='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data, data_key)
        fig = _run_hook(img)
        [hover] = [t for t in fig.toolbar.tools if isinstance(t, HoverTool)]
        labels = [name for name, _ in hover.tooltips]
        assert labels[0] == 'b'

    def test_hover_swaps_axes_when_transposed(self, data_abc, data_key) -> None:
        params = _make_params(('a', 'b', 'c'), axis_x='b', transpose=True)
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        fig = _run_hook(img)
        [hover] = [t for t in fig.toolbar.tools if isinstance(t, HoverTool)]
        labels = [name for name, _ in hover.tooltips]
        assert labels == ['a [m]', 'c [K]', 'b [s]', 'value']
        templates = dict(hover.tooltips)
        assert templates['a [m]'] == '$x{a}'
        assert templates['c [K]'] == '$x{c}'
        assert templates['b [s]'] == '$y{b}'

    def test_hover_formatter_args_describe_axis_levels(
        self, data_abc, data_key
    ) -> None:
        params = _make_params(('a', 'b', 'c'), axis_x='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        fig = _run_hook(img)
        [hover] = [t for t in fig.toolbar.tools if isinstance(t, HoverTool)]
        # Y axis flattens (a outer, c inner); strides are C-order.
        y_fmt = hover.formatters['$y']
        assert y_fmt.args['names'] == ['a', 'c']
        assert y_fmt.args['sizes'] == [3, 5]
        assert y_fmt.args['strides'] == [5, 1]
        assert len(y_fmt.args['values_by_dim'][0]) == 3
        assert len(y_fmt.args['values_by_dim'][1]) == 5
        # X axis is single-dim ('b') — one level, stride 1.
        x_fmt = hover.formatters['$x']
        assert x_fmt.args['names'] == ['b']
        assert x_fmt.args['sizes'] == [4]
        assert x_fmt.args['strides'] == [1]

    @pytest.mark.parametrize(
        ('outer_unit', 'inner_unit', 'expected_a', 'expected_c'),
        [
            ('m', 'K', 'a [m]', 'c [K]'),
            ('m', None, 'a [m]', 'c'),
            (None, 'K', 'a', 'c [K]'),
            (None, None, 'a', 'c'),
        ],
        ids=['both_units', 'outer_only', 'inner_only', 'no_units'],
    )
    def test_hover_row_label_reflects_coord_unit(
        self,
        data_key,
        outer_unit: str | None,
        inner_unit: str | None,
        expected_a: str,
        expected_c: str,
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
        params = _make_params(('a', 'b', 'c'), axis_x='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(da, data_key)
        fig = _run_hook(img)
        [hover] = [t for t in fig.toolbar.tools if isinstance(t, HoverTool)]
        labels = [name for name, _ in hover.tooltips]
        assert expected_a in labels
        assert expected_c in labels

    def test_hover_falls_back_to_bare_index_when_coord_missing(
        self, data_abc, data_key
    ) -> None:
        data = data_abc.copy(deep=False)
        del data.coords['c']
        params = _make_params(('a', 'b', 'c'), axis_x='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data, data_key)
        fig = _run_hook(img)
        [hover] = [t for t in fig.toolbar.tools if isinstance(t, HoverTool)]
        y_fmt = hover.formatters['$y']
        # 'c' has no coord → empty values list → JS emits the bare index.
        assert y_fmt.args['values_by_dim'][1] == []
        # 'a' still carries real coord values.
        assert len(y_fmt.args['values_by_dim'][0]) == 3

    def test_hover_single_dim_axis_without_coord_emits_index(
        self, data_abc, data_key
    ) -> None:
        # With no coord, the JS returns String(Math.round(value)) — the image
        # axis uses integer indices so the cursor is already in index space.
        data = data_abc.copy(deep=False)
        del data.coords['b']
        params = _make_params(('a', 'b', 'c'), axis_x='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data, data_key)
        fig = _run_hook(img)
        [hover] = [t for t in fig.toolbar.tools if isinstance(t, HoverTool)]
        x_fmt = hover.formatters['$x']
        assert x_fmt.args['values_by_dim'] == [[]]

    def test_hover_single_dim_axis_with_coord_uses_values_lookup(
        self, data_abc, data_key
    ) -> None:
        params = _make_params(('a', 'b', 'c'), axis_x='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        fig = _run_hook(img)
        [hover] = [t for t in fig.toolbar.tools if isinstance(t, HoverTool)]
        x_fmt = hover.formatters['$x']
        # 'b' coord is present → values lookup; the row label carries the unit.
        assert len(x_fmt.args['values_by_dim'][0]) == 4

    def test_hook_is_idempotent(self, data_abc, data_key) -> None:
        # HoloViews may invoke hooks on every re-render; the hover must not
        # be installed multiple times.
        params = _make_params(('a', 'b', 'c'), axis_x='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        [hook] = img.opts.get('plot').kwargs['hooks']
        plot = _FakePlot()
        hook(plot, None)
        hook(plot, None)
        fig = plot.handles['plot']
        assert len(fig.added_tools) == 1

    def test_flat_axis_label_is_not_suppressed(self, data_abc, data_key) -> None:
        # The synthetic flat-dim name (e.g. "(a,c)") flows through as the default
        # HoloViews axis label — no xlabel/ylabel opt overriding it.
        params = _make_params(('a', 'b', 'c'), axis_x='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        plot_kwargs = img.opts.get('plot').kwargs
        assert 'xlabel' not in plot_kwargs
        assert 'ylabel' not in plot_kwargs


class TestFlattenPlotterNDimGeneralization:
    """Hover formatter args at higher arities (kdim names live in the
    parametrized ``test_kdims_match_partition_and_flags``)."""

    def test_4d_hover_has_one_row_per_input_dim(self, data_abcd, data_key) -> None:
        # 2-2 partition: tooltip rows span all four input dims and each axis
        # carries a per-dim formatter.
        params = _make_params(('a', 'b', 'c', 'd'), axis_x=('a', 'b'))
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abcd, data_key)
        fig = _run_hook(img)
        [hover] = [t for t in fig.toolbar.tools if isinstance(t, HoverTool)]
        labels = [name for name, _ in hover.tooltips]
        assert labels == ['a [m]', 'b [s]', 'c [K]', 'd [J]', 'value']
        x_fmt = hover.formatters['$x']
        y_fmt = hover.formatters['$y']
        assert x_fmt.args['names'] == ['a', 'b']
        assert x_fmt.args['sizes'] == [2, 3]
        assert x_fmt.args['strides'] == [3, 1]
        assert y_fmt.args['names'] == ['c', 'd']
        assert y_fmt.args['sizes'] == [4, 5]
        assert y_fmt.args['strides'] == [5, 1]

    @pytest.mark.parametrize(
        ('axis_x', 'flags', 'fmt_key', 'names', 'sizes', 'strides'),
        [
            # K=3 flatten on X (3-1 partition).
            (('a', 'b', 'c'), {}, '$x', ['a', 'b', 'c'], [2, 3, 4], [12, 4, 1]),
            # K=3 flatten on Y (1-3 partition) — Y-side stride math.
            ('a', {}, '$y', ['b', 'c', 'd'], [3, 4, 5], [20, 5, 1]),
            # K=3 reversal: (a, b, c) → (c, b, a); strides reflect new order.
            (
                ('a', 'b', 'c'),
                {'transpose_x_flatten': True},
                '$x',
                ['c', 'b', 'a'],
                [4, 3, 2],
                [6, 2, 1],
            ),
        ],
        ids=['k3_x', 'k3_y', 'k3_x_reversed'],
    )
    def test_4d_k3_flatten_strides(
        self,
        data_abcd,
        data_key,
        axis_x,
        flags,
        fmt_key,
        names,
        sizes,
        strides,
    ) -> None:
        params = _make_params(('a', 'b', 'c', 'd'), axis_x=axis_x, **flags)
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abcd, data_key)
        fig = _run_hook(img)
        [hover] = [t for t in fig.toolbar.tools if isinstance(t, HoverTool)]
        fmt = hover.formatters[fmt_key]
        assert fmt.args['names'] == names
        assert fmt.args['sizes'] == sizes
        assert fmt.args['strides'] == strides


class TestFlattenPlotterCompute:
    def test_compute_caches_state(self, data_abc, data_key) -> None:
        params = _make_params(('a', 'b', 'c'), axis_x='b')
        plotter = FlattenPlotter.from_params(params)
        plotter.compute({'primary': {data_key: data_abc}})
        assert plotter.has_cached_state()

    def test_compute_renders_error_text_on_position_out_of_range(
        self, data_key
    ) -> None:
        # 2D data with axis_x position 2 → out-of-range error → base
        # Plotter.compute() wraps the ValueError in a Text element rather
        # than crashing.
        params = FlattenParams(flatten={'axis_x_dims': {2}})
        plotter = FlattenPlotter.from_params(params)
        data_2d = sc.DataArray(sc.zeros(dims=['x', 'y'], shape=[2, 2]))
        plotter.compute({'primary': {data_key: data_2d}})
        assert plotter.get_cached_state() is not None
