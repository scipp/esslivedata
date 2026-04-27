# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for FlattenPlotter and the dynamic param model."""

from __future__ import annotations

import uuid
from typing import Literal

import holoviews as hv
import pydantic
import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard.flatten_plotter import (
    FlattenAxisConfig,
    FlattenParams,
    FlattenPlotter,
    make_flatten_params,
)
from ess.livedata.dashboard.plot_params import PlotScale

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
    keep_dim: str,
    flatten_transposed: bool = False,
    transpose: bool = False,
) -> FlattenParams:
    Cls = make_flatten_params(dims)
    p = Cls(
        flatten={
            'keep_dim': keep_dim,
            'flatten_transposed': flatten_transposed,
            'transpose': transpose,
        }
    )
    p.plot_scale.color_scale = PlotScale.linear
    return p


def _axis_cls(params_cls: type[FlattenParams]) -> type[FlattenAxisConfig]:
    return params_cls.model_fields['flatten'].annotation


class TestMakeFlattenParams:
    def test_narrows_keep_dim_to_literal(self) -> None:
        Cls = make_flatten_params(('a', 'b', 'c'))
        annotation = _axis_cls(Cls).model_fields['keep_dim'].annotation
        assert annotation == Literal['a', 'b', 'c']

    def test_does_not_expose_outer_dim_choice(self) -> None:
        # The flatten ordering is controlled by a single boolean checkbox so
        # the user is exposed to (potentially placeholder) dim names only via
        # the keep_dim dropdown.
        Cls = make_flatten_params(('a', 'b', 'c'))
        fields = _axis_cls(Cls).model_fields
        assert 'flatten_outer' not in fields
        assert fields['flatten_transposed'].annotation is bool

    def test_rejects_invalid_dim_name(self) -> None:
        Cls = make_flatten_params(('a', 'b', 'c'))
        with pytest.raises(pydantic.ValidationError):
            Cls(flatten={'keep_dim': 'nope'})

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

    def test_natural_order_sets_inner_size_in_formatter(
        self, data_abc, data_key
    ) -> None:
        # keep_dim='b' → outer='a' (size 3), inner='c' (size 5) → inner_size=5
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        formatter = img.opts.get('plot').kwargs['yformatter']
        assert formatter.args['inner_size'] == 5
        assert len(formatter.args['outer_values']) == 3
        assert len(formatter.args['inner_values']) == 5

    def test_flatten_transposed_changes_inner_size(self, data_abc, data_key) -> None:
        # outer='c' (size 5), inner='a' (size 3) → inner_size=3
        params = _make_params(('a', 'b', 'c'), keep_dim='b', flatten_transposed=True)
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data_abc, data_key)
        formatter = img.opts.get('plot').kwargs['yformatter']
        assert formatter.args['inner_size'] == 3

    def test_missing_coord_falls_back_to_index_template(
        self, data_abc, data_key
    ) -> None:
        # Drop coord on the inner dim ('c' under natural order with keep='b')
        data = data_abc.copy(deep=False)
        del data.coords['c']
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data, data_key)
        formatter = img.opts.get('plot').kwargs['yformatter']
        # outer (a) has values, inner (c) falls back to "c[%d]"
        assert formatter.args['outer_values'] != []
        assert formatter.args['inner_values'] == []
        assert '%d' in formatter.args['inner_template']

    def test_bin_edge_coord_uses_midpoints(self, data_key) -> None:
        # 'a' (the outer under natural order with keep='b') has bin edges
        # (size 4 for dim size 3) → midpoints = 3 values
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
        formatter = img.opts.get('plot').kwargs['yformatter']
        assert len(formatter.args['outer_values']) == 3  # midpoints of 4 edges

    def test_raises_for_non_3d_input(self, data_key) -> None:
        data = sc.DataArray(sc.zeros(dims=['x', 'y'], shape=[2, 2]))
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        with pytest.raises(ValueError, match='3D input'):
            plotter.plot(data, data_key)

    def test_falls_back_to_positional_when_data_renamed(
        self, data_abc, data_key
    ) -> None:
        # Template dims (a, b, c) configured with keep_dim='b'; runtime data
        # is renamed to (a, d, c). Order preserved → 'b' resolves to position
        # 1 → 'd'. This is the common case when shared templates declare
        # generic dim names that per-instrument transforms rename.
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        data = data_abc.rename_dims({'b': 'd'})
        plotter = FlattenPlotter.from_params(params)
        img = plotter.plot(data, data_key)
        assert img.kdims[0].name == 'd'  # x = kept, resolved positionally
        assert img.kdims[1].name == 'a·c'

    def test_raises_when_neither_name_nor_position_resolves(
        self, data_abc, data_key
    ) -> None:
        # Configured 'b' is not in data.dims and template_dims is unavailable
        # → cannot resolve.
        params = _make_params(('a', 'b', 'c'), keep_dim='b')
        plotter = FlattenPlotter.from_params(params)
        # Manually disable the positional fallback to simulate "no template"
        plotter._template_dims = None
        data = data_abc.rename_dims({'b': 'd'})
        with pytest.raises(ValueError, match='matches neither'):
            plotter.plot(data, data_key)


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
