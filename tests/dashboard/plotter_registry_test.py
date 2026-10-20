# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the plotter registry and cross-cutting plotter contracts."""

import math
import re
from collections.abc import Callable
from typing import Any

import holoviews as hv
import numpy as np
import panel as pn
import pydantic
import pytest
import scipp as sc
from bokeh.document import Document
from bokeh.models import Text
from bokeh.plotting import figure as Figure
from holoviews.plotting.bokeh import BokehRenderer

from ess.livedata.config.models import Interval, PolygonROI, RectangleROI
from ess.livedata.config.workflow_spec import DataKey, WorkflowId
from ess.livedata.dashboard.data_roles import PRIMARY, X_AXIS, Y_AXIS
from ess.livedata.dashboard.direct_figure import DirectFigure
from ess.livedata.dashboard.extractors import FullHistoryExtractor
from ess.livedata.dashboard.plot_params import (
    ErrorDisplay,
    Line1dRenderMode,
    PlotAspectType,
    PlotScale,
)
from ess.livedata.dashboard.plots import TitleResolver
from ess.livedata.dashboard.plotter_registry import plotter_registry
from ess.livedata.dashboard.session_layer import SessionComponents

hv.extension('bokeh')

RoleData = dict[str, dict[DataKey, sc.DataArray]]


def _all_plotter_names() -> list[str]:
    """Get names of all registered plotters."""
    return list(plotter_registry.keys())


class TestPlotterComputeSignature:
    """Verify all registered plotters accept the kwargs passed by PlotOrchestrator."""

    @pytest.mark.parametrize("plotter_name", _all_plotter_names())
    def test_compute_accepts_title_resolver_kwarg(self, plotter_name):
        """PlotOrchestrator passes title_resolver= to all plotters.

        Each plotter's compute() must accept this keyword argument
        (either explicitly or via **kwargs) without raising TypeError.
        """
        entry = plotter_registry[plotter_name]
        default_params = entry.spec.params()
        plotter = entry.factory(default_params)

        try:
            plotter.compute({}, title_resolver=TitleResolver())
        except TypeError as e:
            if "title_resolver" in str(e):
                pytest.fail(
                    f"Plotter '{plotter_name}' does not accept 'title_resolver' kwarg. "
                    f"Add title_resolver to its compute() signature."
                )
        except Exception:  # noqa: S110
            pass  # Other errors (empty data, missing keys, etc.) are expected


def _key(source_name: str) -> DataKey:
    return DataKey(
        workflow_id=WorkflowId(instrument='test', name='wf', version=1),
        source_name=source_name,
        output_name='out',
    )


def _array(
    sizes: dict[str, int], *, variances: bool = False, edges: str | None = None
) -> sc.DataArray:
    """Counts on evenly spaced 1-D coords, as the slicer requires.

    With ``variances`` the 1-D line plotters draw error bars, whose endcaps are
    Bokeh models of their own. The coord of the ``edges`` dim holds bin edges,
    which the 1-D line plotters need to draw a histogram.
    """
    values = np.arange(1.0, math.prod(sizes.values()) + 1).reshape(
        tuple(sizes.values())
    )
    return sc.DataArray(
        sc.array(
            dims=list(sizes),
            values=values,
            variances=values if variances else None,
            unit='counts',
        ),
        coords={
            dim: sc.arange(dim, float(size + (dim == edges)), unit='m')
            for dim, size in sizes.items()
        },
    )


def _history(n: int = 10) -> sc.DataArray:
    """Scalar stream history shaped like the output of FullHistoryExtractor."""
    time = sc.datetime('2026-01-01T00:00:00', unit='ns') + sc.arange(
        'time', n, unit='s'
    ).to(unit='ns')
    return _array({'time': n}, variances=True).assign_coords(
        time=time, start_time=time[0], end_time=time[-1]
    )


def _roi_values() -> sc.DataArray:
    """A per-ROI scalar output, e.g., counts per ROI."""
    return _array({'roi': 3}).assign_coords(
        roi=sc.array(dims=['roi'], values=[0, 1, 4], dtype='int32', unit=None)
    )


def _roi_history(n: int = 10) -> sc.DataArray:
    """Per-ROI output history shaped like the output of FullHistoryExtractor."""
    time = _history(n).coords['time']
    return _array({'time': n, 'roi': 3}, variances=True).assign_coords(
        time=time,
        start_time=time[0],
        end_time=time[-1],
        roi=sc.array(dims=['roi'], values=[0, 1, 4], dtype='int32', unit=None),
    )


def _primary(*arrays: sc.DataArray) -> RoleData:
    return {PRIMARY: {_key(f'source{i}'): da for i, da in enumerate(arrays)}}


_RECTANGLES = RectangleROI.to_concatenated_data_array(
    {
        0: RectangleROI(
            x=Interval(min=0.0, max=2.0, unit='m'),
            y=Interval(min=0.0, max=1.0, unit='m'),
        )
    }
)
_POLYGONS = PolygonROI.to_concatenated_data_array(
    {4: PolygonROI(x=[0.0, 2.0, 1.0], y=[0.0, 0.0, 1.0], x_unit='m', y_unit='m')}
)


def _with_nan(da: sc.DataArray) -> sc.DataArray:
    """``da`` with a NaN value, which splits an error band into two polygons."""
    da = da.copy()
    da.values[1] = np.nan
    return da


# Input to every registered plotter that makes it draw its full frame. The 1-D
# line plotters draw a histogram only for data with bin edges.
_DATA: dict[str, Callable[[], RoleData]] = {
    'image': lambda: _primary(_array({'y': 3, 'x': 4})),
    'lines': lambda: _primary(
        _with_nan(_array({'x': 5}, variances=True)),
        _array({'x': 5}, variances=True, edges='x'),
    ),
    'timeseries': lambda: _primary(_history(), _history()),
    'bars': lambda: _primary(_array({}), _array({})),
    'table': lambda: _primary(_array({}), _array({})),
    'slicer': lambda: _primary(_array({'z': 2, 'y': 3, 'x': 4})),
    'flatten': lambda: _primary(_array({'a': 2, 'b': 3, 'c': 4})),
    # Integer entry coords such as ROI indices, which label the curves; ``_vary``
    # keeps them.
    'overlay_1d': lambda: _primary(
        _array({'roi': 2, 'x': 5}, variances=True, edges='x').assign_coords(
            roi=sc.array(dims=['roi'], values=[0, 4], dtype='int32', unit=None)
        )
    ),
    'timeseries_overlay': lambda: _primary(_roi_history(), _roi_history()),
    'correlation_histogram_1d': lambda: {
        **_primary(_history()),
        X_AXIS: {_key('axis_x'): _history()},
    },
    'correlation_histogram_2d': lambda: {
        **_primary(_history()),
        X_AXIS: {_key('axis_x'): _history()},
        Y_AXIS: {_key('axis_y'): _history()},
    },
    'rectangles': lambda: {},
    'vlines': lambda: {},
    'hlines': lambda: {},
    'rectangles_readback': lambda: _primary(_RECTANGLES),
    'rectangles_request': lambda: _primary(_RECTANGLES),
    'polygons_readback': lambda: _primary(_POLYGONS),
    'polygons_request': lambda: _primary(_POLYGONS),
}

# Further input for plotters with a separate code path per rank: 1D bars carry
# per-element opts, 1D tables an entry column.
_DATA_VARIANTS: dict[str, dict[str, Callable[[], RoleData]]] = {
    'bars': {'1d': lambda: _primary(_roi_values(), _roi_values())},
    'table': {'1d': lambda: _primary(_roi_values(), _roi_values())},
}

# Params without which a plotter draws nothing, i.e., has no geometry to show.
_PARAMS: dict[str, dict[str, Any]] = {
    'rectangles': {'geometry': {'coordinates': '[0, 0, 2, 1]'}},
    'vlines': {'geometry': {'positions': '1, 2'}},
    'hlines': {'geometry': {'positions': '1, 2'}},
    'rectangles_request': {'geometry': {'coordinates': '[0, 0, 2, 1]'}},
    'polygons_request': {'geometry': {'coordinates': '[[0, 0], [2, 0], [1, 1]]'}},
}


def _line_variants(*log_axes: str) -> dict[str, dict[str, Any]]:
    """Each line mode with each error display, and a log scale per axis.

    Each mode and error display draws its own element type.
    """
    return {
        f'{mode}-{errors}': {'line': {'mode': mode, 'errors': errors}}
        for mode in Line1dRenderMode
        for errors in ErrorDisplay
    } | {f'log-{axis}': {'plot_scale': {axis: PlotScale.log}} for axis in log_axes}


# Further params, each drawn with the input in ``_DATA``. A timeseries has a
# datetime x-axis, which cannot be logarithmic.
_PARAMS_VARIANTS: dict[str, dict[str, dict[str, Any]]] = {
    'lines': _line_variants('x_scale', 'y_scale'),
    'overlay_1d': _line_variants('x_scale', 'y_scale'),
    'correlation_histogram_1d': _line_variants('x_scale', 'y_scale'),
    'timeseries': _line_variants('y_scale'),
    'timeseries_overlay': _line_variants('y_scale'),
}


def _session_cases() -> list:
    """Each plotter and input, per aspect where it has one (a fixed aspect adds
    hooks), and each params variant."""
    cases = []
    for name, entry in plotter_registry.items():
        inputs = {name: _DATA[name]} | {
            f'{name}-{variant}': make_data
            for variant, make_data in _DATA_VARIANTS.get(name, {}).items()
        }
        aspects = (
            {
                f'-{aspect.name}': {'plot_aspect': {'aspect_type': aspect}}
                for aspect in (PlotAspectType.free, PlotAspectType.square)
            }
            if 'plot_aspect' in entry.spec.params.model_fields
            else {'': {}}
        )
        cases += [
            pytest.param(name, make_data, params, id=case_id + suffix)
            for case_id, make_data in inputs.items()
            for suffix, params in aspects.items()
        ]
        cases += [
            pytest.param(name, _DATA[name], params, id=f'{name}-{variant}')
            for variant, params in _PARAMS_VARIANTS.get(name, {}).items()
        ]
    return cases


def _make_params(
    plotter_name: str, params: dict[str, Any], dims: tuple[str, ...]
) -> pydantic.BaseModel:
    """Params as the plot config modal builds them for data with ``dims``."""
    entry = plotter_registry[plotter_name]
    params_cls = (
        entry.params_factory(dims) if entry.params_factory else entry.spec.params
    )
    return params_cls.model_validate({**_PARAMS.get(plotter_name, {}), **params})


@pytest.mark.parametrize(('plotter_name', 'make_data', 'params'), _session_cases())
def test_computed_frame_renders_in_several_sessions(
    plotter_name: str,
    make_data: Callable[[], RoleData],
    params: dict[str, Any],
) -> None:
    """Every browser session renders the one frame that compute() shares.

    Bokeh lets a model belong to only one document, so a frame or opts holding a
    Bokeh model instance render in the first session and raise in the second
    (see ``Plotter.style_opts``). Each session gets its own presenter and pipe,
    as in ``SessionComponents.create``.
    """
    data = make_data()
    dims = next((da.dims for da in data.get(PRIMARY, {}).values()), ())
    plotter = plotter_registry[plotter_name].factory(
        _make_params(plotter_name, params, dims)
    )
    plotter.compute(data)

    pipes = []
    for _ in range(2):
        doc = Document()
        pipe = hv.streams.Pipe(data=plotter.get_frame().element)
        dmap = plotter.create_presenter().present(pipe)
        # get_plot(doc=...) does not attach the models yet; add_root does.
        doc.add_root(BokehRenderer.instance().get_plot(dmap, doc=doc).state)
        # A "No data" or error placeholder would pass vacuously.
        assert not list(doc.select({'type': Text}))
        pipes.append(pipe)

    # Sessions keep receiving frames, and HoloViews runs hooks on each update.
    plotter.compute(data)
    for pipe in pipes:
        pipe.send(plotter.get_frame().element)


def test_session_test_covers_every_plotter() -> None:
    """A newly registered plotter needs input in ``_DATA`` for the session test."""
    assert set(_DATA) == set(plotter_registry.keys())
    assert set(_PARAMS) <= set(plotter_registry.keys())
    assert set(_DATA_VARIANTS) <= set(plotter_registry.keys())
    assert set(_PARAMS_VARIANTS) <= set(plotter_registry.keys())


class TestPerEntryPlotters:
    def test_1d_data_is_offered_bars_table_and_a_full_history_plot(self):
        compatible = plotter_registry.get_compatible_plotters(
            {_key('det'): _roi_values()}
        )
        assert compatible['bars'].data_requirements.required_extractor is None
        assert compatible['table'].data_requirements.required_extractor is None
        assert (
            compatible['timeseries_overlay'].data_requirements.required_extractor
            is FullHistoryExtractor
        )

    def test_scalars_are_offered_bars_but_not_the_per_entry_history(self):
        compatible = plotter_registry.get_compatible_plotters({_key('det'): _array({})})
        assert 'bars' in compatible
        assert 'timeseries_overlay' not in compatible

    @pytest.mark.parametrize(
        'data',
        [
            pytest.param(_array({'toa': 3}), id='physical-coord'),
            pytest.param(_array({'pixel': 3}).drop_coords('pixel'), id='no-coord'),
            pytest.param(
                _array({'roi': 3}).assign_coords(
                    roi=sc.array(dims=['roi'], values=[0, 1, 2, 3], unit=None)
                ),
                id='bin-edge-labels',
            ),
        ],
    )
    def test_1d_data_without_label_coord_is_offered_lines_only(
        self, data: sc.DataArray
    ):
        compatible = plotter_registry.get_compatible_plotters({_key('det'): data})
        assert set(compatible) == {'lines'}


# Plotters that draw coord values into the hover tool of the figure, so new
# coords rebuild it. Detector coords do not change between updates.
_COORDS_IN_SPEC = {'flatten'}


def _vary(data: RoleData, *, coords: bool = True) -> RoleData:
    """``data`` with other values, and unless disabled coords, of the same structure.

    Float coords are shifted and datetimes advanced; integer coords label
    entries (ROI indices), so they are kept.
    """

    def shift(coord: sc.Variable) -> sc.Variable:
        if not coords:
            return coord
        if coord.dtype == sc.DType.datetime64:
            return coord + sc.scalar(1, unit='s').to(unit=coord.unit)
        if coord.dtype == sc.DType.float64:
            return coord + sc.scalar(0.5, unit=coord.unit)
        return coord

    return {
        role: {
            key: (3 * da).assign_coords({n: shift(c) for n, c in da.coords.items()})
            for key, da in arrays.items()
        }
        for role, arrays in data.items()
    }


def _computed_plotter(
    plotter_name: str, make_data: Callable[[], RoleData], params: dict[str, Any]
) -> Any:
    data = make_data()
    dims = next((da.dims for da in data.get(PRIMARY, {}).values()), ())
    plotter = plotter_registry[plotter_name].factory(
        _make_params(plotter_name, params, dims)
    )
    plotter.compute(data)
    return plotter


@pytest.mark.parametrize(('plotter_name', 'make_data', 'params'), _session_cases())
def test_frame_spec_does_not_change_with_values(
    plotter_name: str,
    make_data: Callable[[], RoleData],
    params: dict[str, Any],
) -> None:
    """New values in the same structure must not rebuild the figure.

    Each spec change rebuilds the cell in every session. A spec holding a value
    that changes with the data (a per-frame closure, a data-dependent option)
    would rebuild it on every update.
    """
    plotter = _computed_plotter(plotter_name, make_data, params)
    spec = plotter.get_frame().spec

    plotter.compute(_vary(make_data(), coords=plotter_name not in _COORDS_IN_SPEC))

    assert plotter.get_frame().spec == spec


@pytest.mark.parametrize('value', [np.nan, 5.0])
def test_log_scale_image_spec_does_not_change_with_its_fallback_color_range(
    value: float,
) -> None:
    """All-NaN or uniform values give a log-scale image a colour range from the data.

    HoloViews cannot pick one there, so the plotter sets it, and it changes with
    the values. It must not rebuild the figure on every update.
    """

    def make_data(scale: float = 1.0) -> RoleData:
        da = _array({'y': 3, 'x': 4})
        da.values = np.full(da.shape, scale * value)
        return _primary(da)

    params = {'plot_scale': {'color_scale': PlotScale.log}}
    plotter = _computed_plotter('image', make_data, params)
    spec = plotter.get_frame().spec

    plotter.compute(make_data(scale=2.0))

    assert plotter.get_frame().spec == spec


def _direct_figure(plotter: Any) -> tuple[DirectFigure, SessionComponents]:
    """A figure of the plotter's current frame, and the session layer feeding it."""
    components = SessionComponents(
        presenter=plotter.create_presenter(), frame=plotter.get_frame()
    )
    figure = DirectFigure(
        components.frame.element,
        [components],
        margin=pn.pane.HoloViews.param.margin.default,
    )
    return figure, components


def _shown_columns(figure: Any) -> list[dict[str, Any]]:
    return [dict(renderer.data_source.data) for renderer in figure.renderers]


@pytest.mark.parametrize(('plotter_name', 'make_data', 'params'), _session_cases())
def test_written_frame_matches_a_figure_built_from_it(
    plotter_name: str,
    make_data: Callable[[], RoleData],
    params: dict[str, Any],
) -> None:
    """Writing a frame's columns gives the data sources HoloViews builds from it."""
    plotter = _computed_plotter(plotter_name, make_data, params)
    if plotter.get_frame().columns is None:
        pytest.skip('drawn through HoloViews')
    figure, components = _direct_figure(plotter)

    plotter.compute(_vary(make_data(), coords=plotter_name not in _COORDS_IN_SPEC))
    assert components.push_update()

    written = _shown_columns(figure.figure)
    built = _shown_columns(_direct_figure(plotter)[0].figure)
    assert [columns.keys() for columns in written] == [
        columns.keys() for columns in built
    ]
    for got, want in zip(written, built, strict=True):
        for name in want:
            np.testing.assert_array_equal(got[name], want[name], strict=True)


def _model_properties(root: Any) -> list[tuple[str, dict[str, str]]]:
    """Type and non-default properties of every model, in a stable order.

    Model ids and data-source contents (compared by the test above) are left out.
    """
    models = []
    for model in root.references():
        properties = {
            name: re.sub(r"id='[^']*'", '', repr(value))
            for name, value in model.properties_with_values(
                include_defaults=False
            ).items()
            if name not in ('id', 'data')
        }
        models.append((type(model).__name__, properties))
    return sorted(models, key=repr)


@pytest.mark.parametrize(('plotter_name', 'make_data', 'params'), _session_cases())
def test_direct_figure_matches_the_holoviews_pane(
    plotter_name: str,
    make_data: Callable[[], RoleData],
    params: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A direct figure is the figure Panel's HoloViews pane would draw.

    WebGL is disabled as in the dashboard (``reduction``): the figure must be
    built by the configured renderer, not by a default one.
    """
    monkeypatch.setattr(hv.renderer('bokeh'), 'webgl', False)
    plotter = _computed_plotter(plotter_name, make_data, params)
    if plotter.get_frame().columns is None:
        pytest.skip('drawn through HoloViews')
    pane = pn.pane.HoloViews(plotter.get_frame().element)
    (figure,) = pane.get_root(Document()).select({'type': Figure})

    direct, _ = _direct_figure(plotter)
    assert _model_properties(direct.figure) == _model_properties(figure)
