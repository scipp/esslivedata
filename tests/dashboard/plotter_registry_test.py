# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the plotter registry and cross-cutting plotter contracts."""

import math
from collections.abc import Callable
from typing import Any

import holoviews as hv
import numpy as np
import pydantic
import pytest
import scipp as sc
from bokeh.document import Document
from bokeh.models import Text
from holoviews.plotting.bokeh import BokehRenderer

from ess.livedata.config.models import Interval, PolygonROI, RectangleROI
from ess.livedata.config.workflow_spec import DataKey, WorkflowId
from ess.livedata.dashboard.data_roles import PRIMARY, X_AXIS, Y_AXIS
from ess.livedata.dashboard.plot_params import PlotAspectType
from ess.livedata.dashboard.plots import TitleResolver
from ess.livedata.dashboard.plotter_registry import plotter_registry

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


def _array(sizes: dict[str, int], *, variances: bool = False) -> sc.DataArray:
    """Counts on evenly spaced 1-D coords, as the slicer requires.

    With ``variances`` the 1-D line plotters draw error bars, whose endcaps are
    Bokeh models of their own.
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
            dim: sc.arange(dim, float(size), unit='m') for dim, size in sizes.items()
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
    return _array({'time': n, 'roi': 3}).assign_coords(
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

# Input to every registered plotter that makes it draw its full frame.
_DATA: dict[str, Callable[[], RoleData]] = {
    'image': lambda: _primary(_array({'y': 3, 'x': 4})),
    'lines': lambda: _primary(
        _array({'x': 5}, variances=True), _array({'x': 5}, variances=True)
    ),
    'timeseries': lambda: _primary(_history(), _history()),
    'bars': lambda: _primary(_array({}), _array({})),
    'table': lambda: _primary(_array({}), _array({})),
    'slicer': lambda: _primary(_array({'z': 2, 'y': 3, 'x': 4})),
    'flatten': lambda: _primary(_array({'a': 2, 'b': 3, 'c': 4})),
    'overlay_1d': lambda: _primary(_array({'roi': 2, 'x': 5}, variances=True)),
    'overlay_1d_values': lambda: _primary(_roi_values(), _roi_values()),
    'overlay_1d_timeseries': lambda: _primary(_roi_history(), _roi_history()),
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

# Params without which a plotter draws nothing, i.e., has no geometry to show.
_PARAMS: dict[str, dict[str, Any]] = {
    'rectangles': {'geometry': {'coordinates': '[0, 0, 2, 1]'}},
    'vlines': {'geometry': {'positions': '1, 2'}},
    'hlines': {'geometry': {'positions': '1, 2'}},
    'rectangles_request': {'geometry': {'coordinates': '[0, 0, 2, 1]'}},
    'polygons_request': {'geometry': {'coordinates': '[[0, 0], [2, 0], [1, 1]]'}},
}


def _session_cases() -> list:
    """Each plotter, per aspect where it has one: a fixed aspect adds hooks."""
    cases = []
    for name, entry in plotter_registry.items():
        if 'plot_aspect' in entry.spec.params.model_fields:
            cases += [
                pytest.param(name, aspect, id=f'{name}-{aspect.name}')
                for aspect in (PlotAspectType.free, PlotAspectType.square)
            ]
        else:
            cases.append(pytest.param(name, None, id=name))
    return cases


def _make_params(
    plotter_name: str, aspect: PlotAspectType | None, dims: tuple[str, ...]
) -> pydantic.BaseModel:
    """Params as the plot config modal builds them for data with ``dims``."""
    entry = plotter_registry[plotter_name]
    params_cls = (
        entry.params_factory(dims) if entry.params_factory else entry.spec.params
    )
    params = _PARAMS.get(plotter_name, {})
    if aspect is not None:
        params = {**params, 'plot_aspect': {'aspect_type': aspect}}
    return params_cls.model_validate(params)


@pytest.mark.parametrize(('plotter_name', 'aspect'), _session_cases())
def test_computed_frame_renders_in_several_sessions(
    plotter_name: str, aspect: PlotAspectType | None
) -> None:
    """Every browser session renders the one frame that compute() shares.

    Bokeh lets a model belong to only one document, so a frame or opts holding a
    Bokeh model instance render in the first session and raise in the second
    (see ``Plotter.style_opts``). Each session gets its own presenter and pipe,
    as in ``SessionComponents.create``.
    """
    data = _DATA[plotter_name]()
    dims = next((da.dims for da in data.get(PRIMARY, {}).values()), ())
    params = _make_params(plotter_name, aspect, dims)
    plotter = plotter_registry[plotter_name].factory(params)
    plotter.compute(data)

    pipes = []
    for _ in range(2):
        doc = Document()
        pipe = hv.streams.Pipe(data=plotter.get_cached_state())
        dmap = plotter.create_presenter().present(pipe)
        # get_plot(doc=...) does not attach the models yet; add_root does.
        doc.add_root(BokehRenderer.instance().get_plot(dmap, doc=doc).state)
        # A "No data" or error placeholder would pass vacuously.
        assert not list(doc.select({'type': Text}))
        pipes.append(pipe)

    # Sessions keep receiving frames, and HoloViews runs hooks on each update.
    plotter.compute(data)
    for pipe in pipes:
        pipe.send(plotter.get_cached_state())


def test_session_test_covers_every_plotter() -> None:
    """A newly registered plotter needs input in ``_DATA`` for the session test."""
    assert set(_DATA) == set(plotter_registry.keys())
    assert set(_PARAMS) <= set(plotter_registry.keys())
