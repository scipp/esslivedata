# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""The dashboard renders 2D data without HoloViews' ``apply_nodata`` compositor.

HoloViews runs that compositor over every ``Image``/``Raster``/``QuadMesh``/
``ImageStack`` frame. The operation rewrites an integer sentinel value to NaN and
returns its input for anything else, but the compositor machinery around it --
overlay wrapping, pattern matching, cloning the result back in -- runs per frame
regardless and costs ~1.4 ms per 2D layer per update, so ``reduction``
unregisters it.

Removing it is sound because the operation could never fire here: no plotter
offers the ``nodata`` option, and the 2D path converts to float64 before
plotting, which ``apply_nodata`` passes through untouched. Zero-transparency is
ours to do, in scipp (``_prepare_2d_image_data``). These tests pin that the
compositor is gone and that 2D data still arrives in Bokeh as float, unchanged.
"""

from __future__ import annotations

import holoviews as hv
import numpy as np
import pytest
import scipp as sc
from bokeh.models import ColumnDataSource, Plot
from holoviews.core.options import Compositor
from holoviews.plotting.bokeh import BokehRenderer
from holoviews.plotting.util import apply_nodata

# Imported for its side effect: unregistering the compositor.
import ess.livedata.dashboard.reduction  # noqa: F401
from ess.livedata.config.workflow_spec import DataKey, WorkflowId
from ess.livedata.dashboard import plots
from ess.livedata.dashboard.plot_params import PlotParams2d

_VALUES = np.arange(1.0, 13.0)


def test_apply_nodata_compositor_is_unregistered() -> None:
    assert not [d for d in Compositor.definitions if d.operation is apply_nodata]


def _data_2d(*, bin_edges: bool) -> dict:
    nx, ny = (5, 4) if bin_edges else (4, 3)
    return {
        DataKey(
            workflow_id=WorkflowId(instrument='test', name='wf', version=1),
            source_name='s0',
            output_name='out',
        ): sc.DataArray(
            sc.array(dims=['y', 'x'], values=_VALUES.reshape(3, 4)),
            coords={'x': sc.arange('x', float(nx)), 'y': sc.arange('y', float(ny))},
        )
    }


@pytest.mark.parametrize('bin_edges', [False, True], ids=['image', 'quadmesh'])
def test_2d_data_reaches_bokeh_unchanged(bin_edges: bool) -> None:
    """The 2D element types the compositor used to match still render."""
    plotter = plots.ImagePlotter.from_params(PlotParams2d())
    plotter.compute({'primary': _data_2d(bin_edges=bin_edges)})
    state = plotter.get_cached_state()
    assert not isinstance(state, hv.Text), f'plotter did not plot the data: {state}'

    presenter = plotter.create_presenter()
    rendered = presenter.present(hv.streams.Pipe(data=state))
    models = BokehRenderer.instance().get_plot(rendered).state.references()
    assert [model for model in models if isinstance(model, Plot)]

    columns = [
        np.asarray(column).ravel()
        for model in models
        if isinstance(model, ColumnDataSource)
        for column in model.data.values()
    ]
    # Float, so `apply_nodata` would have passed the values through even if a
    # plotter did set `nodata`.
    assert any(
        column.dtype == np.float64 and np.array_equal(np.sort(column), _VALUES)
        for column in columns
        if column.size == _VALUES.size
    ), 'the plotted values did not reach any ColumnDataSource'
