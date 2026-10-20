# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for Frame: which elements can be written into a figure directly.

That the columns of writable elements match HoloViews' own is checked for every
registered plotter in ``plotter_registry_test.py``.
"""

import holoviews as hv
import numpy as np
import pytest

from ess.livedata.dashboard.frame import Frame

hv.extension('bokeh')

_X = np.arange(3.0)


@pytest.mark.parametrize(
    'element',
    [
        hv.Curve((_X, _X)),
        hv.Scatter((_X, _X)),
        hv.Histogram((np.arange(4.0), _X)),
        hv.ErrorBars((_X, _X, _X)),
        hv.Spread((_X, _X, _X)),
        hv.Image(np.ones((2, 3))),
        hv.Text(0.5, 0.5, 'No data'),
    ],
    ids=lambda element: type(element).__name__,
)
def test_element_is_written_directly(element: hv.Element) -> None:
    assert Frame.from_element(hv.Overlay([element])).columns is not None


@pytest.mark.parametrize(
    'element',
    [
        pytest.param(hv.Curve((_X, _X)).opts(interpolation='steps-mid'), id='steps'),
        pytest.param(hv.Scatter((_X, _X)).opts(jitter=0.1), id='jitter'),
        pytest.param(
            hv.Histogram((np.arange(4.0), _X)).opts(logy=True), id='log-y-histogram'
        ),
        pytest.param(hv.Curve((_X, _X)).opts(invert_axes=True), id='inverted'),
        pytest.param(hv.Curve((['a', 'b', 'c'], _X)), id='categorical'),
        pytest.param(hv.ErrorBars((_X, _X, _X), horizontal=True), id='horizontal'),
        pytest.param(
            hv.ErrorBars((_X, _X, _X, _X), vdims=['y', 'neg', 'pos']),
            id='asymmetric',
        ),
        pytest.param(hv.Image(np.ones((2, 3), dtype=bool)), id='bool-image'),
        pytest.param(
            hv.Image((_X, _X[:2], np.ones((2, 3)), np.ones((2, 3))), vdims=['a', 'b']),
            id='two-valued-image',
        ),
        pytest.param(hv.Bars([('a', 1)]), id='no-column-function'),
    ],
)
def test_element_is_drawn_through_holoviews(element: hv.Element) -> None:
    """HoloViews derives other columns for these than the column functions do."""
    frame = Frame.from_element(hv.Overlay([hv.Curve((_X, _X)), element]))

    assert frame.columns is None


def test_frame_of_unwritable_element_is_specified_by_layout_shape_only() -> None:
    """Such a frame's figure follows its updates itself, as HoloViews redraws it."""
    first = Frame.from_element(hv.Overlay([hv.Bars([('a', 1)])]))
    second = Frame.from_element(hv.Overlay([hv.Bars([('a', 1), ('b', 2)])]))

    assert first.spec == second.spec
