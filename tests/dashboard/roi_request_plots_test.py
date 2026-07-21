# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for interactive ROI request plotters."""

import holoviews as hv
import pytest

from ess.livedata.config.models import RectangleROI
from ess.livedata.config.workflow_spec import DataKey, WorkflowId
from ess.livedata.dashboard.data_roles import PRIMARY
from ess.livedata.dashboard.roi_publisher import FakeROIPublisher
from ess.livedata.dashboard.roi_request_plots import (
    RectanglesRequestParams,
    RectanglesRequestPlotter,
)

hv.extension('bokeh')


@pytest.fixture
def data_key() -> DataKey:
    return DataKey(
        workflow_id=WorkflowId(instrument='test', name='wf', version=1),
        source_name='test_source',
        output_name='roi_rectangle',
    )


@pytest.fixture
def computed_plotter(data_key: DataKey) -> RectanglesRequestPlotter:
    """A plotter with an empty initial set, publisher, and DataKey set."""
    plotter = RectanglesRequestPlotter.from_params(RectanglesRequestParams())
    plotter.set_roi_publisher(FakeROIPublisher())
    # compute() sets the DataKey required for publishing.
    plotter.compute({PRIMARY: {data_key: RectangleROI.to_concatenated_data_array({})}})
    return plotter


def _drawn_box() -> dict[str, list[float]]:
    """BoxEdit stream data for a single drawn rectangle."""
    return {'x0': [1.0], 'x1': [5.0], 'y0': [2.0], 'y1': [6.0]}


def test_first_presenter_publishes_initial_set_once(
    computed_plotter: RectanglesRequestPlotter,
) -> None:
    publisher = computed_plotter._roi_publisher
    assert isinstance(publisher, FakeROIPublisher)

    computed_plotter.create_presenter()

    assert len(publisher.published) == 1
    assert publisher.published[0][2] == {}


def test_second_presenter_does_not_republish_initial_set(
    computed_plotter: RectanglesRequestPlotter,
) -> None:
    """A second session must not republish the (stale) initial set."""
    publisher = computed_plotter._roi_publisher
    assert isinstance(publisher, FakeROIPublisher)

    computed_plotter.create_presenter()
    computed_plotter.create_presenter()

    # Exactly one publish across both presenter creations: the initial empty set.
    assert len(publisher.published) == 1
    assert publisher.published[0][2] == {}


def test_edit_persists_and_second_presenter_is_seeded_from_it(
    computed_plotter: RectanglesRequestPlotter,
) -> None:
    """After an edit, a new session sees the edited ROIs, not the config set."""
    publisher = computed_plotter._roi_publisher
    assert isinstance(publisher, FakeROIPublisher)

    presenter1 = computed_plotter.create_presenter()
    presenter1._handle_edit(_drawn_box())

    roi = computed_plotter._current_rois[0]
    assert (roi.x.min, roi.x.max, roi.y.min, roi.y.max) == (1.0, 5.0, 2.0, 6.0)

    presenter2 = computed_plotter.create_presenter()

    # Second presenter creation does not publish: no republish of stale/initial set.
    # Publishes so far: initial empty set, then the drawn ROI from the edit.
    assert len(publisher.published) == 2
    assert publisher.published[0][2] == {}
    assert publisher.published[1][2] == computed_plotter._current_rois

    # The new session's edit stream is seeded from the current (edited) ROIs.
    assert presenter2._edit_stream.data == {
        'x0': [1.0],
        'y0': [2.0],
        'x1': [5.0],
        'y1': [6.0],
    }
