# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for interactive ROI request plotters."""

import holoviews as hv
import numpy as np
import pytest

from ess.livedata.config.models import PolygonROI, RectangleROI
from ess.livedata.config.workflow_spec import DataKey, WorkflowId
from ess.livedata.dashboard.data_roles import PRIMARY
from ess.livedata.dashboard.roi_publisher import FakeROIPublisher
from ess.livedata.dashboard.roi_request_plots import (
    PolygonsRequestParams,
    PolygonsRequestPlotter,
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


@pytest.mark.parametrize('count', [0, 1, 2])
def test_edit_from_browser_synced_arrays_is_applied(
    computed_plotter: RectanglesRequestPlotter, count: int
) -> None:
    """The browser syncs the BoxEdit columns back as numpy arrays.

    Their truth value raises for every length but one, so a truthiness guard
    dropped whole edits (silently -- the handler logs and swallows).
    """
    presenter = computed_plotter.create_presenter()

    presenter._handle_edit(
        {
            'x0': np.arange(count, dtype=float),
            'x1': np.arange(count, dtype=float) + 4.0,
            'y0': np.arange(count, dtype=float),
            'y1': np.arange(count, dtype=float) + 4.0,
        }
    )

    assert len(computed_plotter._current_rois) == count


@pytest.fixture
def persisted_params() -> list[RectanglesRequestParams]:
    return []


@pytest.fixture
def persisting_plotter(
    data_key: DataKey, persisted_params: list[RectanglesRequestParams]
) -> RectanglesRequestPlotter:
    """A computed plotter that records every params update it emits."""
    plotter = RectanglesRequestPlotter.from_params(RectanglesRequestParams())
    plotter.set_roi_publisher(FakeROIPublisher())
    plotter.set_params_persister(persisted_params.append)
    plotter.compute({PRIMARY: {data_key: RectangleROI.to_concatenated_data_array({})}})
    return plotter


def _rebuild(
    params: RectanglesRequestParams,
    data_key: DataKey,
    coord_units: dict[str, str] | None = None,
) -> RectanglesRequestPlotter:
    """Build a fresh plotter from params, as a dashboard restart would."""
    plotter = RectanglesRequestPlotter.from_params(params)
    plotter.set_roi_publisher(FakeROIPublisher())
    plotter.compute(
        {
            PRIMARY: {
                data_key: RectangleROI.to_concatenated_data_array(
                    {}, coord_units=coord_units
                )
            }
        }
    )
    return plotter


def test_edit_is_written_back_into_params(
    persisting_plotter: RectanglesRequestPlotter,
    persisted_params: list[RectanglesRequestParams],
) -> None:
    persisting_plotter.create_presenter()._handle_edit(_drawn_box())

    assert len(persisted_params) == 1
    assert persisted_params[-1].geometry.coordinates == '[1,2,5,6]'


def test_initial_publish_does_not_write_back_params(
    persisting_plotter: RectanglesRequestPlotter,
    persisted_params: list[RectanglesRequestParams],
) -> None:
    """Publishing the config-time set changes nothing that needs storing."""
    persisting_plotter.create_presenter()

    assert persisted_params == []


def test_rebuilt_plotter_seeds_from_persisted_edit(
    persisting_plotter: RectanglesRequestPlotter,
    persisted_params: list[RectanglesRequestParams],
    data_key: DataKey,
) -> None:
    """A dashboard restart must show the drawn ROIs, not the config-time set."""
    persisting_plotter.create_presenter()._handle_edit(_drawn_box())
    edited = persisting_plotter._current_rois

    rebuilt = _rebuild(persisted_params[-1], data_key)

    assert rebuilt._current_rois == edited


def test_rebuilt_plotter_republishes_the_persisted_edit(
    persisting_plotter: RectanglesRequestPlotter,
    persisted_params: list[RectanglesRequestParams],
    data_key: DataKey,
) -> None:
    """The set published to a new job is what the user drew, not the config."""
    persisting_plotter.create_presenter()._handle_edit(_drawn_box())
    edited = persisting_plotter._current_rois

    rebuilt = _rebuild(persisted_params[-1], data_key)
    rebuilt.create_presenter()

    publisher = rebuilt._roi_publisher
    assert isinstance(publisher, FakeROIPublisher)
    assert publisher.published[-1][2] == edited


def test_clearing_all_rois_survives_a_rebuild(
    persisting_plotter: RectanglesRequestPlotter,
    persisted_params: list[RectanglesRequestParams],
    data_key: DataKey,
) -> None:
    presenter = persisting_plotter.create_presenter()
    presenter._handle_edit(_drawn_box())
    presenter._handle_edit({'x0': [], 'x1': [], 'y0': [], 'y1': []})

    assert persisted_params[-1].geometry.coordinates == ''
    assert _rebuild(persisted_params[-1], data_key)._current_rois == {}


def test_stored_coordinates_are_in_plot_axis_units(data_key: DataKey) -> None:
    """Bare numbers in params mean coordinates on the plot's axes.

    Reading them back as pixel indices would silently displace every ROI drawn
    on a view with physical coordinates.
    """
    params = RectanglesRequestParams.model_validate(
        {'geometry': {'coordinates': '[0.1,0.2,0.5,0.6]'}}
    )

    plotter = _rebuild(params, data_key, coord_units={'x': 'm', 'y': 'm'})

    roi = plotter._current_rois[0]
    assert roi.x.unit == 'm'
    assert roi.y.unit == 'm'
    assert (roi.x.min, roi.x.max) == (0.1, 0.5)


def test_edit_on_a_physical_axis_round_trips_through_params(
    data_key: DataKey,
) -> None:
    units = {'x': 'm', 'y': 'm'}
    persisted: list[RectanglesRequestParams] = []
    plotter = _rebuild(RectanglesRequestParams(), data_key, coord_units=units)
    plotter.set_params_persister(persisted.append)

    plotter.create_presenter()._handle_edit(
        {'x0': [0.1], 'x1': [0.5], 'y0': [0.2], 'y1': [0.6]}
    )

    rebuilt = _rebuild(persisted[-1], data_key, coord_units=units)
    assert rebuilt._current_rois == plotter._current_rois


def _computed_polygon_plotter(
    params: PolygonsRequestParams, data_key: DataKey
) -> PolygonsRequestPlotter:
    plotter = PolygonsRequestPlotter.from_params(params)
    plotter.set_roi_publisher(FakeROIPublisher())
    plotter.compute(
        {
            PRIMARY: {
                data_key: PolygonROI.to_concatenated_data_array(
                    {}, coord_units={'x': 'm', 'y': 'm'}
                )
            }
        }
    )
    return plotter


def test_polygon_edit_round_trips_through_params(data_key: DataKey) -> None:
    persisted: list[PolygonsRequestParams] = []
    plotter = _computed_polygon_plotter(PolygonsRequestParams(), data_key)
    plotter.set_params_persister(persisted.append)

    plotter.create_presenter()._handle_edit(
        {'xs': [[0.0, 1.0, 0.5]], 'ys': [[0.0, 0.0, 1.0]]}
    )

    assert persisted[-1].geometry.coordinates == '[[0,0],[1,0],[0.5,1]]'
    rebuilt = _computed_polygon_plotter(persisted[-1], data_key)
    assert rebuilt._current_rois == plotter._current_rois


def test_stored_coordinates_drop_sub_pixel_float_noise(
    data_key: DataKey,
) -> None:
    """The stored string is also the params field the user reads and edits.

    A coordinate is a cursor pixel mapped onto the axis, so the digits past
    the first few are noise from that mapping, not information.
    """
    persisted: list[RectanglesRequestParams] = []
    plotter = _rebuild(RectanglesRequestParams(), data_key)
    plotter.set_params_persister(persisted.append)

    plotter.create_presenter()._handle_edit(
        {
            'x0': [0.10000000000000003],
            'x1': [12.345678901234567],
            'y0': [0.20000000000000004],
            'y1': [6.000000000000001],
        }
    )

    assert persisted[-1].geometry.coordinates == '[0.1,0.2,12.3457,6]'


def test_rounding_keeps_small_magnitude_axes_intact(data_key: DataKey) -> None:
    """Significant digits, not decimal places: a wavelength axis is ~1e-10 m."""
    persisted: list[RectanglesRequestParams] = []
    units = {'x': 'm', 'y': 'm'}
    plotter = _rebuild(RectanglesRequestParams(), data_key, coord_units=units)
    plotter.set_params_persister(persisted.append)

    plotter.create_presenter()._handle_edit(
        {'x0': [1.2345678e-10], 'x1': [5.4321e-10], 'y0': [1e-10], 'y1': [2e-10]}
    )

    rebuilt = _rebuild(persisted[-1], data_key, coord_units=units)
    roi = rebuilt._current_rois[0]
    assert roi.x.min == pytest.approx(1.2345678e-10, rel=1e-6)
    assert roi.x.max == pytest.approx(5.4321e-10, rel=1e-6)


def test_stored_geometry_is_a_fixed_point_of_the_round_trip(
    data_key: DataKey,
) -> None:
    """Rebuilding must not perturb geometry, however often it happens."""
    persisted: list[RectanglesRequestParams] = []
    plotter = _rebuild(RectanglesRequestParams(), data_key)
    plotter.set_params_persister(persisted.append)
    plotter.create_presenter()._handle_edit(
        {
            'x0': [0.10000000000000003],
            'x1': [12.345678901234567],
            'y0': [2.0],
            'y1': [6.0],
        }
    )
    stored = persisted[-1]

    rebuilt = _rebuild(stored, data_key)
    reformatted = rebuilt._format_geometry(rebuilt._current_rois)

    assert reformatted == stored.geometry.coordinates


def test_polygon_stored_geometry_is_a_fixed_point_of_the_round_trip(
    data_key: DataKey,
) -> None:
    persisted: list[PolygonsRequestParams] = []
    plotter = _computed_polygon_plotter(PolygonsRequestParams(), data_key)
    plotter.set_params_persister(persisted.append)
    plotter.create_presenter()._handle_edit(
        {
            'xs': [[0.10000000000000003, 1.2345678901234567, 0.5]],
            'ys': [[0.0, 0.0, 1.0000000000000002]],
        }
    )
    stored = persisted[-1]

    rebuilt = _computed_polygon_plotter(stored, data_key)
    reformatted = rebuilt._format_geometry(rebuilt._current_rois)

    assert reformatted == stored.geometry.coordinates


def test_degenerate_stored_rectangle_is_dropped_not_raised(data_key: DataKey) -> None:
    """A hand-edited zero-width rectangle must not take the plot down."""
    params = RectanglesRequestParams.model_validate(
        {'geometry': {'coordinates': '[1,2,1,6]'}}
    )

    assert _rebuild(params, data_key)._current_rois == {}


def _two_drawn_boxes() -> dict[str, list[float]]:
    """BoxEdit stream data after a second rectangle is drawn."""
    return {'x0': [1.0, 10.0], 'x1': [5.0, 15.0], 'y0': [2.0, 12.0], 'y1': [6.0, 16.0]}


def test_rendered_element_follows_the_drawn_rois(
    computed_plotter: RectanglesRequestPlotter,
) -> None:
    """Drawn ROIs must reach the element, not only the browser's edit tool.

    A re-render (tab switch, cell rebuilt on job state change) rebuilds the
    edit tool from the element. An element left at what the presenter was
    constructed with resyncs that stale set back as an edit, dropping the
    ROIs the user drew since.
    """
    presenter = computed_plotter.create_presenter()

    presenter._handle_edit(_drawn_box())
    presenter._handle_edit(_two_drawn_boxes())

    assert presenter._pipe.data == computed_plotter._converter.to_hv_data(
        computed_plotter._current_rois, index_to_color=None
    )
    assert len(presenter._pipe.data) == 2


def test_an_edit_that_changes_nothing_leaves_the_element_alone(
    computed_plotter: RectanglesRequestPlotter,
) -> None:
    presenter = computed_plotter.create_presenter()
    presenter._handle_edit(_drawn_box())
    rendered = presenter._pipe.data

    presenter._handle_edit(_drawn_box())

    assert presenter._pipe.data is rendered


def test_rendered_element_follows_a_drawn_polygon(data_key: DataKey) -> None:
    plotter = _computed_polygon_plotter(PolygonsRequestParams(), data_key)
    presenter = plotter.create_presenter()

    presenter._handle_edit({'xs': [[0.0, 1.0, 0.5]], 'ys': [[0.0, 0.0, 1.0]]})

    assert presenter._pipe.data == plotter._converter.to_hv_data(
        plotter._current_rois, index_to_color=None
    )
    assert presenter._create_element(presenter._pipe.data)


def test_a_polygon_still_being_drawn_does_not_reach_the_element(
    data_key: DataKey,
) -> None:
    """The trailing duplicate vertex is a cursor position, not a vertex."""
    plotter = _computed_polygon_plotter(PolygonsRequestParams(), data_key)
    presenter = plotter.create_presenter()

    presenter._handle_edit({'xs': [[0.0, 1.0, 1.0]], 'ys': [[0.0, 0.0, 0.0]]})

    assert presenter._pipe.data == []
