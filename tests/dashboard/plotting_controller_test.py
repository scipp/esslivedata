# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import holoviews as hv
import pydantic
import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import (
    REDUCTION,
    WorkflowId,
    WorkflowOutputsBase,
    WorkflowSpec,
)
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.plotting_controller import (
    PlottingController,
    output_view_supports_windowing,
    since_start_available,
)
from ess.livedata.dashboard.stream_manager import StreamManager

hv.extension('bokeh')


@pytest.fixture
def data_service():
    """Create a DataService for testing."""
    return DataService()


@pytest.fixture
def stream_manager(data_service):
    """Create a StreamManager for testing."""
    return StreamManager(data_service=data_service)


@pytest.fixture
def plotting_controller(stream_manager):
    """Create a PlottingController for testing."""
    return PlottingController(
        stream_manager=stream_manager,
    )


class TestGetAvailablePlottersFromSpec:
    """Tests for PlottingController.get_available_plotters_from_spec()."""

    def test_returns_compatible_plotters_for_1d_template(self, plotting_controller):
        """Test that 1D output template returns compatible plotters."""

        class TestOutputs(WorkflowOutputsBase):
            i_of_q: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['Q'], shape=[0], unit='counts'),
                    coords={'Q': sc.arange('Q', 0, unit='1/angstrom')},
                )
            )

        spec = WorkflowSpec(
            instrument='test',
            name='test_workflow',
            version=1,
            title='Test',
            description='Test',
            outputs=TestOutputs,
            params=None,
            group=REDUCTION,
        )

        plotters, has_template = plotting_controller.get_available_plotters_from_spec(
            workflow_spec=spec, view_name='i_of_q'
        )

        assert has_template is True
        assert 'lines' in plotters
        assert 'image' not in plotters

    def test_returns_compatible_plotters_for_2d_template(self, plotting_controller):
        """Test that 2D output template returns compatible plotters."""

        class TestOutputs(WorkflowOutputsBase):
            detector: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['x', 'y'], shape=[0, 0], unit='counts'),
                    coords={
                        'x': sc.arange('x', 0, unit='m'),
                        'y': sc.arange('y', 0, unit='m'),
                    },
                )
            )

        spec = WorkflowSpec(
            instrument='test',
            name='test_workflow',
            version=1,
            title='Test',
            description='Test',
            outputs=TestOutputs,
            params=None,
            group=REDUCTION,
        )

        plotters, has_template = plotting_controller.get_available_plotters_from_spec(
            workflow_spec=spec, view_name='detector'
        )

        assert has_template is True
        assert 'image' in plotters
        assert 'lines' not in plotters

    def test_returns_all_plotters_when_no_template(self, plotting_controller):
        """Test that all plotters are returned as fallback when no template exists."""

        class TestOutputs(WorkflowOutputsBase):
            result: sc.DataArray = pydantic.Field(title='Result')

        spec = WorkflowSpec(
            instrument='test',
            name='test_workflow',
            version=1,
            title='Test',
            description='Test',
            outputs=TestOutputs,
            params=None,
            group=REDUCTION,
        )

        plotters, has_template = plotting_controller.get_available_plotters_from_spec(
            workflow_spec=spec, view_name='result'
        )

        assert has_template is False
        assert len(plotters) > 0
        assert 'lines' in plotters
        assert 'image' in plotters


class TestGetAvailableOverlays:
    """Tests for PlottingController.get_available_overlays()."""

    def test_returns_empty_for_non_image_plotter(self, plotting_controller):
        """Test that non-image plotters return no overlays."""

        class TestOutputs(WorkflowOutputsBase):
            i_of_q: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['Q'], shape=[0], unit='counts'),
                    coords={'Q': sc.arange('Q', 0, unit='1/angstrom')},
                )
            )

        spec = WorkflowSpec(
            instrument='test',
            name='test_workflow',
            version=1,
            title='Test',
            description='Test',
            outputs=TestOutputs,
            params=None,
            group=REDUCTION,
        )

        overlays = plotting_controller.get_available_overlays(
            workflow_spec=spec, base_plotter_name='lines'
        )

        assert overlays == []

    def test_returns_empty_when_no_roi_outputs(self, plotting_controller):
        """Test that image plotter without ROI outputs returns no overlays."""

        class TestOutputs(WorkflowOutputsBase):
            detector: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['x', 'y'], shape=[0, 0], unit='counts'),
                    coords={
                        'x': sc.arange('x', 0, unit='m'),
                        'y': sc.arange('y', 0, unit='m'),
                    },
                )
            )

        spec = WorkflowSpec(
            instrument='test',
            name='test_workflow',
            version=1,
            title='Test',
            description='Test',
            outputs=TestOutputs,
            params=None,
            group=REDUCTION,
        )

        overlays = plotting_controller.get_available_overlays(
            workflow_spec=spec, base_plotter_name='image'
        )

        assert overlays == []

    def test_returns_rectangle_overlays_when_available(self, plotting_controller):
        """Test that rectangle overlays are returned when roi_rectangle exists."""

        class TestOutputs(WorkflowOutputsBase):
            detector: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['x', 'y'], shape=[0, 0], unit='counts'),
                    coords={
                        'x': sc.arange('x', 0, unit='m'),
                        'y': sc.arange('y', 0, unit='m'),
                    },
                )
            )
            roi_rectangle: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['bounds'], shape=[0]),
                    coords={
                        'roi_index': sc.arange('bounds', 0),
                        'x': sc.arange('bounds', 0, unit='m'),
                        'y': sc.arange('bounds', 0, unit='m'),
                    },
                )
            )

        spec = WorkflowSpec(
            instrument='test',
            name='test_workflow',
            version=1,
            title='Test',
            description='Test',
            outputs=TestOutputs,
            params=None,
            group=REDUCTION,
        )

        overlays = plotting_controller.get_available_overlays(
            workflow_spec=spec, base_plotter_name='image'
        )

        # Image only suggests readback (request is suggested by readback)
        assert len(overlays) == 1
        output_name, plotter_name, _ = overlays[0]
        assert output_name == 'roi_rectangle'
        assert plotter_name == 'rectangles_readback'

    def test_returns_polygon_overlays_when_available(self, plotting_controller):
        """Test that polygon overlays are returned when roi_polygon exists."""

        class TestOutputs(WorkflowOutputsBase):
            detector: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['x', 'y'], shape=[0, 0], unit='counts'),
                    coords={
                        'x': sc.arange('x', 0, unit='m'),
                        'y': sc.arange('y', 0, unit='m'),
                    },
                )
            )
            roi_polygon: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['vertex'], shape=[0]),
                    coords={
                        'roi_index': sc.arange('vertex', 0),
                        'x': sc.arange('vertex', 0, unit='m'),
                        'y': sc.arange('vertex', 0, unit='m'),
                    },
                )
            )

        spec = WorkflowSpec(
            instrument='test',
            name='test_workflow',
            version=1,
            title='Test',
            description='Test',
            outputs=TestOutputs,
            params=None,
            group=REDUCTION,
        )

        overlays = plotting_controller.get_available_overlays(
            workflow_spec=spec, base_plotter_name='image'
        )

        # Image only suggests readback (request is suggested by readback)
        assert len(overlays) == 1
        output_name, plotter_name, _ = overlays[0]
        assert output_name == 'roi_polygon'
        assert plotter_name == 'polygons_readback'

    def test_returns_all_overlays_when_both_roi_types_available(
        self, plotting_controller
    ):
        """Test all overlays returned when both rectangles and polygons exist."""

        class TestOutputs(WorkflowOutputsBase):
            detector: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['x', 'y'], shape=[0, 0], unit='counts'),
                    coords={
                        'x': sc.arange('x', 0, unit='m'),
                        'y': sc.arange('y', 0, unit='m'),
                    },
                )
            )
            roi_rectangle: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['bounds'], shape=[0]),
                    coords={
                        'roi_index': sc.arange('bounds', 0),
                        'x': sc.arange('bounds', 0, unit='m'),
                        'y': sc.arange('bounds', 0, unit='m'),
                    },
                )
            )
            roi_polygon: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['vertex'], shape=[0]),
                    coords={
                        'roi_index': sc.arange('vertex', 0),
                        'x': sc.arange('vertex', 0, unit='m'),
                        'y': sc.arange('vertex', 0, unit='m'),
                    },
                )
            )

        spec = WorkflowSpec(
            instrument='test',
            name='test_workflow',
            version=1,
            title='Test',
            description='Test',
            outputs=TestOutputs,
            params=None,
            group=REDUCTION,
        )

        overlays = plotting_controller.get_available_overlays(
            workflow_spec=spec, base_plotter_name='image'
        )

        # Image only suggests readbacks (requests are suggested by readbacks)
        assert len(overlays) == 2
        plotter_names = [o[1] for o in overlays]
        assert 'rectangles_readback' in plotter_names
        assert 'polygons_readback' in plotter_names

    def test_overlay_entries_have_correct_structure(self, plotting_controller):
        """Test overlay entries have (output_name, plotter_name, title) structure."""

        class TestOutputs(WorkflowOutputsBase):
            detector: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['x', 'y'], shape=[0, 0], unit='counts'),
                    coords={
                        'x': sc.arange('x', 0, unit='m'),
                        'y': sc.arange('y', 0, unit='m'),
                    },
                )
            )
            roi_rectangle: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['bounds'], shape=[0]),
                    coords={
                        'roi_index': sc.arange('bounds', 0),
                        'x': sc.arange('bounds', 0, unit='m'),
                        'y': sc.arange('bounds', 0, unit='m'),
                    },
                )
            )

        spec = WorkflowSpec(
            instrument='test',
            name='test_workflow',
            version=1,
            title='Test',
            description='Test',
            outputs=TestOutputs,
            params=None,
            group=REDUCTION,
        )

        overlays = plotting_controller.get_available_overlays(
            workflow_spec=spec, base_plotter_name='image'
        )

        # Check structure of first entry
        assert len(overlays[0]) == 3
        output_name, plotter_name, title = overlays[0]
        assert isinstance(output_name, str)
        assert isinstance(plotter_name, str)
        assert isinstance(title, str)
        # Title should be human readable (from PlotterSpec)
        # Image only suggests readback, not request
        assert title == 'ROI Rectangles (Readback)'

    def test_readback_layer_suggests_request_overlay(self, plotting_controller):
        """Test that rectangles_readback layer suggests rectangles_request."""

        class TestOutputs(WorkflowOutputsBase):
            roi_rectangle: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['bounds'], shape=[0]),
                    coords={
                        'roi_index': sc.arange('bounds', 0),
                        'x': sc.arange('bounds', 0, unit='m'),
                        'y': sc.arange('bounds', 0, unit='m'),
                    },
                )
            )

        spec = WorkflowSpec(
            instrument='test',
            name='test_workflow',
            version=1,
            title='Test',
            description='Test',
            outputs=TestOutputs,
            params=None,
            group=REDUCTION,
        )

        overlays = plotting_controller.get_available_overlays(
            workflow_spec=spec, base_plotter_name='rectangles_readback'
        )

        # Readback suggests request
        assert len(overlays) == 1
        output_name, plotter_name, title = overlays[0]
        assert output_name == 'roi_rectangle'
        assert plotter_name == 'rectangles_request'
        assert title == 'ROI Rectangles (Interactive)'


class TestOverlayPatterns:
    """Tests for OVERLAY_PATTERNS constant."""

    def test_image_has_readback_patterns(self):
        """Test that image plotter suggests readback overlays only."""
        from ess.livedata.dashboard.plotter_registry import OVERLAY_PATTERNS

        assert 'image' in OVERLAY_PATTERNS
        patterns = OVERLAY_PATTERNS['image']
        # Image only suggests readbacks (not requests)
        assert len(patterns) == 2
        plotter_names = [p[1] for p in patterns]
        assert 'rectangles_readback' in plotter_names
        assert 'polygons_readback' in plotter_names

    def test_readback_suggests_request(self):
        """Test that readback plotters suggest their corresponding request overlays."""
        from ess.livedata.dashboard.plotter_registry import OVERLAY_PATTERNS

        # rectangles_readback suggests rectangles_request
        assert 'rectangles_readback' in OVERLAY_PATTERNS
        rect_patterns = OVERLAY_PATTERNS['rectangles_readback']
        assert len(rect_patterns) == 1
        assert rect_patterns[0] == ('roi_rectangle', 'rectangles_request')

        # polygons_readback suggests polygons_request
        assert 'polygons_readback' in OVERLAY_PATTERNS
        poly_patterns = OVERLAY_PATTERNS['polygons_readback']
        assert len(poly_patterns) == 1
        assert poly_patterns[0] == ('roi_polygon', 'polygons_request')

    def test_patterns_have_correct_structure(self):
        """Test that patterns are (output_name, plotter_name) tuples."""
        from ess.livedata.dashboard.plotter_registry import OVERLAY_PATTERNS

        for patterns in OVERLAY_PATTERNS.values():
            for pattern in patterns:
                assert len(pattern) == 2
                output_name, plotter_name = pattern
                assert isinstance(output_name, str)
                assert isinstance(plotter_name, str)

    def test_overlay_chain_enforces_order(self):
        """Test that overlay chain enforces image -> readback -> request order."""
        from ess.livedata.dashboard.plotter_registry import OVERLAY_PATTERNS

        # Image can only go to readback
        image_overlays = [p[1] for p in OVERLAY_PATTERNS['image']]
        assert all('readback' in name for name in image_overlays)
        assert not any('request' in name for name in image_overlays)

        # Readback can only go to request
        for key in ['rectangles_readback', 'polygons_readback']:
            if key in OVERLAY_PATTERNS:
                overlays = [p[1] for p in OVERLAY_PATTERNS[key]]
                assert all('request' in name for name in overlays)


class TestOutputViewSupportsWindowing:
    def test_true_when_view_has_both_streams(self) -> None:
        from typing import ClassVar

        from ess.livedata.config.workflow_spec import OutputView

        class Outputs(WorkflowOutputsBase):
            output_views: ClassVar[tuple[OutputView, ...]] = (
                OutputView(
                    name='image',
                    title='Image',
                    streams={'since_start': 'cumulative', 'per_update': 'current'},
                ),
            )

            current: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['x'], shape=[0]),
                )
            )
            cumulative: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['x'], shape=[0]),
                )
            )

        spec = WorkflowSpec(
            instrument='test',
            name='wf',
            version=1,
            title='T',
            description='D',
            outputs=Outputs,
            params=None,
            group=REDUCTION,
        )
        assert output_view_supports_windowing(spec, 'image') is True

    def test_false_for_cumulative_only_view(self) -> None:
        from typing import ClassVar

        from ess.livedata.config.workflow_spec import OutputView

        class Outputs(WorkflowOutputsBase):
            output_views: ClassVar[tuple[OutputView, ...]] = (
                OutputView(
                    name='i_of_q',
                    title='I(Q)',
                    streams={'since_start': 'i_of_q'},
                ),
            )
            i_of_q: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(
                    sc.zeros(dims=['Q'], shape=[0], unit='counts'),
                )
            )

        spec = WorkflowSpec(
            instrument='test',
            name='wf',
            version=1,
            title='T',
            description='D',
            outputs=Outputs,
            params=None,
            group=REDUCTION,
        )
        assert output_view_supports_windowing(spec, 'i_of_q') is False

    def test_true_for_per_update_only_view(self) -> None:
        # A per_update-only view still supports the duration/aggregation controls;
        # the since_start mode is rejected at config time, not by hiding controls.
        from typing import ClassVar

        from ess.livedata.config.workflow_spec import OutputView

        class Outputs(WorkflowOutputsBase):
            output_views: ClassVar[tuple[OutputView, ...]] = (
                OutputView(
                    name='total',
                    title='Total',
                    streams={'per_update': 'counts_total'},
                ),
            )
            counts_total: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.scalar(0, unit='counts'))
            )

        spec = WorkflowSpec(
            instrument='test',
            name='wf',
            version=1,
            title='T',
            description='D',
            outputs=Outputs,
            params=None,
            group=REDUCTION,
        )
        assert output_view_supports_windowing(spec, 'total') is True

    def test_true_when_view_unknown(self) -> None:
        class Outputs(WorkflowOutputsBase):
            result: sc.DataArray = pydantic.Field(title='Result')

        spec = WorkflowSpec(
            instrument='test',
            name='wf',
            version=1,
            title='T',
            description='D',
            outputs=Outputs,
            params=None,
            group=REDUCTION,
        )
        # Default (fallback) view per field uses `since_start`, so no windowing.
        assert output_view_supports_windowing(spec, 'result') is False
        # Unknown view name → True (be permissive).
        assert output_view_supports_windowing(spec, 'nonexistent') is True


class TestSinceStartAvailable:
    def _spec(self, outputs: type[WorkflowOutputsBase]) -> WorkflowSpec:
        return WorkflowSpec(
            instrument='test',
            name='wf',
            version=1,
            title='T',
            description='D',
            outputs=outputs,
            params=None,
            group=REDUCTION,
        )

    def test_true_when_view_has_since_start_stream(self) -> None:
        from typing import ClassVar

        from ess.livedata.config.workflow_spec import OutputView

        class Outputs(WorkflowOutputsBase):
            output_views: ClassVar[tuple[OutputView, ...]] = (
                OutputView(
                    name='total',
                    title='Total',
                    streams={'since_start': 'cumulative', 'per_update': 'current'},
                ),
            )
            cumulative: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.scalar(0, unit='counts'))
            )
            current: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.scalar(0, unit='counts'))
            )

        spec = self._spec(Outputs)
        assert since_start_available(spec, 'total') is True

    def test_false_for_per_update_only_view(self) -> None:
        from typing import ClassVar

        from ess.livedata.config.workflow_spec import OutputView

        class Outputs(WorkflowOutputsBase):
            output_views: ClassVar[tuple[OutputView, ...]] = (
                OutputView(
                    name='total', title='Total', streams={'per_update': 'counts_total'}
                ),
            )
            counts_total: sc.DataArray = pydantic.Field(
                default_factory=lambda: sc.DataArray(sc.scalar(0, unit='counts'))
            )

        spec = self._spec(Outputs)
        assert since_start_available(spec, 'total') is False
        # Unknown view name → permissive.
        assert since_start_available(spec, 'nonexistent') is True


class TestCreateExtractorsFromParams:
    """Verify the new ``since_start`` window mode uses LatestValueExtractor."""

    def _make_keys(self) -> list:
        from ess.livedata.config.workflow_spec import DataKey

        wf_id = WorkflowId(instrument='test', name='wf', version=1)
        return [
            DataKey(
                workflow_id=wf_id,
                source_name='src',
                output_name='current',
            )
        ]

    def test_since_start_uses_latest_value_extractor(self) -> None:
        import uuid as _uuid  # noqa: F401  (avoid masking)

        from ess.livedata.dashboard.extractors import LatestValueExtractor
        from ess.livedata.dashboard.plot_params import TimeWindowMode, TimeWindowParams
        from ess.livedata.dashboard.plotting_controller import (
            create_extractors_from_params,
        )

        keys = self._make_keys()
        params = TimeWindowParams(mode=TimeWindowMode.since_start)
        extractors = create_extractors_from_params(keys, params)
        assert all(isinstance(e, LatestValueExtractor) for e in extractors.values())

    def test_window_with_zero_duration_uses_latest_value_extractor(self) -> None:
        from ess.livedata.dashboard.extractors import LatestValueExtractor
        from ess.livedata.dashboard.plot_params import TimeWindowMode, TimeWindowParams
        from ess.livedata.dashboard.plotting_controller import (
            create_extractors_from_params,
        )

        keys = self._make_keys()
        params = TimeWindowParams(
            mode=TimeWindowMode.window, window_duration_seconds=0.0
        )
        extractors = create_extractors_from_params(keys, params)
        assert all(isinstance(e, LatestValueExtractor) for e in extractors.values())

    def test_window_uses_window_aggregating_extractor(self) -> None:
        from ess.livedata.dashboard.extractors import WindowAggregatingExtractor
        from ess.livedata.dashboard.plot_params import TimeWindowMode, TimeWindowParams
        from ess.livedata.dashboard.plotting_controller import (
            create_extractors_from_params,
        )

        keys = self._make_keys()
        params = TimeWindowParams(
            mode=TimeWindowMode.window, window_duration_seconds=5.0
        )
        extractors = create_extractors_from_params(keys, params)
        assert all(
            isinstance(e, WindowAggregatingExtractor) for e in extractors.values()
        )


class TestResolveFieldName:
    """Verify ``resolve_field_name`` maps (view, role) to the backing field."""

    def _spec(self) -> WorkflowSpec:
        from typing import ClassVar

        from ess.livedata.config.workflow_spec import OutputView

        class Outputs(WorkflowOutputsBase):
            output_views: ClassVar[tuple[OutputView, ...]] = (
                OutputView(
                    name='image',
                    title='Image',
                    streams={'since_start': 'cumulative', 'per_update': 'current'},
                ),
                OutputView(
                    name='total_counts',
                    title='Total',
                    streams={'per_update': 'counts_total'},
                ),
            )
            cumulative: sc.DataArray = pydantic.Field()
            current: sc.DataArray = pydantic.Field()
            counts_total: sc.DataArray = pydantic.Field()

        return WorkflowSpec(
            instrument='test',
            name='wf',
            version=1,
            title='T',
            description='D',
            outputs=Outputs,
            params=None,
            group=REDUCTION,
        )

    def test_since_start_role_resolves_to_cumulative(self) -> None:
        from ess.livedata.dashboard.plot_orchestrator import resolve_field_name

        assert (
            resolve_field_name(self._spec(), 'image', role='since_start')
            == 'cumulative'
        )

    def test_per_update_role_resolves_to_current(self) -> None:
        from ess.livedata.dashboard.plot_orchestrator import resolve_field_name

        assert resolve_field_name(self._spec(), 'image', role='per_update') == 'current'

    def test_falls_back_to_other_role_when_requested_missing(self) -> None:
        """``total_counts`` only declares ``per_update``; asking for
        ``since_start`` falls back to it."""
        from ess.livedata.dashboard.plot_orchestrator import resolve_field_name

        assert (
            resolve_field_name(self._spec(), 'total_counts', role='since_start')
            == 'counts_total'
        )

    def test_falls_back_to_view_name_when_no_view_declared(self) -> None:
        """For unannotated outputs classes, the view name is treated as the
        backing field name."""
        from ess.livedata.dashboard.plot_orchestrator import resolve_field_name

        class BareOutputs(WorkflowOutputsBase):
            result: sc.DataArray = pydantic.Field()

        spec = WorkflowSpec(
            instrument='test',
            name='wf',
            version=1,
            title='T',
            description='D',
            outputs=BareOutputs,
            params=None,
            group=REDUCTION,
        )
        # The auto-generated view maps `result` via since_start. Asking for
        # per_update on this view falls back to since_start.
        assert resolve_field_name(spec, 'result', role='per_update') == 'result'
