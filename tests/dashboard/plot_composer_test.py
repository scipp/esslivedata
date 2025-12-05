# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the plot composition system."""

from __future__ import annotations

import uuid
from typing import Any

import holoviews as hv
import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import JobNumber, ResultKey, WorkflowId
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.data_subscriber import Pipe
from ess.livedata.dashboard.plot_composer import (
    HLinesFactory,
    LayerConfig,
    LayerState,
    PipelineSource,
    PlotComposer,
    StaticSource,
    VLinesFactory,
)
from ess.livedata.dashboard.stream_manager import StreamManager

hv.extension('bokeh')


# =============================================================================
# Fake implementations for testing
# =============================================================================


class FakePipe(Pipe):
    """Fake implementation of Pipe for testing."""

    def __init__(self, data: Any) -> None:
        self.send_calls: list[Any] = []
        self.data: Any = data

    def send(self, data: Any) -> None:
        self.send_calls.append(data)
        self.data = data


class FakePipeFactory:
    """Fake pipe factory for testing."""

    def __init__(self) -> None:
        self.call_count = 0
        self.created_pipes: list[FakePipe] = []

    def __call__(self, data: Any) -> FakePipe:
        self.call_count += 1
        pipe = FakePipe(data)
        self.created_pipes.append(pipe)
        return pipe


class FakePlotter:
    """Fake plotter for testing composition."""

    def __init__(self, name: str = "fake"):
        self.name = name
        self._kdims = None
        self.call_count = 0

    def initialize_from_data(self, data: Any) -> None:
        pass

    @property
    def kdims(self) -> list[hv.Dimension] | None:
        return self._kdims

    def __call__(self, data: dict[ResultKey, Any]) -> hv.Element:
        self.call_count += 1
        return hv.Curve([(0, 0), (1, 1)], label=self.name)


class FakeDataRequirements:
    """Fake DataRequirements for testing."""

    required_extractor = None


class FakePlotterSpec:
    """Fake PlotterSpec for testing."""

    def __init__(self):
        from ess.livedata.dashboard.plot_params import PlotParams1d

        self.params = PlotParams1d
        self.data_requirements = FakeDataRequirements()


class FakePlotterRegistry:
    """Fake plotter registry for testing."""

    def __init__(self):
        self._plotters: dict[str, FakePlotter] = {}
        self._specs: dict[str, FakePlotterSpec] = {}

    def register(self, name: str, plotter: FakePlotter | None = None):
        self._plotters[name] = plotter or FakePlotter(name)
        self._specs[name] = FakePlotterSpec()

    def get_spec(self, name: str) -> FakePlotterSpec:
        if name not in self._specs:
            self._specs[name] = FakePlotterSpec()
        return self._specs[name]

    def create_plotter(self, name: str, params: Any) -> FakePlotter:
        if name not in self._plotters:
            self._plotters[name] = FakePlotter(name)
        return self._plotters[name]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def data_service() -> DataService:
    """Real DataService instance for testing."""
    return DataService()


@pytest.fixture
def fake_pipe_factory() -> FakePipeFactory:
    """Fake pipe factory that creates FakePipe instances."""
    return FakePipeFactory()


@pytest.fixture
def stream_manager(data_service, fake_pipe_factory) -> StreamManager:
    """StreamManager with fake pipe factory."""
    return StreamManager(data_service=data_service, pipe_factory=fake_pipe_factory)


@pytest.fixture
def fake_registry() -> FakePlotterRegistry:
    """Fake plotter registry."""
    registry = FakePlotterRegistry()
    registry.register("lines")
    registry.register("image")
    return registry


@pytest.fixture
def workflow_id() -> WorkflowId:
    """Sample workflow ID."""
    return WorkflowId(instrument="test", namespace="ns", name="wf", version=1)


@pytest.fixture
def job_number() -> JobNumber:
    """Sample job number."""
    return uuid.uuid4()


@pytest.fixture
def sample_1d_data() -> sc.DataArray:
    """Sample 1D data array for testing."""
    return sc.DataArray(
        data=sc.array(dims=['x'], values=[1.0, 2.0, 3.0]),
        coords={'x': sc.array(dims=['x'], values=[10.0, 20.0, 30.0])},
    )


# =============================================================================
# Tests for Data Sources
# =============================================================================


class TestPipelineSource:
    """Test cases for PipelineSource."""

    def test_get_result_keys_single_source(self, workflow_id, job_number):
        """Test that get_result_keys returns correct keys for single source."""
        source = PipelineSource(
            workflow_id=workflow_id,
            job_number=job_number,
            source_names=["detector1"],
            output_name="result",
        )

        keys = source.get_result_keys()

        assert len(keys) == 1
        assert keys[0].workflow_id == workflow_id
        assert keys[0].job_id.source_name == "detector1"
        assert keys[0].output_name == "result"

    def test_get_result_keys_multiple_sources(self, workflow_id, job_number):
        """Test that get_result_keys returns keys for all sources."""
        source = PipelineSource(
            workflow_id=workflow_id,
            job_number=job_number,
            source_names=["det1", "det2", "det3"],
            output_name="image",
        )

        keys = source.get_result_keys()

        assert len(keys) == 3
        assert {k.job_id.source_name for k in keys} == {"det1", "det2", "det3"}


class TestStaticSource:
    """Test cases for StaticSource."""

    def test_get_result_keys_returns_empty(self):
        """Test that StaticSource returns empty list for result keys."""
        source = StaticSource(data=[1.0, 2.0, 3.0])
        assert source.get_result_keys() == []

    def test_stores_data(self):
        """Test that StaticSource stores the provided data."""
        data = [1.5, 2.5, 3.5]
        source = StaticSource(data=data)
        assert source.data == data


# =============================================================================
# Tests for Element Factories
# =============================================================================


class TestVLinesFactory:
    """Test cases for VLinesFactory."""

    def test_create_element_empty_data(self):
        """Test creating VLines with empty data."""
        factory = VLinesFactory(color="red")
        result = factory.create_element([])
        assert isinstance(result, hv.Overlay)
        assert len(result) == 0

    def test_create_element_none_data(self):
        """Test creating VLines with None data."""
        factory = VLinesFactory(color="red")
        result = factory.create_element(None)
        assert isinstance(result, hv.Overlay)
        assert len(result) == 0

    def test_create_element_with_positions(self):
        """Test creating VLines with position data."""
        factory = VLinesFactory(color="blue", line_dash="dashed")
        positions = [1.0, 2.5, 4.0]
        result = factory.create_element(positions)

        assert isinstance(result, hv.Overlay)
        assert len(result) == 3
        # Each element should be a VLine
        for element in result:
            assert isinstance(element, hv.VLine)


class TestHLinesFactory:
    """Test cases for HLinesFactory."""

    def test_create_element_empty_data(self):
        """Test creating HLines with empty data."""
        factory = HLinesFactory(color="red")
        result = factory.create_element([])
        assert isinstance(result, hv.Overlay)
        assert len(result) == 0

    def test_create_element_with_positions(self):
        """Test creating HLines with position data."""
        factory = HLinesFactory(color="green")
        positions = [0.5, 1.5]
        result = factory.create_element(positions)

        assert isinstance(result, hv.Overlay)
        assert len(result) == 2


# =============================================================================
# Tests for LayerConfig
# =============================================================================


class TestLayerConfig:
    """Test cases for LayerConfig."""

    def test_create_with_pipeline_source(self, workflow_id, job_number):
        """Test creating LayerConfig with PipelineSource."""
        source = PipelineSource(
            workflow_id=workflow_id,
            job_number=job_number,
            source_names=["det1"],
        )
        config = LayerConfig(name="main", element="lines", source=source)

        assert config.name == "main"
        assert config.element == "lines"
        assert config.source == source

    def test_create_with_static_source(self):
        """Test creating LayerConfig with StaticSource."""
        source = StaticSource(data=[1.0, 2.0, 3.0])
        config = LayerConfig(
            name="peaks",
            element="vlines",
            source=source,
            params={"color": "red"},
        )

        assert config.name == "peaks"
        assert config.element == "vlines"
        assert config.params == {"color": "red"}


# =============================================================================
# Tests for PlotComposer
# =============================================================================


class TestPlotComposerStaticLayers:
    """Test cases for PlotComposer with static layers."""

    def test_add_static_vlines_layer(self, stream_manager, fake_registry):
        """Test adding a static VLines layer."""
        composer = PlotComposer(
            stream_manager=stream_manager, plotter_registry=fake_registry
        )

        config = LayerConfig(
            name="peaks",
            element="vlines",
            source=StaticSource(data=[1.0, 2.5, 4.0]),
            params={"color": "red", "line_dash": "dashed"},
        )

        composer.add_static_layer(config)

        assert "peaks" in composer.get_layer_names()
        assert composer.is_layer_ready("peaks")

    def test_add_static_hlines_layer(self, stream_manager, fake_registry):
        """Test adding a static HLines layer."""
        composer = PlotComposer(
            stream_manager=stream_manager, plotter_registry=fake_registry
        )

        config = LayerConfig(
            name="thresholds",
            element="hlines",
            source=StaticSource(data=[0.5, 1.5]),
            params={"color": "green"},
        )

        composer.add_static_layer(config)

        assert "thresholds" in composer.get_layer_names()

    def test_update_static_layer(self, stream_manager, fake_registry):
        """Test updating data for a static layer."""
        composer = PlotComposer(
            stream_manager=stream_manager, plotter_registry=fake_registry
        )

        config = LayerConfig(
            name="markers",
            element="vlines",
            source=StaticSource(data=[1.0]),
        )
        composer.add_static_layer(config)

        # Update the layer data
        composer.update_static_layer("markers", [2.0, 3.0, 4.0])

        # Should not raise and layer should still be ready
        assert composer.is_layer_ready("markers")

    def test_update_nonexistent_layer_raises(self, stream_manager, fake_registry):
        """Test that updating nonexistent layer raises KeyError."""
        composer = PlotComposer(
            stream_manager=stream_manager, plotter_registry=fake_registry
        )

        with pytest.raises(KeyError):
            composer.update_static_layer("nonexistent", [1.0])

    def test_add_static_layer_invalid_type_raises(self, stream_manager, fake_registry):
        """Test that adding static layer with unknown element type raises."""
        composer = PlotComposer(
            stream_manager=stream_manager, plotter_registry=fake_registry
        )

        config = LayerConfig(
            name="bad",
            element="unknown_element",
            source=StaticSource(data=[1.0]),
        )

        with pytest.raises(ValueError, match="Unknown static element type"):
            composer.add_static_layer(config)


class TestPlotComposerRemoveLayer:
    """Test cases for removing layers."""

    def test_remove_static_layer(self, stream_manager, fake_registry):
        """Test removing a static layer."""
        composer = PlotComposer(
            stream_manager=stream_manager, plotter_registry=fake_registry
        )

        config = LayerConfig(
            name="peaks",
            element="vlines",
            source=StaticSource(data=[1.0]),
        )
        composer.add_static_layer(config)
        assert "peaks" in composer.get_layer_names()

        composer.remove_layer("peaks")
        assert "peaks" not in composer.get_layer_names()

    def test_remove_nonexistent_layer_no_error(self, stream_manager, fake_registry):
        """Test that removing nonexistent layer doesn't raise."""
        composer = PlotComposer(
            stream_manager=stream_manager, plotter_registry=fake_registry
        )
        # Should not raise, just log warning
        composer.remove_layer("nonexistent")


class TestPlotComposerComposition:
    """Test cases for layer composition."""

    def test_get_composition_empty(self, stream_manager, fake_registry):
        """Test getting composition with no layers."""
        composer = PlotComposer(
            stream_manager=stream_manager, plotter_registry=fake_registry
        )

        result = composer.get_composition()

        # Should return empty DynamicMap
        assert isinstance(result, hv.DynamicMap)

    def test_get_composition_single_static_layer(self, stream_manager, fake_registry):
        """Test getting composition with single static layer."""
        composer = PlotComposer(
            stream_manager=stream_manager, plotter_registry=fake_registry
        )

        config = LayerConfig(
            name="peaks",
            element="vlines",
            source=StaticSource(data=[1.0, 2.0]),
        )
        composer.add_static_layer(config)

        result = composer.get_composition()

        # Single layer returns its DynamicMap directly
        assert isinstance(result, hv.DynamicMap)

    def test_get_composition_multiple_static_layers(
        self, stream_manager, fake_registry
    ):
        """Test getting composition with multiple static layers."""
        composer = PlotComposer(
            stream_manager=stream_manager, plotter_registry=fake_registry
        )

        config1 = LayerConfig(
            name="vlines",
            element="vlines",
            source=StaticSource(data=[1.0]),
        )
        config2 = LayerConfig(
            name="hlines",
            element="hlines",
            source=StaticSource(data=[0.5]),
        )
        composer.add_static_layer(config1)
        composer.add_static_layer(config2)

        result = composer.get_composition()

        # Multiple layers compose via * operator
        # Result is an Overlay (from DynamicMap composition)
        assert isinstance(result, hv.DynamicMap | hv.Overlay)

    def test_get_layer_names(self, stream_manager, fake_registry):
        """Test getting list of layer names."""
        composer = PlotComposer(
            stream_manager=stream_manager, plotter_registry=fake_registry
        )

        config1 = LayerConfig(
            name="layer1", element="vlines", source=StaticSource(data=[1.0])
        )
        config2 = LayerConfig(
            name="layer2", element="hlines", source=StaticSource(data=[0.5])
        )
        composer.add_static_layer(config1)
        composer.add_static_layer(config2)

        names = composer.get_layer_names()
        assert set(names) == {"layer1", "layer2"}


class TestPlotComposerPipelineLayers:
    """Test cases for PlotComposer with pipeline layers."""

    def test_add_pipeline_layer_with_source_check(
        self, stream_manager, fake_registry, workflow_id, job_number
    ):
        """Test adding a pipeline layer checks source type."""
        composer = PlotComposer(
            stream_manager=stream_manager, plotter_registry=fake_registry
        )

        # Using StaticSource with pipeline layer should raise TypeError
        config = LayerConfig(
            name="bad",
            element="lines",
            source=StaticSource(data=[1.0]),
        )

        with pytest.raises(TypeError, match="Expected PipelineSource"):
            composer.add_pipeline_layer(config)

    def test_add_static_layer_with_pipeline_source_raises(
        self, stream_manager, fake_registry, workflow_id, job_number
    ):
        """Test that add_static_layer rejects pipeline sources."""
        composer = PlotComposer(
            stream_manager=stream_manager, plotter_registry=fake_registry
        )

        config = LayerConfig(
            name="bad",
            element="vlines",
            source=PipelineSource(
                workflow_id=workflow_id,
                job_number=job_number,
                source_names=["det1"],
            ),
        )

        with pytest.raises(TypeError, match="Expected StaticSource"):
            composer.add_static_layer(config)

    def test_pipeline_layer_not_ready_until_data(
        self, stream_manager, fake_registry, workflow_id, job_number
    ):
        """Test that pipeline layer is not ready until data arrives."""
        composer = PlotComposer(
            stream_manager=stream_manager, plotter_registry=fake_registry
        )

        config = LayerConfig(
            name="data",
            element="lines",
            source=PipelineSource(
                workflow_id=workflow_id,
                job_number=job_number,
                source_names=["det1"],
            ),
        )
        composer.add_pipeline_layer(config)

        # Layer exists but is not ready
        assert "data" in composer.get_layer_names()
        assert not composer.is_layer_ready("data")
        assert "data" not in composer.get_ready_layer_names()


class TestPlotComposerReplacement:
    """Test cases for layer replacement behavior."""

    def test_adding_layer_with_same_name_replaces(self, stream_manager, fake_registry):
        """Test that adding a layer with same name replaces existing."""
        composer = PlotComposer(
            stream_manager=stream_manager, plotter_registry=fake_registry
        )

        config1 = LayerConfig(
            name="markers",
            element="vlines",
            source=StaticSource(data=[1.0]),
        )
        config2 = LayerConfig(
            name="markers",
            element="hlines",
            source=StaticSource(data=[2.0]),
        )

        composer.add_static_layer(config1)
        composer.add_static_layer(config2)

        # Should only have one layer
        assert composer.get_layer_names() == ["markers"]


class TestLayerState:
    """Test cases for LayerState dataclass."""

    def test_default_values(self):
        """Test LayerState default values."""
        config = LayerConfig(
            name="test", element="vlines", source=StaticSource(data=[])
        )
        state = LayerState(config=config)

        assert state.config == config
        assert state.pipe is None
        assert state.dmap is None
        assert state.factory is None
