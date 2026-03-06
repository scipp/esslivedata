# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pydantic
import pytest
import scipp as sc

from ess.livedata.config.instrument import (
    Instrument,
    InstrumentRegistry,
    SourceMetadata,
)
from ess.livedata.config.workflow_spec import WorkflowOutputsBase
from ess.livedata.handlers.workflow_factory import (
    Workflow,
    WorkflowFactory,
)


class SimpleTestOutputs(WorkflowOutputsBase):
    """Simple outputs model for testing."""

    result: sc.DataArray = pydantic.Field(title='Result')


class TestInstrumentRegistry:
    def test_registry_creation(self):
        """Test that registry can be created and behaves like a dict."""
        registry = InstrumentRegistry()
        assert len(registry) == 0

    def test_register_instrument(self):
        """Test registering an instrument."""
        registry = InstrumentRegistry()
        instrument = Instrument(name="test_instrument")

        registry.register(instrument)

        assert len(registry) == 1
        assert "test_instrument" in registry
        assert registry["test_instrument"] is instrument

    def test_register_duplicate_instrument_raises_error(self):
        """Test that registering duplicate instrument names raises an error."""
        registry = InstrumentRegistry()
        instrument1 = Instrument(name="test_instrument")
        instrument2 = Instrument(name="test_instrument")

        registry.register(instrument1)

        with pytest.raises(
            ValueError, match="Instrument test_instrument is already registered"
        ):
            registry.register(instrument2)

    def test_registry_dict_operations(self):
        """Test that registry supports standard dict operations."""
        registry = InstrumentRegistry()
        instrument1 = Instrument(name="instrument1")
        instrument2 = Instrument(name="instrument2")

        registry.register(instrument1)
        registry.register(instrument2)

        assert len(registry) == 2
        assert list(registry.keys()) == ["instrument1", "instrument2"]
        assert list(registry.values()) == [instrument1, instrument2]
        assert ("instrument1", instrument1) in registry.items()

    def test_registry_access_nonexistent_instrument(self):
        """Test accessing non-existent instrument raises KeyError."""
        registry = InstrumentRegistry()

        with pytest.raises(KeyError):
            registry["nonexistent"]


class TestInstrument:
    def test_instrument_creation_with_defaults(self):
        """Test creating instrument with default values."""
        instrument = Instrument(name="test_instrument")

        assert instrument.name == "test_instrument"
        assert isinstance(instrument.workflow_factory, WorkflowFactory)
        assert instrument.f144_attribute_registry == {}
        assert instrument.active_namespace is None
        assert instrument.detector_names == []

    def test_instrument_creation_with_custom_values(self):
        """Test creating instrument with custom values."""
        custom_factory = WorkflowFactory()
        f144_registry = {"attr1": {"key": "value"}}

        instrument = Instrument(
            name="custom_instrument",
            workflow_factory=custom_factory,
            f144_attribute_registry=f144_registry,
            active_namespace="custom_namespace",
        )

        assert instrument.name == "custom_instrument"
        assert instrument.workflow_factory is custom_factory
        assert instrument.f144_attribute_registry == f144_registry
        assert instrument.active_namespace == "custom_namespace"

    def test_configure_detector_with_explicit_number(self):
        """Test configuring detector with explicit detector number."""
        instrument = Instrument(name="test_instrument", detector_names=["detector1"])
        detector_number = sc.array(dims=['detector'], values=[1, 2, 3])

        instrument.configure_detector("detector1", detector_number)

        assert "detector1" in instrument.detector_names
        assert sc.identical(
            instrument.get_detector_number("detector1"), detector_number
        )

    def test_configure_detector_fails_if_not_in_detector_names(self):
        """Test that configure_detector fails if detector not in detector_names."""
        instrument = Instrument(name="test_instrument")

        with pytest.raises(ValueError, match="not in declared detector_names"):
            instrument.configure_detector(
                "detector1", sc.array(dims=['detector'], values=[1, 2, 3])
            )

    def test_get_detector_number_for_nonexistent_detector(self):
        """Test getting detector number for non-existent detector raises KeyError."""
        instrument = Instrument(name="test_instrument")

        with pytest.raises(KeyError):
            instrument.get_detector_number("nonexistent_detector")

    def test_load_factories_loads_detector_numbers_from_nexus_file(self):
        """Test that load_factories automatically loads detector_numbers from nexus."""
        instrument = Instrument(name="dream", detector_names=["mantle_detector"])
        # Before load_factories, detector_number is not available
        with pytest.raises(KeyError):
            instrument.get_detector_number("mantle_detector")
        # load_factories should load it from nexus
        instrument.load_factories()
        detector_number = instrument.get_detector_number("mantle_detector")
        assert isinstance(detector_number, sc.Variable)

    def test_multiple_detectors(self):
        """Test managing multiple detectors."""
        instrument = Instrument(
            name="test_instrument", detector_names=["detector1", "detector2"]
        )
        detector1_number = sc.array(dims=['detector'], values=[1, 2])
        detector2_number = sc.array(dims=['detector'], values=[3, 4, 5])

        instrument.configure_detector("detector1", detector1_number)
        instrument.configure_detector("detector2", detector2_number)

        assert len(instrument.detector_names) == 2
        assert "detector1" in instrument.detector_names
        assert "detector2" in instrument.detector_names
        assert sc.identical(
            instrument.get_detector_number("detector1"), detector1_number
        )
        assert sc.identical(
            instrument.get_detector_number("detector2"), detector2_number
        )

    def test_register_spec_and_attach_factory(self):
        """Test two-phase spec registration and factory attachment."""
        instrument = Instrument(name="test_instrument")

        # Phase 1: Register spec
        handle = instrument.register_spec(
            namespace="test_namespace",
            name="test_workflow",
            version=1,
            title="Test Workflow",
            description="A test workflow",
            source_names=["source1", "source2"],
            outputs=SimpleTestOutputs,
        )

        # Verify spec is registered
        specs = instrument.workflow_factory
        assert len(specs) == 1
        spec = next(iter(specs.values()))
        assert spec.instrument == "test_instrument"
        assert spec.namespace == "test_namespace"
        assert spec.name == "test_workflow"
        assert spec.version == 1
        assert spec.title == "Test Workflow"
        assert spec.description == "A test workflow"
        assert spec.source_names == ["source1", "source2"]
        assert spec.aux_sources is None
        assert spec.outputs is SimpleTestOutputs

        # Phase 2: Attach factory
        def simple_processor_factory(source_name: str) -> Workflow:
            # Return a mock processor for testing
            class MockProcessor(Workflow):
                def accumulate(self, data, *, start_time: int, end_time: int) -> None:
                    pass

                def finalize(self):
                    return {"source": source_name}

                def clear(self) -> None:
                    pass

            return MockProcessor()

        # Attach factory using decorator
        decorator = handle.attach_factory()
        registered_factory = decorator(simple_processor_factory)

        # Verify the factory is returned unchanged
        assert registered_factory is simple_processor_factory

    def test_register_spec_with_defaults(self):
        """Test spec registration with default values."""
        instrument = Instrument(name="test_instrument")

        instrument.register_spec(
            name="minimal_workflow",
            version=1,
            title="Minimal Workflow",
            outputs=SimpleTestOutputs,
        )

        specs = instrument.workflow_factory
        assert len(specs) == 1
        spec = next(iter(specs.values()))
        assert spec.namespace == "data_reduction"  # default
        assert spec.description == ""  # default
        assert spec.source_names == []  # default
        assert spec.aux_sources is None  # default
        assert spec.outputs is SimpleTestOutputs

    def test_register_spec_with_aux_sources_explicit(self):
        """Test that aux_sources can be set explicitly."""
        from typing import Literal

        import pydantic

        instrument = Instrument(name="test_instrument")

        class AuxSourcesModel(pydantic.BaseModel):
            monitor1: Literal['monitor1'] = 'monitor1'
            aux_stream: Literal['aux_stream'] = 'aux_stream'

        instrument.register_spec(
            name="workflow_with_aux",
            version=1,
            title="Workflow with Aux Sources",
            aux_sources=AuxSourcesModel,
            outputs=SimpleTestOutputs,
        )

        specs = instrument.workflow_factory
        assert len(specs) == 1
        spec = next(iter(specs.values()))

        # aux_sources should be set explicitly
        assert spec.aux_sources is AuxSourcesModel

        # Verify it's a Pydantic model with the expected fields
        model_instance = spec.aux_sources()
        assert hasattr(model_instance, 'monitor1')
        assert hasattr(model_instance, 'aux_stream')

        # The default values should match the source names
        assert model_instance.monitor1 == 'monitor1'
        assert model_instance.aux_stream == 'aux_stream'

    def test_multiple_spec_registrations(self):
        """Test registering multiple specs."""
        instrument = Instrument(name="test_instrument")

        # Register two specs
        instrument.register_spec(
            name="workflow1", version=1, title="Workflow 1", outputs=SimpleTestOutputs
        )
        instrument.register_spec(
            name="workflow2", version=1, title="Workflow 2", outputs=SimpleTestOutputs
        )

        specs = instrument.workflow_factory
        assert len(specs) == 2

        workflow_names = {spec.name for spec in specs.values()}
        assert workflow_names == {"workflow1", "workflow2"}


class TestInstrumentRegisterSpec:
    """Test the new register_spec() convenience method for two-phase registration."""

    def test_register_spec_returns_handle(self):
        """Test that register_spec() returns a SpecHandle."""
        from ess.livedata.handlers.workflow_factory import SpecHandle

        instrument = Instrument(name="test_instrument")

        handle = instrument.register_spec(
            name="test_workflow",
            version=1,
            title="Test Workflow",
            outputs=SimpleTestOutputs,
        )

        assert isinstance(handle, SpecHandle)

    def test_register_spec_with_all_params(self):
        """Test register_spec() with all parameters."""
        import pydantic

        from ess.livedata.handlers.workflow_factory import SpecHandle

        class MyParams(pydantic.BaseModel):
            value: int

        class MyAuxSources(pydantic.BaseModel):
            monitor: str

        class MyOutputs(pydantic.BaseModel):
            result: int

        instrument = Instrument(name="test_instrument")

        handle = instrument.register_spec(
            namespace="custom_namespace",
            name="test_workflow",
            version=1,
            title="Test Workflow",
            description="Test description",
            source_names=["source1", "source2"],
            params=MyParams,
            aux_sources=MyAuxSources,
            outputs=MyOutputs,
        )

        assert isinstance(handle, SpecHandle)

        # Verify spec was registered
        spec_id = handle.workflow_id
        spec = instrument.workflow_factory[spec_id]
        assert spec.instrument == "test_instrument"
        assert spec.namespace == "custom_namespace"
        assert spec.name == "test_workflow"
        assert spec.version == 1
        assert spec.title == "Test Workflow"
        assert spec.description == "Test description"
        assert spec.source_names == ["source1", "source2"]
        assert spec.params is MyParams
        assert spec.aux_sources is MyAuxSources
        assert spec.outputs is MyOutputs

    def test_register_spec_with_defaults(self):
        """Test register_spec() with default values."""
        instrument = Instrument(name="test_instrument")

        handle = instrument.register_spec(
            name="minimal_workflow",
            version=1,
            title="Minimal",
            outputs=SimpleTestOutputs,
        )

        spec = instrument.workflow_factory[handle.workflow_id]
        assert spec.namespace == "data_reduction"  # default
        assert spec.description == ""  # default
        assert spec.source_names == []  # default
        assert spec.params is None  # default
        assert spec.aux_sources is None  # default
        assert spec.outputs is SimpleTestOutputs

    def test_register_spec_then_attach_factory(self):
        """Test two-phase registration via Instrument.register_spec()."""
        import pydantic

        class MyParams(pydantic.BaseModel):
            value: int

        instrument = Instrument(name="test_instrument")

        handle = instrument.register_spec(
            name="test_workflow",
            version=1,
            title="Test Workflow",
            params=MyParams,
            outputs=SimpleTestOutputs,
        )

        @handle.attach_factory()
        def factory(*, params: MyParams) -> Workflow:
            class MockProcessor(Workflow):
                def accumulate(self, data, *, start_time: int, end_time: int) -> None:
                    pass

                def finalize(self):
                    return {"value": params.value}

                def clear(self) -> None:
                    pass

            return MockProcessor()

        # Verify factory was attached
        from ess.livedata.config.workflow_spec import WorkflowConfig

        config = WorkflowConfig(identifier=handle.workflow_id, params={"value": 42})
        processor = instrument.workflow_factory.create(
            source_name="any-source", config=config
        )
        # Verify processor has the Workflow protocol methods
        assert hasattr(processor, 'accumulate')
        assert hasattr(processor, 'finalize')
        assert hasattr(processor, 'clear')


class TestSourceMetadata:
    """Tests for SourceMetadata model and Instrument title/description lookup."""

    def test_source_metadata_creation(self):
        """Test creating SourceMetadata with title and description."""
        metadata = SourceMetadata(
            title='Detector One', description='First detector bank'
        )

        assert metadata.title == 'Detector One'
        assert metadata.description == 'First detector bank'

    def test_source_metadata_default_description(self):
        """Test that description defaults to empty string."""
        metadata = SourceMetadata(title='Detector')

        assert metadata.title == 'Detector'
        assert metadata.description == ''

    def test_get_source_title_returns_title_when_defined(self):
        """Test get_source_title returns the title when metadata is defined."""
        instrument = Instrument(
            name='test',
            detector_names=['det1'],
            source_metadata={'det1': SourceMetadata(title='Detector One')},
        )

        assert instrument.get_source_title('det1') == 'Detector One'

    def test_get_source_title_falls_back_to_name(self):
        """Test get_source_title returns source name when no metadata defined."""
        instrument = Instrument(name='test', detector_names=['det1'])

        assert instrument.get_source_title('det1') == 'det1'

    def test_get_source_title_unknown_source_returns_name(self):
        """Test get_source_title returns the name for unknown sources."""
        instrument = Instrument(name='test')

        assert instrument.get_source_title('unknown_source') == 'unknown_source'

    def test_get_source_description_returns_description_when_defined(self):
        """Test get_source_description returns the description when defined."""
        instrument = Instrument(
            name='test',
            detector_names=['det1'],
            source_metadata={
                'det1': SourceMetadata(
                    title='Detector One', description='Main detector bank'
                )
            },
        )

        assert instrument.get_source_description('det1') == 'Main detector bank'

    def test_get_source_description_returns_empty_when_not_defined(self):
        """Test get_source_description returns empty string when no metadata."""
        instrument = Instrument(name='test', detector_names=['det1'])

        assert instrument.get_source_description('det1') == ''

    def test_get_source_description_returns_empty_for_title_only_metadata(self):
        """Test get_source_description returns empty when only title is defined."""
        instrument = Instrument(
            name='test',
            detector_names=['det1'],
            source_metadata={'det1': SourceMetadata(title='Detector One')},
        )

        assert instrument.get_source_description('det1') == ''

    def test_source_metadata_for_multiple_sources(self):
        """Test source metadata works for multiple sources of different types."""
        instrument = Instrument(
            name='test',
            detector_names=['det1', 'det2'],
            monitors=['monitor1'],
            source_metadata={
                'det1': SourceMetadata(title='First Detector', description='Desc 1'),
                'det2': SourceMetadata(title='Second Detector'),
                'monitor1': SourceMetadata(title='Beam Monitor'),
            },
        )

        assert instrument.get_source_title('det1') == 'First Detector'
        assert instrument.get_source_title('det2') == 'Second Detector'
        assert instrument.get_source_title('monitor1') == 'Beam Monitor'

        assert instrument.get_source_description('det1') == 'Desc 1'
        assert instrument.get_source_description('det2') == ''
        assert instrument.get_source_description('monitor1') == ''
