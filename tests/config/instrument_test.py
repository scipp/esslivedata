# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc

from ess.livedata.config.instrument import Instrument, InstrumentRegistry
from ess.livedata.handlers.workflow_factory import (
    Workflow,
    WorkflowFactory,
)


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

    def test_add_detector_with_explicit_number(self):
        """Test adding detector with explicit detector number."""
        instrument = Instrument(name="test_instrument")
        detector_number = sc.array(dims=['detector'], values=[1, 2, 3])

        instrument.add_detector("detector1", detector_number)

        assert "detector1" in instrument.detector_names
        assert sc.identical(
            instrument.get_detector_number("detector1"), detector_number
        )

    def test_add_detector_without_number_fails_without_nexus(self):
        """Test adding detector without number fails when no nexus file available."""
        instrument = Instrument(name="nonexistent_instrument")

        with pytest.raises(ValueError, match="Nexus file not set or found"):
            instrument.add_detector("detector1")

    def test_get_detector_number_for_nonexistent_detector(self):
        """Test getting detector number for non-existent detector raises KeyError."""
        instrument = Instrument(name="test_instrument")

        with pytest.raises(KeyError):
            instrument.get_detector_number("nonexistent_detector")

    def test_add_detector_with_detector_numbers_from_nexus_file(self):
        instrument = Instrument(name="dream")
        instrument.add_detector("mantle_detector")
        detector_number = instrument.get_detector_number("mantle_detector")
        assert isinstance(detector_number, sc.Variable)

    def test_multiple_detectors(self):
        """Test managing multiple detectors."""
        instrument = Instrument(name="test_instrument")
        detector1_number = sc.array(dims=['detector'], values=[1, 2])
        detector2_number = sc.array(dims=['detector'], values=[3, 4, 5])

        instrument.add_detector("detector1", detector1_number)
        instrument.add_detector("detector2", detector2_number)

        assert len(instrument.detector_names) == 2
        assert "detector1" in instrument.detector_names
        assert "detector2" in instrument.detector_names
        assert sc.identical(
            instrument.get_detector_number("detector1"), detector1_number
        )
        assert sc.identical(
            instrument.get_detector_number("detector2"), detector2_number
        )

    def test_register_workflow_decorator(self):
        """Test workflow registration decorator functionality."""
        instrument = Instrument(name="test_instrument")

        # Create a simple factory function
        def simple_processor_factory(source_name: str) -> Workflow:
            # Return a mock processor for testing
            class MockProcessor(Workflow):
                def __call__(self, *args, **kwargs):
                    return {"source": source_name}

            return MockProcessor()

        # Register the workflow
        decorator = instrument.register_workflow(
            namespace="test_namespace",
            name="test_workflow",
            version=1,
            title="Test Workflow",
            description="A test workflow",
            source_names=["source1", "source2"],
        )

        # Apply decorator
        registered_factory = decorator(simple_processor_factory)

        # Verify the factory is returned unchanged
        assert registered_factory is simple_processor_factory

        # Verify it was registered in the processor factory
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
        assert spec.aux_sources is None  # Will be filled from type hint in Phase 2

    def test_register_workflow_with_defaults(self):
        """Test workflow registration with default values."""
        instrument = Instrument(name="test_instrument")

        def simple_factory() -> Workflow:
            class MockProcessor(Workflow):
                def __call__(self, *args, **kwargs):
                    return {}

            return MockProcessor()

        decorator = instrument.register_workflow(
            name="minimal_workflow", version=1, title="Minimal Workflow"
        )

        registered_factory = decorator(simple_factory)
        assert registered_factory is simple_factory

        specs = instrument.workflow_factory
        assert len(specs) == 1
        spec = next(iter(specs.values()))
        assert spec.namespace == "data_reduction"  # default
        assert spec.description == ""  # default
        assert spec.source_names == []  # default
        assert spec.aux_sources is None  # default

    def test_register_workflow_with_aux_source_names_backward_compat(self):
        """Test backward compatibility for aux_source_names parameter."""
        instrument = Instrument(name="test_instrument")

        def simple_factory() -> Workflow:
            class MockProcessor(Workflow):
                def __call__(self, *args, **kwargs):
                    return {}

            return MockProcessor()

        # Use the deprecated aux_source_names parameter
        decorator = instrument.register_workflow(
            name="compat_workflow",
            version=1,
            title="Compat Workflow",
            aux_source_names=["monitor1", "aux_stream"],
        )

        registered_factory = decorator(simple_factory)
        assert registered_factory is simple_factory

        specs = instrument.workflow_factory
        assert len(specs) == 1
        spec = next(iter(specs.values()))

        # aux_sources should be auto-created from aux_source_names
        assert spec.aux_sources is not None

        # Verify it's a Pydantic model with the expected fields
        model_instance = spec.aux_sources()
        assert hasattr(model_instance, 'monitor1')
        assert hasattr(model_instance, 'aux_stream')

        # The default values should match the source names
        assert model_instance.monitor1.value == 'monitor1'
        assert model_instance.aux_stream.value == 'aux_stream'

    def test_multiple_workflow_registrations(self):
        """Test registering multiple workflows."""
        instrument = Instrument(name="test_instrument")

        def factory1() -> Workflow:
            class MockProcessor(Workflow):
                def __call__(self, *args, **kwargs):
                    return {"workflow": 1}

            return MockProcessor()

        def factory2() -> Workflow:
            class MockProcessor(Workflow):
                def __call__(self, *args, **kwargs):
                    return {"workflow": 2}

            return MockProcessor()

        # Register two workflows
        instrument.register_workflow(name="workflow1", version=1, title="Workflow 1")(
            factory1
        )
        instrument.register_workflow(name="workflow2", version=1, title="Workflow 2")(
            factory2
        )

        specs = instrument.workflow_factory
        assert len(specs) == 2

        workflow_names = {spec.name for spec in specs.values()}
        assert workflow_names == {"workflow1", "workflow2"}
