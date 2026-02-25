# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest
import sciline
from ess.reduce.streaming import StreamProcessor
from pydantic import BaseModel, ValidationError

from ess.livedata.config.workflow_spec import WorkflowConfig, WorkflowId, WorkflowSpec
from ess.livedata.handlers.workflow_factory import WorkflowFactory


class MyParams(BaseModel):
    value: int = 42
    name: str = "test"


class MyAuxSources(BaseModel):
    monitor: str
    rotation: str


@pytest.fixture
def workflow_id():
    """Fixture to create a WorkflowId for testing."""
    return WorkflowId(
        instrument="test-instrument",
        namespace="test-namespace",
        name="test-workflow",
        version=1,
    )


@pytest.fixture
def workflow_spec(workflow_id):
    """Fixture to create a basic WorkflowSpec for testing."""
    return WorkflowSpec(
        instrument=workflow_id.instrument,
        namespace=workflow_id.namespace,
        name=workflow_id.name,
        version=workflow_id.version,
        title="Pretty name",
        description="Test description",
        params=None,
    )


@pytest.fixture
def workflow_spec_with_sources(workflow_id):
    """Fixture to create a WorkflowSpec with source names for testing."""
    return WorkflowSpec(
        instrument=workflow_id.instrument,
        namespace=workflow_id.namespace,
        name=workflow_id.name,
        version=workflow_id.version,
        title="test-workflow",
        description="Test",
        source_names=["source1", "source2"],
        params=None,
    )


def make_dummy_workflow() -> StreamProcessor:
    """Fixture to create a mock StreamProcessor."""
    workflow = sciline.Pipeline()
    return StreamProcessor(
        base_workflow=workflow, dynamic_keys=(int,), target_keys=(), accumulators=()
    )


def make_dummy_workflow_with_source(*, source_name: str) -> StreamProcessor:
    """Fixture to create a mock StreamProcessor that uses source_name."""
    workflow = sciline.Pipeline()
    # In a real implementation, the source_name would be used to customize the processor
    return StreamProcessor(
        base_workflow=workflow, dynamic_keys=(int,), target_keys=(), accumulators=()
    )


def make_dummy_workflow_with_params(*, params: MyParams) -> StreamProcessor:
    """Fixture to create a mock StreamProcessor that uses params."""
    workflow = sciline.Pipeline()
    return StreamProcessor(
        base_workflow=workflow, dynamic_keys=(int,), target_keys=(), accumulators=()
    )


def make_dummy_workflow_with_aux_sources() -> StreamProcessor:
    """Fixture to create a mock StreamProcessor that uses aux_sources."""
    workflow = sciline.Pipeline()
    return StreamProcessor(
        base_workflow=workflow, dynamic_keys=(int,), target_keys=(), accumulators=()
    )


class TestWorkflowFactory:
    def test_init_factory_is_empty(self):
        factory = WorkflowFactory()
        assert len(factory) == 0
        assert list(factory) == []

    def test_register_adds_workflow_spec(self, workflow_id, workflow_spec):
        factory = WorkflowFactory()

        handle = factory.register_spec(workflow_spec)

        @handle.attach_factory()
        def factory_func():
            return make_dummy_workflow()

        assert len(factory) == 1
        assert workflow_id in factory
        stored_spec = factory[workflow_id]
        assert stored_spec.name == "test-workflow"
        assert stored_spec.description == "Test description"
        assert stored_spec.source_names == []

    def test_register_with_source_names(self, workflow_id, workflow_spec_with_sources):
        factory = WorkflowFactory()

        handle = factory.register_spec(workflow_spec_with_sources)

        @handle.attach_factory()
        def factory_func():
            return make_dummy_workflow()

        stored_spec = factory[workflow_id]
        assert stored_spec.source_names == ["source1", "source2"]

    def test_register_duplicate_id_raises_error(self, workflow_id, workflow_spec):
        factory = WorkflowFactory()

        handle = factory.register_spec(workflow_spec)

        @handle.attach_factory()
        def factory_func():
            return make_dummy_workflow()

        with pytest.raises(
            ValueError,
            match="already registered",
        ):
            factory.register_spec(workflow_spec)

    def test_create_returns_stream_processor(self, workflow_id, workflow_spec):
        factory = WorkflowFactory()

        handle = factory.register_spec(workflow_spec)

        @handle.attach_factory()
        def factory_func():
            return make_dummy_workflow()

        config = WorkflowConfig(identifier=workflow_id)
        processor = factory.create(source_name="any-source", config=config)
        assert isinstance(processor, StreamProcessor)

    def test_create_with_source_name_parameter(self):
        factory = WorkflowFactory()
        workflow_id = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="test-workflow",
            version=1,
        )
        spec = WorkflowSpec(
            instrument=workflow_id.instrument,
            namespace=workflow_id.namespace,
            name=workflow_id.name,
            version=workflow_id.version,
            title="test-workflow",
            description="Test",
            source_names=["source1"],
            params=None,
        )

        handle = factory.register_spec(spec)

        @handle.attach_factory()
        def factory_func(*, source_name):
            return make_dummy_workflow_with_source(source_name=source_name)

        config = WorkflowConfig(identifier=workflow_id)
        processor = factory.create(source_name="source1", config=config)
        assert isinstance(processor, StreamProcessor)

    def test_create_with_params(self):
        factory = WorkflowFactory()
        workflow_id = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="test-workflow",
            version=1,
        )
        spec = WorkflowSpec(
            instrument=workflow_id.instrument,
            namespace=workflow_id.namespace,
            name=workflow_id.name,
            version=workflow_id.version,
            title="test-workflow",
            description="Test",
            params=MyParams,
        )

        handle = factory.register_spec(spec)

        @handle.attach_factory()
        def factory_func(*, params: MyParams):
            return make_dummy_workflow_with_params(params=params)

        config = WorkflowConfig(
            identifier=workflow_id, params={"value": 100, "name": "custom"}
        )
        processor = factory.create(source_name="any-source", config=config)
        assert isinstance(processor, StreamProcessor)

    def test_create_invalid_params_raises_pydantic_error(self):
        factory = WorkflowFactory()
        workflow_id = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="test-workflow",
            version=1,
        )
        spec = WorkflowSpec(
            instrument=workflow_id.instrument,
            namespace=workflow_id.namespace,
            name=workflow_id.name,
            version=workflow_id.version,
            title="test-workflow",
            description="Test",
            params=MyParams,
        )

        handle = factory.register_spec(spec)

        @handle.attach_factory()
        def factory_func(*, params: MyParams):
            return make_dummy_workflow_with_params(params=params)

        config = WorkflowConfig(
            identifier=workflow_id,
            params={"value": "not-an-int", "name": "test"},  # Invalid type for 'value'
        )

        # This should raise a pydantic validation error
        with pytest.raises(ValidationError):
            factory.create(source_name="any-source", config=config)

    def test_unknown_workflow_id_raises_key_error(self):
        factory = WorkflowFactory()
        non_existent_id = WorkflowId(
            instrument="non-existent",
            namespace="non-existent",
            name="non-existent",
            version=1,
        )
        config = WorkflowConfig(identifier=non_existent_id)

        with pytest.raises(KeyError, match="Unknown workflow ID"):
            factory.create(source_name="any-source", config=config)

    def test_invalid_source_name_raises_value_error(self, workflow_spec_with_sources):
        factory = WorkflowFactory()
        workflow_id = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="test-workflow",
            version=1,
        )
        # Override source_names for this specific test
        spec = WorkflowSpec(
            instrument=workflow_spec_with_sources.instrument,
            namespace=workflow_spec_with_sources.namespace,
            name=workflow_spec_with_sources.name,
            version=workflow_spec_with_sources.version,
            title=workflow_spec_with_sources.title,
            description=workflow_spec_with_sources.description,
            source_names=["allowed-source"],
            params=None,
        )

        handle = factory.register_spec(spec)

        @handle.attach_factory()
        def factory_func():
            return make_dummy_workflow()

        config = WorkflowConfig(identifier=workflow_id)

        with pytest.raises(ValueError, match="Source 'invalid-source' is not allowed"):
            factory.create(source_name="invalid-source", config=config)

    def test_multiple_registrations_create_distinct_entries(self):
        factory = WorkflowFactory()
        workflow_id1 = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="workflow1",
            version=1,
        )
        workflow_id2 = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="workflow2",
            version=1,
        )
        spec1 = WorkflowSpec(
            instrument=workflow_id1.instrument,
            namespace=workflow_id1.namespace,
            name=workflow_id1.name,
            version=workflow_id1.version,
            title="workflow1",
            description="Test 1",
            params=None,
        )
        spec2 = WorkflowSpec(
            instrument=workflow_id2.instrument,
            namespace=workflow_id2.namespace,
            name=workflow_id2.name,
            version=workflow_id2.version,
            title="workflow2",
            description="Test 2",
            params=None,
        )

        handle1 = factory.register_spec(spec1)
        handle2 = factory.register_spec(spec2)

        @handle1.attach_factory()
        def factory_func1():
            return make_dummy_workflow()

        @handle2.attach_factory()
        def factory_func2():
            return make_dummy_workflow()

        assert len(factory) == 2
        specs = list(factory.values())
        names = [spec.name for spec in specs]
        assert sorted(names) == ["workflow1", "workflow2"]

    def test_mapping_interface(self, workflow_id, workflow_spec):
        factory = WorkflowFactory()

        handle = factory.register_spec(workflow_spec)

        @handle.attach_factory()
        def factory_func():
            return make_dummy_workflow()

        assert factory[workflow_id].name == "test-workflow"
        assert list(iter(factory)) == [workflow_id]
        assert len(factory) == 1
        assert list(factory.keys()) == [workflow_id]
        assert len(list(factory.values())) == 1
        assert next(iter(factory.items()))[0] == workflow_id

    def test_duplicate_workflow_names_different_versions(self):
        factory = WorkflowFactory()
        workflow_id1 = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="same-name",
            version=1,
        )
        workflow_id2 = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="same-name",
            version=2,
        )
        spec1 = WorkflowSpec(
            instrument=workflow_id1.instrument,
            namespace=workflow_id1.namespace,
            name=workflow_id1.name,
            version=workflow_id1.version,
            title="V1",
            description="Test 1",
            params=None,
        )
        spec2 = WorkflowSpec(
            instrument=workflow_id2.instrument,
            namespace=workflow_id2.namespace,
            name=workflow_id2.name,
            version=workflow_id2.version,
            title="V2",
            description="Test 2",
            params=None,
        )

        handle1 = factory.register_spec(spec1)
        handle2 = factory.register_spec(spec2)

        @handle1.attach_factory()
        def factory_func1():
            return make_dummy_workflow()

        @handle2.attach_factory()
        def factory_func2():
            return make_dummy_workflow()

        # Both functions should be registered with different IDs but same name
        assert len(factory) == 2
        specs = list(factory.values())
        names = [spec.name for spec in specs]
        assert names.count("same-name") == 2

        # IDs should be different
        assert workflow_id1 in factory
        assert workflow_id2 in factory

        # Both workflows should be callable
        config1 = WorkflowConfig(identifier=workflow_id1)
        config2 = WorkflowConfig(identifier=workflow_id2)
        processor1 = factory.create(source_name="any", config=config1)
        processor2 = factory.create(source_name="any", config=config2)
        assert isinstance(processor1, StreamProcessor)
        assert isinstance(processor2, StreamProcessor)

    def test_empty_name(self):
        factory = WorkflowFactory()
        workflow_id = WorkflowId(
            instrument="test-instrument", namespace="test-namespace", name="", version=1
        )
        spec = WorkflowSpec(
            instrument=workflow_id.instrument,
            namespace=workflow_id.namespace,
            name=workflow_id.name,
            version=workflow_id.version,
            title="",
            description="Test",
            params=None,
        )

        handle = factory.register_spec(spec)

        @handle.attach_factory()
        def factory_func():
            return make_dummy_workflow()

        stored_spec = factory[workflow_id]
        assert stored_spec.name == ""

        # Should still create a processor
        config = WorkflowConfig(identifier=workflow_id)
        processor = factory.create(source_name="any", config=config)
        assert isinstance(processor, StreamProcessor)

    def test_case_sensitivity_in_source_names(self):
        factory = WorkflowFactory()
        sources = ["Source1", "SOURCE2"]
        workflow_id = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="test-workflow",
            version=1,
        )
        spec = WorkflowSpec(
            instrument=workflow_id.instrument,
            namespace=workflow_id.namespace,
            name=workflow_id.name,
            version=workflow_id.version,
            title="test-workflow",
            description="Test",
            source_names=sources,
            params=None,
        )

        handle = factory.register_spec(spec)

        @handle.attach_factory()
        def factory_func():
            return make_dummy_workflow()

        config = WorkflowConfig(identifier=workflow_id)

        # These should work
        processor1 = factory.create(source_name="Source1", config=config)
        processor2 = factory.create(source_name="SOURCE2", config=config)
        assert isinstance(processor1, StreamProcessor)
        assert isinstance(processor2, StreamProcessor)

        # These should fail due to case sensitivity
        with pytest.raises(ValueError, match="is not allowed"):
            factory.create(source_name="source1", config=config)

        with pytest.raises(ValueError, match="is not allowed"):
            factory.create(source_name="source2", config=config)

    def test_create_with_aux_sources(self):
        """Test that workflows can be created with aux_sources parameter."""
        factory = WorkflowFactory()
        workflow_id = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="test-workflow",
            version=1,
        )
        spec = WorkflowSpec(
            instrument=workflow_id.instrument,
            namespace=workflow_id.namespace,
            name=workflow_id.name,
            version=workflow_id.version,
            title="test-workflow",
            description="Test",
            params=None,
            aux_sources=MyAuxSources,
        )

        handle = factory.register_spec(spec)

        @handle.attach_factory()
        def factory_func():
            return make_dummy_workflow_with_aux_sources()

        config = WorkflowConfig(
            identifier=workflow_id,
            aux_source_names={"monitor": "monitor2", "rotation": "rotation2"},
        )
        processor = factory.create(source_name="any-source", config=config)
        assert isinstance(processor, StreamProcessor)

    def test_create_without_aux_sources_when_not_required(self):
        """Test that workflows without aux_sources work normally."""
        factory = WorkflowFactory()
        workflow_id = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="test-workflow",
            version=1,
        )
        spec = WorkflowSpec(
            instrument=workflow_id.instrument,
            namespace=workflow_id.namespace,
            name=workflow_id.name,
            version=workflow_id.version,
            title="test-workflow",
            description="Test",
            params=None,
        )

        handle = factory.register_spec(spec)

        @handle.attach_factory()
        def factory_func():
            return make_dummy_workflow()

        config = WorkflowConfig(identifier=workflow_id)
        processor = factory.create(source_name="any-source", config=config)
        assert isinstance(processor, StreamProcessor)

    def test_create_with_aux_sources_rejects_when_not_expected(self):
        """Test that providing aux_sources when not required raises an error."""
        factory = WorkflowFactory()
        workflow_id = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="test-workflow",
            version=1,
        )
        spec = WorkflowSpec(
            instrument=workflow_id.instrument,
            namespace=workflow_id.namespace,
            name=workflow_id.name,
            version=workflow_id.version,
            title="test-workflow",
            description="Test",
            params=None,
        )

        handle = factory.register_spec(spec)

        @handle.attach_factory()
        def factory_func():
            return make_dummy_workflow()

        config = WorkflowConfig(
            identifier=workflow_id,
            aux_source_names={"monitor": "monitor1"},
        )

        with pytest.raises(
            ValueError,
            match="does not require auxiliary sources",
        ):
            factory.create(source_name="any-source", config=config)

    def test_create_with_invalid_aux_sources_raises_pydantic_error(self):
        """Test that invalid aux_sources data raises a Pydantic validation error."""
        factory = WorkflowFactory()
        workflow_id = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="test-workflow",
            version=1,
        )
        spec = WorkflowSpec(
            instrument=workflow_id.instrument,
            namespace=workflow_id.namespace,
            name=workflow_id.name,
            version=workflow_id.version,
            title="test-workflow",
            description="Test",
            params=None,
            aux_sources=MyAuxSources,
        )

        handle = factory.register_spec(spec)

        @handle.attach_factory()
        def factory_func():
            return make_dummy_workflow_with_aux_sources()

        # Missing required field 'rotation'
        config = WorkflowConfig(
            identifier=workflow_id,
            aux_source_names={"monitor": "monitor1"},
        )

        with pytest.raises(ValidationError):
            factory.create(source_name="any-source", config=config)

    def test_register_sets_aux_sources_type_explicitly(self):
        """Test that the register decorator sets aux_sources type explicitly in spec."""
        factory = WorkflowFactory()
        workflow_id = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="test-workflow",
            version=1,
        )
        spec = WorkflowSpec(
            instrument=workflow_id.instrument,
            namespace=workflow_id.namespace,
            name=workflow_id.name,
            version=workflow_id.version,
            title="test-workflow",
            description="Test",
            params=None,
            aux_sources=MyAuxSources,
        )

        handle = factory.register_spec(spec)

        @handle.attach_factory()
        def factory_func():
            return make_dummy_workflow_with_aux_sources()

        # Check that aux_sources type was set in spec
        stored_spec = factory[workflow_id]
        assert stored_spec.aux_sources is MyAuxSources


class TestTwoPhaseRegistration:
    """Test the new two-phase registration pattern with SpecHandle."""

    def test_register_spec_returns_handle(self):
        """Test that register_spec() returns a SpecHandle."""
        from ess.livedata.handlers.workflow_factory import SpecHandle

        factory = WorkflowFactory()
        workflow_id = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="test-workflow",
            version=1,
        )
        spec = WorkflowSpec(
            instrument=workflow_id.instrument,
            namespace=workflow_id.namespace,
            name=workflow_id.name,
            version=workflow_id.version,
            title="test-workflow",
            description="Test",
            params=None,
        )

        handle = factory.register_spec(spec)

        assert isinstance(handle, SpecHandle)
        assert handle.workflow_id == workflow_id

    def test_register_spec_stores_spec(self):
        """Test that register_spec() stores the spec in the factory."""
        factory = WorkflowFactory()
        workflow_id = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="test-workflow",
            version=1,
        )
        spec = WorkflowSpec(
            instrument=workflow_id.instrument,
            namespace=workflow_id.namespace,
            name=workflow_id.name,
            version=workflow_id.version,
            title="test-workflow",
            description="Test",
            params=None,
        )

        factory.register_spec(spec)

        assert workflow_id in factory
        assert factory[workflow_id] is spec

    def test_register_spec_duplicate_raises_error(self):
        """Test that registering the same spec twice raises an error."""
        factory = WorkflowFactory()
        workflow_id = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="test-workflow",
            version=1,
        )
        spec = WorkflowSpec(
            instrument=workflow_id.instrument,
            namespace=workflow_id.namespace,
            name=workflow_id.name,
            version=workflow_id.version,
            title="test-workflow",
            description="Test",
            params=None,
        )

        factory.register_spec(spec)

        with pytest.raises(ValueError, match="already registered"):
            factory.register_spec(spec)

    def test_attach_factory_via_handle(self):
        """Test attaching factory via SpecHandle.attach_factory()."""
        factory = WorkflowFactory()
        workflow_id = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="test-workflow",
            version=1,
        )
        spec = WorkflowSpec(
            instrument=workflow_id.instrument,
            namespace=workflow_id.namespace,
            name=workflow_id.name,
            version=workflow_id.version,
            title="test-workflow",
            description="Test",
            params=None,
        )

        handle = factory.register_spec(spec)

        @handle.attach_factory()
        def factory_func():
            return make_dummy_workflow()

        # Spec should be accessible
        assert workflow_id in factory

        # Should be able to create workflow
        config = WorkflowConfig(identifier=workflow_id)
        processor = factory.create(source_name="any-source", config=config)
        assert isinstance(processor, StreamProcessor)

    def test_attach_factory_with_params(self):
        """Test attaching factory with params type hint."""
        factory = WorkflowFactory()
        workflow_id = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="test-workflow",
            version=1,
        )
        spec = WorkflowSpec(
            instrument=workflow_id.instrument,
            namespace=workflow_id.namespace,
            name=workflow_id.name,
            version=workflow_id.version,
            title="test-workflow",
            description="Test",
            params=MyParams,  # Explicit params!
        )

        handle = factory.register_spec(spec)

        @handle.attach_factory()
        def factory_func(*, params: MyParams):
            return make_dummy_workflow_with_params(params=params)

        # Should be able to create workflow with params
        config = WorkflowConfig(
            identifier=workflow_id, params={"value": 100, "name": "custom"}
        )
        processor = factory.create(source_name="any-source", config=config)
        assert isinstance(processor, StreamProcessor)

    def test_attach_factory_params_type_mismatch_raises_error(self):
        """Test that type mismatch between spec and factory params raises an error."""
        factory = WorkflowFactory()
        workflow_id = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="test-workflow",
            version=1,
        )

        class OtherParams(BaseModel):
            other: str

        spec = WorkflowSpec(
            instrument=workflow_id.instrument,
            namespace=workflow_id.namespace,
            name=workflow_id.name,
            version=workflow_id.version,
            title="test-workflow",
            description="Test",
            params=MyParams,  # Spec expects MyParams
        )

        handle = factory.register_spec(spec)

        with pytest.raises(TypeError, match="Params type mismatch"):

            @handle.attach_factory()
            def factory_func(*, params: OtherParams):  # Factory has OtherParams
                return make_dummy_workflow_with_params(params=params)

    def test_attach_factory_spec_has_params_factory_has_none_raises_error(self):
        """Test error when spec has params but factory doesn't."""
        factory = WorkflowFactory()
        workflow_id = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="test-workflow",
            version=1,
        )
        spec = WorkflowSpec(
            instrument=workflow_id.instrument,
            namespace=workflow_id.namespace,
            name=workflow_id.name,
            version=workflow_id.version,
            title="test-workflow",
            description="Test",
            params=MyParams,  # Spec has params
        )

        handle = factory.register_spec(spec)

        with pytest.raises(TypeError, match="Spec has params but factory has none"):

            @handle.attach_factory()
            def factory_func():  # Factory has no params
                return make_dummy_workflow()

    def test_attach_factory_spec_has_none_factory_has_params_raises_error(self):
        """Test error when spec has no params but factory does."""
        factory = WorkflowFactory()
        workflow_id = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="test-workflow",
            version=1,
        )
        spec = WorkflowSpec(
            instrument=workflow_id.instrument,
            namespace=workflow_id.namespace,
            name=workflow_id.name,
            version=workflow_id.version,
            title="test-workflow",
            description="Test",
            params=None,  # Spec has no params
        )

        handle = factory.register_spec(spec)

        with pytest.raises(TypeError, match="Factory has params but spec has none"):

            @handle.attach_factory()
            def factory_func(*, params: MyParams):  # Factory has params
                return make_dummy_workflow_with_params(params=params)

    def test_attach_factory_without_registered_spec_raises_error(self):
        """Test that attach_factory() raises error if spec not registered."""
        factory = WorkflowFactory()
        workflow_id = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="test-workflow",
            version=1,
        )

        with pytest.raises(ValueError, match="not registered"):
            factory.attach_factory(workflow_id)

    def test_spec_params_immutable_after_registration(self):
        """Test that spec.params is not mutated after register_spec()."""
        factory = WorkflowFactory()
        workflow_id = WorkflowId(
            instrument="test-instrument",
            namespace="test-namespace",
            name="test-workflow",
            version=1,
        )
        spec = WorkflowSpec(
            instrument=workflow_id.instrument,
            namespace=workflow_id.namespace,
            name=workflow_id.name,
            version=workflow_id.version,
            title="test-workflow",
            description="Test",
            params=MyParams,  # Explicit params
        )

        handle = factory.register_spec(spec)

        # Verify params is still MyParams
        assert factory[workflow_id].params is MyParams

        @handle.attach_factory()
        def factory_func(*, params: MyParams):
            return make_dummy_workflow_with_params(params=params)

        # Verify params is still MyParams (not mutated)
        assert factory[workflow_id].params is MyParams
