# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

import pydantic
import pytest
import scipp as sc

from ess.livedata.config.instrument import (
    DEFAULT_DIM_TITLES,
    Instrument,
    InstrumentRegistry,
    SourceMetadata,
)
from ess.livedata.config.stream import ContextBinding, Device, F144Stream
from ess.livedata.config.workflow_spec import (
    MONITORS,
    REDUCTION,
    JobId,
    WorkflowId,
    WorkflowOutputsBase,
)
from ess.livedata.workflows.workflow_factory import (
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
        assert instrument.streams == {}
        assert instrument.f144_streams == {}
        assert instrument.detector_names == []

    def test_instrument_creation_with_custom_values(self):
        """Test creating instrument with custom values."""
        custom_factory = WorkflowFactory()
        stream = F144Stream(source='src', topic='topic', units='mm')

        instrument = Instrument(
            name="custom_instrument",
            workflow_factory=custom_factory,
            streams={'attr1': stream},
        )

        assert instrument.name == "custom_instrument"
        assert instrument.workflow_factory is custom_factory
        assert instrument.streams == {'attr1': stream}
        assert instrument.f144_streams == {'attr1': stream}

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

    def test_configure_pixellated_monitor(self):
        instrument = Instrument(name="test_instrument", monitors=["mon1", "mon2"])
        det_num = sc.array(dims=['event_id'], values=[4, 5, 6], unit=None)
        instrument.configure_pixellated_monitor("mon1", detector_number=det_num)

        assert instrument.pixellated_monitor_sources == frozenset({"mon1"})
        assert sc.identical(instrument.get_detector_number("mon1"), det_num)

    def test_configure_pixellated_monitor_without_detector_number(self):
        instrument = Instrument(name="test_instrument", monitors=["mon1"])
        instrument.configure_pixellated_monitor("mon1")

        assert instrument.pixellated_monitor_sources == frozenset({"mon1"})
        with pytest.raises(KeyError):
            instrument.get_detector_number("mon1")

    def test_configure_pixellated_monitor_fails_if_not_in_monitors(self):
        instrument = Instrument(name="test_instrument", monitors=["mon1"])

        with pytest.raises(ValueError, match="not in declared monitors"):
            instrument.configure_pixellated_monitor("nonexistent")

    def test_register_spec_and_attach_factory(self):
        """Test two-phase spec registration and factory attachment."""
        instrument = Instrument(name="test_instrument")

        # Phase 1: Register spec
        handle = instrument.register_spec(
            group=MONITORS,
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
        assert spec.group is MONITORS
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
        assert spec.group is REDUCTION  # default
        assert spec.description == ""  # default
        assert spec.source_names == []  # default
        assert spec.aux_sources is None  # default
        assert spec.outputs is SimpleTestOutputs

    def test_register_spec_with_aux_sources_explicit(self):
        """Test that aux_sources can be set explicitly."""
        from ess.livedata.config.workflow_spec import AuxSources

        instrument = Instrument(name="test_instrument")

        aux_sources = AuxSources(
            {
                'monitor1': 'monitor1',
                'aux_stream': 'aux_stream',
            }
        )

        instrument.register_spec(
            name="workflow_with_aux",
            version=1,
            title="Workflow with Aux Sources",
            aux_sources=aux_sources,
            outputs=SimpleTestOutputs,
        )

        specs = instrument.workflow_factory
        assert len(specs) == 1
        spec = next(iter(specs.values()))

        assert spec.aux_sources is aux_sources
        assert 'monitor1' in spec.aux_sources.inputs
        assert 'aux_stream' in spec.aux_sources.inputs
        assert spec.aux_sources.get_defaults() == {
            'monitor1': 'monitor1',
            'aux_stream': 'aux_stream',
        }

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


class _Key:
    """Sentinel Sciline-key stand-in for binding tests."""


def _f144(name: str) -> F144Stream:
    return F144Stream(
        source=name,
        topic='topic',
        units='mm',
        nexus_path=f'/entry/instrument/{name}/value',
    )


class TestContextBindings:
    def test_add_binding_records_entry(self):
        instrument = Instrument(name='test', streams={'rot': _f144('rot')})

        instrument.add_context_binding(
            stream_name='rot',
            workflow_key=_Key,
            dependent_sources=['det1'],
        )

        binding = instrument.context_bindings[0]
        assert binding.stream_name == 'rot'
        assert binding.workflow_key is _Key
        assert binding.dependent_sources == frozenset({'det1'})

    def test_add_binding_rejects_unknown_stream(self):
        instrument = Instrument(name='test', streams={'rot': _f144('rot')})

        with pytest.raises(ValueError, match='unknown stream'):
            instrument.add_context_binding(
                stream_name='missing',
                workflow_key=_Key,
                dependent_sources=['det1'],
            )

    def test_constructor_validates_binding_stream_names(self):
        bad = ContextBinding(
            stream_name='missing',
            workflow_key=_Key,
            dependent_sources=frozenset({'det1'}),
        )
        with pytest.raises(ValueError, match='unknown stream'):
            Instrument(name='test', context_bindings=[bad])

    def test_add_binding_accepts_device_stream_target(self):
        """A :class:`Device` entry in ``streams`` is a valid binding target.

        Synthesised Device streams sit in ``streams`` alongside f144 entries
        and are eligible binding targets via the same dict-membership check.
        Bifrost relies on this to route ``InstrumentAngle``/``SampleAngle``
        from merged device streams.
        """
        instrument = Instrument(
            name='test',
            streams={
                'rot_rbv': _f144('rot_rbv'),
                'rot_target': _f144('rot_target'),
                'rot': Device(value='rot_rbv', target='rot_target', units='mm'),
            },
        )

        instrument.add_context_binding(
            stream_name='rot',
            workflow_key=_Key,
            dependent_sources=['det1'],
        )

        binding = instrument.context_bindings[0]
        assert binding.stream_name == 'rot'
        assert binding.workflow_key is _Key
        assert binding.dependent_sources == frozenset({'det1'})

    def test_validate_rejects_binding_with_unknown_dependent_source(self):
        instrument = Instrument(
            name='test', detector_names=['det1'], streams={'rot': _f144('rot')}
        )
        instrument.register_spec(
            name='w',
            version=1,
            title='W',
            source_names=['det1'],
            outputs=SimpleTestOutputs,
        )
        instrument.add_context_binding(
            stream_name='rot', workflow_key=_Key, dependent_sources=['det1', 'ghost']
        )

        with pytest.raises(ValueError, match='ghost'):
            instrument.validate()

    def test_validates_duplicate_value_log_subclass_across_bindings(self):
        """Two chain-patch entries with different streams must not share a
        :class:`ValueLog` subclass -- Sciline keys identify parameters by
        class, so a shared key would silently merge two streams into one
        Sciline node."""
        from ess.livedata.config.value_log import ValueLog

        class _SharedLog(ValueLog):
            pass

        instrument = Instrument(
            name='test',
            detector_names=['det1', 'det2'],
            streams={'a': _f144('a'), 'b': _f144('b')},
        )
        instrument.register_spec(
            name='w',
            version=1,
            title='W',
            source_names=['det1', 'det2'],
            outputs=SimpleTestOutputs,
        )
        instrument.add_context_binding(
            stream_name='a',
            dependent_sources=['det1'],
            workflow_key=_SharedLog,
        )
        instrument.add_context_binding(
            stream_name='b',
            dependent_sources=['det2'],
            workflow_key=_SharedLog,
        )

        with pytest.raises(ValueError, match=r'ValueLog subclass.*shared'):
            instrument.validate()

    def test_validates_chain_patch_stream_uniqueness(self):
        """Two chain-patch entries for one stream must declare the same
        :class:`ValueLog` subclass.

        ``wire_dynamic_transforms`` indexes bindings by ``stream_name``
        per component type; conflicting subclasses would silently collapse
        with last-write-wins semantics.
        """
        from ess.livedata.config.value_log import ValueLog

        class _LogA(ValueLog):
            pass

        class _LogB(ValueLog):
            pass

        instrument = Instrument(
            name='test',
            detector_names=['det1', 'det2'],
            streams={'shared': _f144('shared')},
        )
        instrument.register_spec(
            name='w',
            version=1,
            title='W',
            source_names=['det1', 'det2'],
            outputs=SimpleTestOutputs,
        )
        instrument.add_context_binding(
            stream_name='shared',
            dependent_sources=['det1'],
            workflow_key=_LogA,
        )
        instrument.add_context_binding(
            stream_name='shared',
            dependent_sources=['det2'],
            workflow_key=_LogB,
        )
        with pytest.raises(ValueError, match='conflicting chain-patch'):
            instrument.validate()

    def test_chain_patch_stream_allows_exact_duplicates(self):
        """Repeated identical chain-patch declarations are not a conflict.

        ``load_factories`` may be called multiple times in a long-lived
        process or across tests; redundant ``add_context_binding`` calls with
        matching ``workflow_key`` describe the same binding and must pass.
        """
        from ess.livedata.config.value_log import ValueLog

        class _Log(ValueLog):
            pass

        instrument = Instrument(
            name='test', detector_names=['det1'], streams={'s': _f144('s')}
        )
        instrument.register_spec(
            name='w',
            version=1,
            title='W',
            source_names=['det1'],
            outputs=SimpleTestOutputs,
        )
        for _ in range(2):
            instrument.add_context_binding(
                stream_name='s',
                dependent_sources=['det1'],
                workflow_key=_Log,
            )
        instrument.validate()

    def test_validates_context_vs_aux_field_collision(self):
        """A context stream_name must not match any aux_sources field name.

        At ``JobFactory.create``, context wire names and rendered aux names
        are merged into a single field→wire dict; a clashing key would
        silently overwrite the aux entry. The validator must catch the
        collision at registration.
        """
        from ess.livedata.config.workflow_spec import AuxInput, AuxSources

        instrument = Instrument(
            name='test', detector_names=['det1'], streams={'rot': _f144('rot')}
        )
        instrument.register_spec(
            name='w',
            version=1,
            title='W',
            source_names=['det1'],
            outputs=SimpleTestOutputs,
            aux_sources=AuxSources(
                {'rot': AuxInput(choices=('other',), default='other')}
            ),
        )
        instrument.add_context_binding(
            stream_name='rot', workflow_key=_Key, dependent_sources=['det1']
        )

        with pytest.raises(ValueError, match='aux_sources field'):
            instrument.validate()

    def test_skip_instrument_contexts_suppresses_context_vs_aux_collision(self):
        """``skip_instrument_contexts`` removes instrument-scope context for the
        spec, so a matching aux field name does not collide for that spec."""
        from ess.livedata.config.workflow_spec import AuxInput, AuxSources

        instrument = Instrument(
            name='test', detector_names=['det1'], streams={'rot': _f144('rot')}
        )
        handle = instrument.register_spec(
            name='w',
            version=1,
            title='W',
            source_names=['det1'],
            outputs=SimpleTestOutputs,
            aux_sources=AuxSources(
                {'rot': AuxInput(choices=('other',), default='other')}
            ),
        )
        instrument.add_context_binding(
            stream_name='rot', workflow_key=_Key, dependent_sources=['det1']
        )
        handle.skip_instrument_contexts()

        instrument.validate()

    def test_validates_wire_name_collision_between_instrument_and_spec(self):
        """Instrument- and spec-level ContextBinding entries must not name-collide.

        Per ADR 0003 § "Validation": when both scopes apply to the same
        (spec, source) pair and resolve to the same wire-stream name, the
        ambiguity is a registration error.
        """
        instrument = Instrument(
            name='test', detector_names=['det1'], streams={'rot': _f144('rot')}
        )
        handle = instrument.register_spec(
            name='w',
            version=1,
            title='W',
            source_names=['det1'],
            outputs=SimpleTestOutputs,
        )
        instrument.add_context_binding(
            stream_name='rot', workflow_key=_Key, dependent_sources=['det1']
        )
        handle.add_context_binding(stream_name='rot', workflow_key=_Key)

        with pytest.raises(ValueError, match='collision'):
            instrument.validate()

    def test_no_collision_when_dependent_sources_disjoint(self):
        """Same stream name on instrument and spec scope is fine when the sources
        do not overlap -- the (spec, source) pair never sees both bindings.
        """
        instrument = Instrument(
            name='test', detector_names=['det1', 'det2'], streams={'rot': _f144('rot')}
        )
        handle = instrument.register_spec(
            name='w',
            version=1,
            title='W',
            source_names=['det1', 'det2'],
            outputs=SimpleTestOutputs,
        )
        instrument.add_context_binding(
            stream_name='rot', workflow_key=_Key, dependent_sources=['det1']
        )
        handle.add_context_binding(
            stream_name='rot', workflow_key=_Key, dependent_sources=['det2']
        )

        # No exception.
        instrument.validate()

    def test_resolve_context_keys_matches_instrument_binding_by_source(self):
        instrument = Instrument(
            name='test', detector_names=['det1', 'det2'], streams={'rot': _f144('rot')}
        )
        handle = instrument.register_spec(
            name='w',
            version=1,
            title='W',
            source_names=['det1', 'det2'],
            outputs=SimpleTestOutputs,
        )
        instrument.add_context_binding(
            stream_name='rot', workflow_key=_Key, dependent_sources=['det1']
        )

        assert instrument.resolve_context_keys(handle.workflow_id, 'det1') == {
            'rot': _Key
        }
        assert instrument.resolve_context_keys(handle.workflow_id, 'det2') == {}

    def test_resolve_context_keys_honours_skip_instrument_contexts(self):
        instrument = Instrument(
            name='test', detector_names=['det1'], streams={'rot': _f144('rot')}
        )
        handle = instrument.register_spec(
            name='w',
            version=1,
            title='W',
            source_names=['det1'],
            outputs=SimpleTestOutputs,
        )
        instrument.add_context_binding(
            stream_name='rot', workflow_key=_Key, dependent_sources=['det1']
        )
        handle.skip_instrument_contexts()

        assert instrument.resolve_context_keys(handle.workflow_id, 'det1') == {}

    def test_resolve_context_keys_includes_spec_scope_binding(self):
        instrument = Instrument(
            name='test', detector_names=['det1'], streams={'rot': _f144('rot')}
        )
        handle = instrument.register_spec(
            name='w',
            version=1,
            title='W',
            source_names=['det1'],
            outputs=SimpleTestOutputs,
        )
        handle.skip_instrument_contexts()
        handle.add_context_binding(stream_name='rot', workflow_key=_Key)

        assert instrument.resolve_context_keys(handle.workflow_id, 'det1') == {
            'rot': _Key
        }

    def test_resolve_context_keys_raises_for_unregistered_workflow(self):
        instrument = Instrument(name='test', detector_names=['det1'])
        workflow_id = WorkflowId(
            instrument='test', namespace='spec', name='ghost', version=1
        )

        with pytest.raises(KeyError, match=r'not.*registered'):
            instrument.resolve_context_keys(workflow_id, 'det1')


class TestInstrumentRegisterSpec:
    """Test the new register_spec() convenience method for two-phase registration."""

    def test_register_spec_returns_handle(self):
        """Test that register_spec() returns a SpecHandle."""
        from ess.livedata.workflows.workflow_factory import SpecHandle

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

        from ess.livedata.config.workflow_spec import AuxSources
        from ess.livedata.workflows.workflow_factory import SpecHandle

        class MyParams(pydantic.BaseModel):
            value: int

        my_aux_sources = AuxSources({'monitor': 'monitor1'})

        class MyOutputs(pydantic.BaseModel):
            result: int

        instrument = Instrument(name="test_instrument")

        handle = instrument.register_spec(
            group=REDUCTION,
            name="test_workflow",
            version=1,
            title="Test Workflow",
            description="Test description",
            source_names=["source1", "source2"],
            params=MyParams,
            aux_sources=my_aux_sources,
            outputs=MyOutputs,
        )

        assert isinstance(handle, SpecHandle)

        # Verify spec was registered
        spec_id = handle.workflow_id
        spec = instrument.workflow_factory[spec_id]
        assert spec.instrument == "test_instrument"
        assert spec.group is REDUCTION
        assert spec.name == "test_workflow"
        assert spec.version == 1
        assert spec.title == "Test Workflow"
        assert spec.description == "Test description"
        assert spec.source_names == ["source1", "source2"]
        assert spec.params is MyParams
        assert spec.aux_sources is my_aux_sources
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
        assert spec.group is REDUCTION  # default
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

        config = WorkflowConfig(
            identifier=handle.workflow_id,
            job_id=JobId(source_name="test_source", job_number=uuid.uuid4()),
            params={"value": 42},
        )
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


class TestDimTitles:
    """Tests for coord/dim display title lookup on Instrument."""

    def test_default_dim_title_known_name(self):
        instrument = Instrument(name='test')

        assert instrument.get_dim_title('wavelength') == 'λ'
        assert instrument.get_dim_title('two_theta') == '2θ'
        assert instrument.get_dim_title('time_of_arrival') == 'Time of arrival'

    def test_default_dim_title_unknown_falls_back_to_name(self):
        instrument = Instrument(name='test')

        assert instrument.get_dim_title('detector_number') == 'detector_number'
        assert instrument.get_dim_title('foo_bar') == 'foo_bar'

    def test_per_instrument_override(self):
        instrument = Instrument(
            name='test',
            dim_titles={'detector_number': 'Pixel ID'},
        )

        assert instrument.get_dim_title('detector_number') == 'Pixel ID'

    def test_per_instrument_override_takes_precedence_over_default(self):
        # Default maps 'wavelength' -> 'Wavelength'; override wins.
        instrument = Instrument(name='test', dim_titles={'wavelength': 'λ'})

        assert instrument.get_dim_title('wavelength') == 'λ'

    def test_default_dim_titles_unaffected_by_overrides(self):
        # Sanity check: setting an override does not mutate the global default
        # (which would leak across instruments).
        before = dict(DEFAULT_DIM_TITLES)
        Instrument(name='test', dim_titles={'wavelength': 'λ'})

        assert DEFAULT_DIM_TITLES == before
