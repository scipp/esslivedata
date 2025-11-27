# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

import pytest

from ess.livedata.config.instrument import Instrument
from ess.livedata.config.workflow_spec import JobId
from ess.livedata.handlers.detector_view_specs import DetectorROIAuxSources
from ess.livedata.handlers.workflow_factory import SpecHandle


class TestRegisterDetectorViewSpecs:
    """Test the lightweight register_detector_view_specs() function."""

    def test_register_detector_view_specs_returns_handles(self):
        """Test that register_detector_view_spec() returns a handle."""
        from ess.livedata.handlers.detector_view_specs import (
            register_detector_view_spec,
        )

        instrument = Instrument(name="test_instrument")
        source_names = ["detector1", "detector2"]

        handle = register_detector_view_spec(
            instrument=instrument, projection="xy_plane", source_names=source_names
        )

        assert isinstance(handle, SpecHandle)

    def test_register_multiple_projections_via_separate_calls(self):
        """Test registering multiple projections via separate calls."""
        from ess.livedata.handlers.detector_view_specs import (
            register_detector_view_spec,
        )

        instrument = Instrument(name="test_instrument")
        source_names = ["detector1"]

        xy_handle = register_detector_view_spec(
            instrument=instrument,
            projection="xy_plane",
            source_names=source_names,
        )
        cylinder_handle = register_detector_view_spec(
            instrument=instrument,
            projection="cylinder_mantle_z",
            source_names=source_names,
        )

        assert isinstance(xy_handle, SpecHandle)
        assert isinstance(cylinder_handle, SpecHandle)

        # Verify both are registered in the factory
        xy_id = xy_handle.workflow_id
        cylinder_id = cylinder_handle.workflow_id
        assert xy_id in instrument.workflow_factory
        assert cylinder_id in instrument.workflow_factory

    def test_specs_registered_in_workflow_factory(self):
        """Test that spec is actually registered in the workflow factory."""
        from ess.livedata.handlers.detector_view_specs import (
            register_detector_view_spec,
        )

        instrument = Instrument(name="test_instrument")
        source_names = ["detector1"]

        handle = register_detector_view_spec(
            instrument=instrument, projection="xy_plane", source_names=source_names
        )

        # Verify spec is in the workflow factory
        spec_id = handle.workflow_id

        assert spec_id in instrument.workflow_factory

        # Verify spec details
        spec = instrument.workflow_factory[spec_id]
        assert spec.instrument == "test_instrument"
        assert spec.namespace == "detector_data"
        assert spec.name == "detector_xy_projection"
        assert spec.source_names == source_names

    def test_specs_have_correct_params(self):
        """Test that registered spec has the correct params and aux_sources types."""
        from ess.livedata.handlers.detector_view_specs import (
            DetectorROIAuxSources,
            DetectorViewParams,
            register_detector_view_spec,
        )

        instrument = Instrument(name="test_instrument")
        source_names = ["detector1"]

        handle = register_detector_view_spec(
            instrument=instrument, projection="xy_plane", source_names=source_names
        )

        spec_id = handle.workflow_id
        spec = instrument.workflow_factory[spec_id]

        # Check params and aux_sources are set correctly
        assert spec.params is DetectorViewParams
        assert spec.aux_sources is DetectorROIAuxSources

    def test_no_heavy_imports(self):
        """Test that importing detector_view_specs doesn't import heavy dependencies."""
        import sys

        # Ensure ess.reduce is not loaded yet
        if 'ess.reduce' in sys.modules:
            pytest.skip("ess.reduce already loaded, cannot test import isolation")

        # Import the lightweight module
        from ess.livedata.handlers import detector_view_specs  # noqa: F401

        # Verify ess.reduce was NOT imported
        assert 'ess.reduce' not in sys.modules
        assert 'ess.reduce.live' not in sys.modules
        assert 'ess.reduce.live.raw' not in sys.modules


class TestDetectorROIAuxSources:
    """Tests for DetectorROIAuxSources auxiliary source model."""

    def test_default_roi_shape_is_rectangle(self) -> None:
        """Test that the default ROI shape is rectangle."""
        aux_sources = DetectorROIAuxSources()
        assert aux_sources.roi == 'rectangle'

    def test_can_select_rectangle_roi(self) -> None:
        """Test that rectangle ROI shape can be selected."""
        aux_sources = DetectorROIAuxSources(roi='rectangle')
        assert aux_sources.roi == 'rectangle'

    def test_validator_rejects_polygon_roi(self) -> None:
        """Test that polygon ROI shape is rejected by validator."""
        with pytest.raises(ValueError, match="Currently only 'rectangle' ROI shape"):
            DetectorROIAuxSources(roi='polygon')

    def test_validator_rejects_ellipse_roi(self) -> None:
        """Test that ellipse ROI shape is rejected by validator."""
        with pytest.raises(ValueError, match="Currently only 'rectangle' ROI shape"):
            DetectorROIAuxSources(roi='ellipse')

    def test_render_prefixes_stream_name_with_job_id_and_roi(self) -> None:
        """Test that render() prefixes stream name with full job_id and roi_ prefix."""
        aux_sources = DetectorROIAuxSources(roi='rectangle')
        job_id = JobId(source_name='detector1', job_number=uuid.UUID(int=123))

        rendered = aux_sources.render(job_id)

        # Should prefix with source_name/job_number and roi_ prefix
        expected_stream = f"detector1/{job_id.job_number}/roi_rectangle"
        assert rendered == {'roi': expected_stream}

    def test_render_creates_unique_streams_for_different_jobs(self) -> None:
        """Test that different jobs get unique ROI stream names."""
        aux_sources = DetectorROIAuxSources(roi='rectangle')
        job_id_1 = JobId(source_name='detector1', job_number=uuid.UUID(int=111))
        job_id_2 = JobId(source_name='detector1', job_number=uuid.UUID(int=222))

        rendered_1 = aux_sources.render(job_id_1)
        rendered_2 = aux_sources.render(job_id_2)

        # Each job should get its own unique stream name
        assert rendered_1['roi'] != rendered_2['roi']
        assert rendered_1['roi'] == f"detector1/{job_id_1.job_number}/roi_rectangle"
        assert rendered_2['roi'] == f"detector1/{job_id_2.job_number}/roi_rectangle"

    def test_render_field_name_is_roi(self) -> None:
        """Test that the field name in rendered dict is 'roi'."""
        aux_sources = DetectorROIAuxSources()
        job_id = JobId(source_name='detector1', job_number=uuid.UUID(int=789))

        rendered = aux_sources.render(job_id)

        # Field name should be 'roi' (what the workflow expects)
        assert 'roi' in rendered
        assert len(rendered) == 1

    def test_model_dump_returns_roi_shape(self) -> None:
        """Test that model_dump returns the selected ROI shape."""
        aux_sources = DetectorROIAuxSources(roi='rectangle')
        dumped = aux_sources.model_dump(mode='json')
        assert dumped == {'roi': 'rectangle'}

    def test_render_isolates_roi_streams_per_detector_in_multi_detector_workflow(
        self,
    ) -> None:
        """
        Test that ROI streams are unique per detector in multi-detector workflows.

        When the same workflow runs on multiple detectors (same job_number),
        each detector must get its own unique ROI stream to prevent cross-talk.
        This is critical because job_number is shared across all detectors in
        the same workflow run.
        """
        aux_sources = DetectorROIAuxSources(roi='rectangle')
        shared_job_number = uuid.uuid4()

        # Same job_number, different source_names (real multi-detector scenario)
        job_id_mantle = JobId(source_name='mantle', job_number=shared_job_number)
        job_id_high_res = JobId(
            source_name='high_resolution', job_number=shared_job_number
        )

        rendered_mantle = aux_sources.render(job_id_mantle)
        rendered_high_res = aux_sources.render(job_id_high_res)

        # Each detector should get its own unique stream
        assert rendered_mantle['roi'] != rendered_high_res['roi']
        assert rendered_mantle['roi'] == f"mantle/{shared_job_number}/roi_rectangle"
        assert (
            rendered_high_res['roi']
            == f"high_resolution/{shared_job_number}/roi_rectangle"
        )


class TestRegisterLogicalDetectorViewSpec:
    """Tests for register_logical_detector_view_spec() helper function."""

    def test_returns_spec_handle(self) -> None:
        """Test that register_logical_detector_view_spec returns a SpecHandle."""
        from ess.livedata.handlers.detector_view_specs import (
            register_logical_detector_view_spec,
        )

        instrument = Instrument(name="test_instrument")

        handle = register_logical_detector_view_spec(
            instrument=instrument,
            name='test_view',
            title='Test View',
            description='A test view',
            source_names=['detector1'],
        )

        assert isinstance(handle, SpecHandle)

    def test_registers_spec_with_correct_metadata(self) -> None:
        """Test that spec is registered with correct name, title, description."""
        from ess.livedata.handlers.detector_view_specs import (
            register_logical_detector_view_spec,
        )

        instrument = Instrument(name="test_instrument")

        handle = register_logical_detector_view_spec(
            instrument=instrument,
            name='my_custom_view',
            title='My Custom View',
            description='A custom logical detector view',
            source_names=['detector1', 'detector2'],
        )

        spec = instrument.workflow_factory[handle.workflow_id]
        assert spec.name == 'my_custom_view'
        assert spec.title == 'My Custom View'
        assert spec.description == 'A custom logical detector view'
        assert spec.source_names == ['detector1', 'detector2']
        assert spec.namespace == 'detector_data'

    def test_roi_support_true_includes_aux_sources(self) -> None:
        """Test that roi_support=True includes DetectorROIAuxSources."""
        from ess.livedata.handlers.detector_view_specs import (
            register_logical_detector_view_spec,
        )

        instrument = Instrument(name="test_instrument")

        handle = register_logical_detector_view_spec(
            instrument=instrument,
            name='roi_view',
            title='ROI View',
            description='View with ROI support',
            source_names=['detector1'],
            roi_support=True,
        )

        spec = instrument.workflow_factory[handle.workflow_id]
        assert spec.aux_sources is DetectorROIAuxSources

    def test_roi_support_false_excludes_aux_sources(self) -> None:
        """Test that roi_support=False excludes aux_sources."""
        from ess.livedata.handlers.detector_view_specs import (
            register_logical_detector_view_spec,
        )

        instrument = Instrument(name="test_instrument")

        handle = register_logical_detector_view_spec(
            instrument=instrument,
            name='no_roi_view',
            title='No ROI View',
            description='View without ROI support',
            source_names=['detector1'],
            roi_support=False,
        )

        spec = instrument.workflow_factory[handle.workflow_id]
        assert spec.aux_sources is None

    def test_roi_support_defaults_to_true(self) -> None:
        """Test that roi_support defaults to True."""
        from ess.livedata.handlers.detector_view_specs import (
            register_logical_detector_view_spec,
        )

        instrument = Instrument(name="test_instrument")

        handle = register_logical_detector_view_spec(
            instrument=instrument,
            name='default_roi_view',
            title='Default ROI View',
            description='View with default ROI support',
            source_names=['detector1'],
            # roi_support not specified, should default to True
        )

        spec = instrument.workflow_factory[handle.workflow_id]
        assert spec.aux_sources is DetectorROIAuxSources

    def test_uses_detector_view_params_and_outputs(self) -> None:
        """Test that spec uses DetectorViewParams and DetectorViewOutputs."""
        from ess.livedata.handlers.detector_view_specs import (
            DetectorViewOutputs,
            DetectorViewParams,
            register_logical_detector_view_spec,
        )

        instrument = Instrument(name="test_instrument")

        handle = register_logical_detector_view_spec(
            instrument=instrument,
            name='params_test_view',
            title='Params Test',
            description='Test params and outputs',
            source_names=['detector1'],
        )

        spec = instrument.workflow_factory[handle.workflow_id]
        assert spec.params is DetectorViewParams
        assert spec.outputs is DetectorViewOutputs
