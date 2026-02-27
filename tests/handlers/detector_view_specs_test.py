# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

import pytest

from ess.livedata.config.instrument import Instrument
from ess.livedata.config.workflow_spec import JobId
from ess.livedata.handlers.detector_view_specs import (
    CoordinateModeSettings,
    DetectorROIAuxSources,
    DetectorViewParams,
)
from ess.livedata.handlers.workflow_factory import SpecHandle
from ess.livedata.parameter_models import TimeUnit, TOARange, TOFRange


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

    def test_register_mixed_projections_as_dict(self):
        """Test registering mixed projections with a dict mapping sources to types."""
        from ess.livedata.handlers.detector_view_specs import (
            register_detector_view_spec,
        )

        instrument = Instrument(name="test_instrument")
        projections = {
            "mantle_detector": "cylinder_mantle_z",
            "endcap_detector": "xy_plane",
        }

        handle = register_detector_view_spec(
            instrument=instrument,
            projection=projections,
        )

        assert isinstance(handle, SpecHandle)

        # Verify spec is registered with unified name
        spec_id = handle.workflow_id
        assert spec_id in instrument.workflow_factory

        spec = instrument.workflow_factory[spec_id]
        assert spec.name == "detector_projection"
        assert spec.title == "Detector Projection"
        # source_names should be derived from dict keys
        assert set(spec.source_names) == {"mantle_detector", "endcap_detector"}

    def test_mixed_projections_spec_includes_roi_support(self):
        """Test that mixed projection spec includes ROI aux sources."""
        from ess.livedata.handlers.detector_view_specs import (
            DetectorROIAuxSources,
            register_detector_view_spec,
        )

        instrument = Instrument(name="test_instrument")
        projections = {
            "detector1": "xy_plane",
            "detector2": "cylinder_mantle_z",
        }

        handle = register_detector_view_spec(
            instrument=instrument,
            projection=projections,
        )

        spec = instrument.workflow_factory[handle.workflow_id]
        assert spec.aux_sources is DetectorROIAuxSources

    def test_single_projection_requires_source_names(self):
        """Test that source_names is required when projection is a string."""
        from ess.livedata.handlers.detector_view_specs import (
            register_detector_view_spec,
        )

        instrument = Instrument(name="test_instrument")

        with pytest.raises(ValueError, match="source_names is required"):
            register_detector_view_spec(
                instrument=instrument,
                projection="xy_plane",
            )

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

    def test_render_returns_all_roi_geometry_streams(self) -> None:
        """Test that render() returns streams for all supported ROI geometries."""
        aux_sources = DetectorROIAuxSources()
        job_id = JobId(source_name='detector1', job_number=uuid.UUID(int=123))

        rendered = aux_sources.render(job_id)

        assert 'roi_rectangle' in rendered
        assert 'roi_polygon' in rendered
        assert len(rendered) == 2

    def test_render_prefixes_stream_names_with_job_id(self) -> None:
        """Test that render() prefixes stream names with full job_id."""
        aux_sources = DetectorROIAuxSources()
        job_id = JobId(source_name='detector1', job_number=uuid.UUID(int=123))

        rendered = aux_sources.render(job_id)

        expected_rectangle = f"detector1/{job_id.job_number}/roi_rectangle"
        expected_polygon = f"detector1/{job_id.job_number}/roi_polygon"
        assert rendered['roi_rectangle'] == expected_rectangle
        assert rendered['roi_polygon'] == expected_polygon

    def test_render_creates_unique_streams_for_different_jobs(self) -> None:
        """Test that different jobs get unique ROI stream names."""
        aux_sources = DetectorROIAuxSources()
        job_id_1 = JobId(source_name='detector1', job_number=uuid.UUID(int=111))
        job_id_2 = JobId(source_name='detector1', job_number=uuid.UUID(int=222))

        rendered_1 = aux_sources.render(job_id_1)
        rendered_2 = aux_sources.render(job_id_2)

        # Each job should get unique stream names
        assert rendered_1['roi_rectangle'] != rendered_2['roi_rectangle']
        assert rendered_1['roi_polygon'] != rendered_2['roi_polygon']

    def test_render_isolates_roi_streams_per_detector_in_multi_detector_workflow(
        self,
    ) -> None:
        """
        Test that ROI streams are unique per detector in multi-detector workflows.

        When the same workflow runs on multiple detectors (same job_number),
        each detector must get its own unique ROI stream to prevent cross-talk.
        """
        aux_sources = DetectorROIAuxSources()
        shared_job_number = uuid.uuid4()

        job_id_mantle = JobId(source_name='mantle', job_number=shared_job_number)
        job_id_high_res = JobId(
            source_name='high_resolution', job_number=shared_job_number
        )

        rendered_mantle = aux_sources.render(job_id_mantle)
        rendered_high_res = aux_sources.render(job_id_high_res)

        # Each detector should get unique streams
        assert rendered_mantle['roi_rectangle'] != rendered_high_res['roi_rectangle']
        assert rendered_mantle['roi_polygon'] != rendered_high_res['roi_polygon']
        assert (
            rendered_mantle['roi_rectangle']
            == f"mantle/{shared_job_number}/roi_rectangle"
        )
        assert (
            rendered_high_res['roi_rectangle']
            == f"high_resolution/{shared_job_number}/roi_rectangle"
        )


class TestDetectorViewParamsGetActiveRange:
    """Tests for DetectorViewParams.get_active_range() unit handling."""

    @pytest.mark.parametrize(
        'unit', [TimeUnit.NS, TimeUnit.US, TimeUnit.MS, TimeUnit.S]
    )
    def test_toa_range_preserves_user_unit(self, unit: TimeUnit):
        params = DetectorViewParams(
            coordinate_mode=CoordinateModeSettings(mode='toa'),
            toa_range=TOARange(enabled=True, start=0.0, stop=71.4, unit=unit),
        )
        range_filter = params.get_active_range()
        assert range_filter is not None
        low, high = range_filter
        assert low.unit == unit.value
        assert high.unit == unit.value

    @pytest.mark.parametrize(
        'unit', [TimeUnit.NS, TimeUnit.US, TimeUnit.MS, TimeUnit.S]
    )
    def test_tof_range_preserves_user_unit(self, unit: TimeUnit):
        params = DetectorViewParams(
            coordinate_mode=CoordinateModeSettings(mode='tof'),
            tof_range=TOFRange(enabled=True, start=10.0, stop=50.0, unit=unit),
        )
        range_filter = params.get_active_range()
        assert range_filter is not None
        low, high = range_filter
        assert low.unit == unit.value
        assert high.unit == unit.value

    def test_disabled_range_returns_none(self):
        params = DetectorViewParams(
            coordinate_mode=CoordinateModeSettings(mode='toa'),
            toa_range=TOARange(enabled=False),
        )
        assert params.get_active_range() is None
