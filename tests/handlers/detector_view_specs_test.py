# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

import pytest

from ess.livedata.config.instrument import Instrument
from ess.livedata.config.workflow_spec import JobId
from ess.livedata.handlers.detector_view_specs import (
    CoordinateModeSettings,
    DetectorViewParams,
)
from ess.livedata.handlers.workflow_factory import SpecHandle
from ess.livedata.parameter_models import (
    TimeUnit,
    TOARange,
    WavelengthRangeFilter,
    WavelengthUnit,
)


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

    def test_mixed_projections_spec_includes_roi_context_bindings(self):
        """Test that mixed projection spec declares ROI ContextBinding records."""
        from ess.livedata.handlers.detector_view_specs import (
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

        reg = instrument.workflow_factory.registration(handle.workflow_id)
        assert {ci.stream_name for ci in reg.context_bindings} == {
            'roi_rectangle',
            'roi_polygon',
        }

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
        assert spec.group.name == "detector_data"
        assert spec.name == "detector_xy_projection"
        assert spec.source_names == source_names

    def test_specs_have_correct_params(self):
        """Test that registered spec has the correct params."""
        from ess.livedata.handlers.detector_view_specs import (
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

        assert spec.params is DetectorViewParams

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


class TestRegisterDetectorViewSpecROIContextBindings:
    """ROI ContextBinding records registered by register_detector_view_spec()."""

    def test_register_adds_roi_rectangle_and_polygon_context_bindings(self) -> None:
        from ess.livedata.handlers.detector_view.types import (
            ROIPolygonRequest,
            ROIRectangleRequest,
        )
        from ess.livedata.handlers.detector_view_specs import (
            register_detector_view_spec,
        )

        instrument = Instrument(name="test_instrument")
        source_names = ["detector1", "detector2"]
        handle = register_detector_view_spec(
            instrument=instrument,
            projection="xy_plane",
            source_names=source_names,
        )

        reg = instrument.workflow_factory.registration(handle.workflow_id)
        by_stream = {ci.stream_name: ci for ci in reg.context_bindings}
        assert set(by_stream) == {'roi_rectangle', 'roi_polygon'}
        assert by_stream['roi_rectangle'].workflow_key is ROIRectangleRequest
        assert by_stream['roi_polygon'].workflow_key is ROIPolygonRequest
        # Defaults to the spec's source_names.
        assert by_stream['roi_rectangle'].dependent_sources == frozenset(source_names)
        assert by_stream['roi_polygon'].dependent_sources == frozenset(source_names)

    def test_roi_context_binding_resolver_prefixes_with_job_id(self) -> None:
        from ess.livedata.handlers.detector_view_specs import (
            register_detector_view_spec,
        )

        instrument = Instrument(name="test_instrument")
        handle = register_detector_view_spec(
            instrument=instrument,
            projection="xy_plane",
            source_names=["detector1"],
        )
        reg = instrument.workflow_factory.registration(handle.workflow_id)
        rect = next(
            ci for ci in reg.context_bindings if ci.stream_name == 'roi_rectangle'
        )
        job_id = JobId(source_name='detector1', job_number=uuid.uuid4())
        assert rect.stream_resolver is not None
        assert (
            rect.stream_resolver(job_id, rect.stream_name) == f"{job_id}/roi_rectangle"
        )

    def test_roi_context_binding_seed_factory_yields_empty_geometry_message(
        self,
    ) -> None:
        import scipp as sc

        from ess.livedata.config import models
        from ess.livedata.core.message import StreamKind
        from ess.livedata.handlers.detector_view_specs import (
            register_detector_view_spec,
        )

        instrument = Instrument(name="test_instrument")
        handle = register_detector_view_spec(
            instrument=instrument,
            projection="xy_plane",
            source_names=["detector1"],
        )
        reg = instrument.workflow_factory.registration(handle.workflow_id)
        seeds = {ci.stream_name: ci.seed_factory for ci in reg.context_bindings}
        job_id = JobId(source_name='detector1', job_number=uuid.uuid4())

        rect_msg = seeds['roi_rectangle'](job_id)
        assert rect_msg.stream.kind is StreamKind.LIVEDATA_ROI
        assert rect_msg.stream.name == f"{job_id}/roi_rectangle"
        assert sc.identical(
            rect_msg.value, models.RectangleROI.to_concatenated_data_array({})
        )

        poly_msg = seeds['roi_polygon'](job_id)
        assert poly_msg.stream.kind is StreamKind.LIVEDATA_ROI
        assert poly_msg.stream.name == f"{job_id}/roi_polygon"
        assert sc.identical(
            poly_msg.value, models.PolygonROI.to_concatenated_data_array({})
        )

    def test_logical_view_with_roi_support_adds_context_bindings(self) -> None:
        from ess.livedata.handlers.detector_view.types import (
            ROIPolygonRequest,
            ROIRectangleRequest,
        )

        instrument = Instrument(name="test_instrument")
        handle = instrument.add_logical_view(
            name='custom_view',
            title='Custom View',
            description='',
            source_names=['detector1'],
            roi_support=True,
        )
        reg = instrument.workflow_factory.registration(handle.workflow_id)
        by_stream = {ci.stream_name: ci for ci in reg.context_bindings}
        assert by_stream['roi_rectangle'].workflow_key is ROIRectangleRequest
        assert by_stream['roi_polygon'].workflow_key is ROIPolygonRequest

    def test_logical_view_without_roi_support_has_no_context_bindings(self) -> None:
        instrument = Instrument(name="test_instrument")
        handle = instrument.add_logical_view(
            name='no_roi_view',
            title='No ROI',
            description='',
            source_names=['detector1'],
            roi_support=False,
        )
        reg = instrument.workflow_factory.registration(handle.workflow_id)
        assert reg.context_bindings == ()


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
        'unit', [WavelengthUnit.ANGSTROM, WavelengthUnit.NANOMETER]
    )
    def test_wavelength_range_preserves_user_unit(self, unit: WavelengthUnit):
        params = DetectorViewParams(
            coordinate_mode=CoordinateModeSettings(mode='wavelength'),
            wavelength_range=WavelengthRangeFilter(
                enabled=True, start=1.0, stop=5.0, unit=unit
            ),
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
