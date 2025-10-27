# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest

from ess.livedata.config.instrument import Instrument
from ess.livedata.handlers.workflow_factory import SpecHandle


class TestRegisterDetectorViewSpecs:
    """Test the lightweight register_detector_view_specs() function."""

    def test_register_detector_view_specs_returns_handles(self):
        """Test that register_detector_view_specs() returns handles."""
        from ess.livedata.handlers.detector_view_specs import (
            register_detector_view_spec,
        )

        instrument = Instrument(name="test_instrument")
        source_names = ["detector1", "detector2"]

        handles = register_detector_view_spec(
            instrument=instrument, projection="xy_plane", source_names=source_names
        )

        assert isinstance(handles, dict)
        assert "view" in handles
        assert "roi" in handles
        assert isinstance(handles["view"], SpecHandle)
        assert isinstance(handles["roi"], SpecHandle)

    def test_register_multiple_projections_via_separate_calls(self):
        """Test registering multiple projections via separate calls."""
        from ess.livedata.handlers.detector_view_specs import (
            register_detector_view_spec,
        )

        instrument = Instrument(name="test_instrument")
        source_names = ["detector1"]

        xy_handles = register_detector_view_spec(
            instrument=instrument,
            projection="xy_plane",
            source_names=source_names,
        )
        cylinder_handles = register_detector_view_spec(
            instrument=instrument,
            projection="cylinder_mantle_z",
            source_names=source_names,
        )

        assert isinstance(xy_handles["view"], SpecHandle)
        assert isinstance(cylinder_handles["view"], SpecHandle)

        # Verify both are registered in the factory
        xy_id = xy_handles["view"].workflow_id
        cylinder_id = cylinder_handles["view"].workflow_id
        assert xy_id in instrument.workflow_factory
        assert cylinder_id in instrument.workflow_factory

    def test_specs_registered_in_workflow_factory(self):
        """Test that specs are actually registered in the workflow factory."""
        from ess.livedata.handlers.detector_view_specs import (
            register_detector_view_spec,
        )

        instrument = Instrument(name="test_instrument")
        source_names = ["detector1"]

        handles = register_detector_view_spec(
            instrument=instrument, projection="xy_plane", source_names=source_names
        )

        # Verify specs are in the workflow factory
        view_id = handles["view"].workflow_id
        roi_id = handles["roi"].workflow_id

        assert view_id in instrument.workflow_factory
        assert roi_id in instrument.workflow_factory

        # Verify spec details
        view_spec = instrument.workflow_factory[view_id]
        assert view_spec.instrument == "test_instrument"
        assert view_spec.namespace == "detector_data"
        assert view_spec.name == "detector_xy_projection"
        assert view_spec.source_names == source_names

        roi_spec = instrument.workflow_factory[roi_id]
        assert roi_spec.name == "detector_xy_projection_roi"

    def test_specs_have_correct_params(self):
        """Test that registered specs have the correct params types."""
        from ess.livedata.handlers.detector_view_specs import (
            DetectorViewParams,
            ROIHistogramParams,
            register_detector_view_spec,
        )

        instrument = Instrument(name="test_instrument")
        source_names = ["detector1"]

        handles = register_detector_view_spec(
            instrument=instrument, projection="xy_plane", source_names=source_names
        )

        view_id = handles["view"].workflow_id
        roi_id = handles["roi"].workflow_id

        view_spec = instrument.workflow_factory[view_id]
        roi_spec = instrument.workflow_factory[roi_id]

        # Check params are set correctly
        assert view_spec.params is DetectorViewParams
        assert roi_spec.params is ROIHistogramParams

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
