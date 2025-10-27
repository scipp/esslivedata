# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest

from ess.livedata.config.instrument import Instrument
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
