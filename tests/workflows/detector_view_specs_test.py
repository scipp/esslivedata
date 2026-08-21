# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import subprocess
import sys
import uuid

import pydantic
import pytest

from ess.livedata.config.instrument import Instrument
from ess.livedata.config.workflow_spec import DETECTORS, JobId
from ess.livedata.parameter_models import (
    TimeUnit,
    TOARange,
    WavelengthRangeFilter,
    WavelengthUnit,
)
from ess.livedata.workflows.detector_view_specs import (
    CoordinateModeSettings,
    DetectorROIAuxSources,
    DetectorViewOutputs,
    DetectorViewParams,
    SpectrumViewSpec,
    TOAOnlyDetectorViewParams,
    make_detector_view_params,
)


def test_import_does_not_pull_in_ess_reduce() -> None:
    """This module is imported to build the spec list, so it must stay cheap.

    Runs in a subprocess: in-process the check is worthless, since any earlier
    test may already have imported ess.reduce.
    """
    script = (
        "import sys, ess.livedata.workflows.detector_view_specs;"
        "leaked = sorted(m for m in sys.modules if m.startswith('ess.reduce'));"
        "assert not leaked, leaked"
    )
    subprocess.run([sys.executable, '-c', script], check=True)  # noqa: S603


def _register_detector_view(instrument: Instrument, source_names: list[str]):
    """Register a detector view spec the way an instrument's specs.py does."""
    return instrument.register_spec(
        group=DETECTORS,
        name='detector_xy_projection',
        version=1,
        title='Detector XY Projection',
        description='Projection of a detector bank onto an XY-plane.',
        source_names=source_names,
        aux_sources=DetectorROIAuxSources(),
        params=DetectorViewParams,
        outputs=DetectorViewOutputs,
    )


class TestDetectorROIAuxSources:
    """ROI auxiliary sources on detector view specs.

    ROI is an auxiliary source, not a gated context binding: the factory wires
    the ROI streams into ``set_context`` itself and the providers treat a
    missing/empty request as "no ROI selected", so there is nothing to gate.
    """

    def test_spec_exposes_roi_rectangle_and_polygon_aux_sources(self) -> None:
        instrument = Instrument(name="test_instrument")
        handle = _register_detector_view(instrument, ["detector1", "detector2"])

        spec = instrument.workflow_factory[handle.workflow_id]
        assert isinstance(spec.aux_sources, DetectorROIAuxSources)
        assert set(spec.aux_sources.inputs) == {'roi_rectangle', 'roi_polygon'}
        # ROI is not gated, so no context bindings are declared.
        reg = instrument.workflow_factory.registration(handle.workflow_id)
        assert reg.context_bindings == ()

    def test_roi_aux_render_prefixes_with_job_id(self) -> None:
        job_id = JobId(source_name='detector1', job_number=uuid.uuid4())
        assert DetectorROIAuxSources().render(job_id) == {
            'roi_rectangle': f"{job_id}/roi_rectangle",
            'roi_polygon': f"{job_id}/roi_polygon",
        }

    def test_logical_view_with_roi_support_adds_roi_aux_sources(self) -> None:
        instrument = Instrument(name="test_instrument")
        handle = instrument.add_logical_view(
            name='custom_view',
            title='Custom View',
            description='',
            source_names=['detector1'],
            roi_support=True,
        )
        spec = instrument.workflow_factory[handle.workflow_id]
        assert isinstance(spec.aux_sources, DetectorROIAuxSources)

    def test_logical_view_without_roi_support_has_no_aux_sources(self) -> None:
        instrument = Instrument(name="test_instrument")
        handle = instrument.add_logical_view(
            name='no_roi_view',
            title='No ROI',
            description='',
            source_names=['detector1'],
            roi_support=False,
        )
        spec = instrument.workflow_factory[handle.workflow_id]
        assert spec.aux_sources is None
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


class TestDetectorViewCoordinateModeRestriction:
    """Coordinate mode is a spec-level property, not a runtime choice.

    The wavelength path consumes a lookup table as gated context, and gating
    is resolved per ``(workflow_id, source_name)`` and never per parameter
    value, so the two modes need separate specs (ADR 0010).
    """

    def test_toa_only_params_reject_wavelength(self):
        with pytest.raises(pydantic.ValidationError):
            TOAOnlyDetectorViewParams(coordinate_mode={'mode': 'wavelength'})

    def test_toa_only_params_carry_no_wavelength_fields(self):
        assert 'wavelength_edges' not in TOAOnlyDetectorViewParams.model_fields
        assert 'wavelength_range' not in TOAOnlyDetectorViewParams.model_fields

    def test_both_modes_remain_available_on_the_unrestricted_model(self):
        for mode in ('toa', 'wavelength'):
            params = DetectorViewParams(coordinate_mode={'mode': mode})
            assert params.coordinate_mode.mode == mode

    def test_make_params_keeps_the_requested_base(self):
        params = make_detector_view_params(base=TOAOnlyDetectorViewParams)
        assert params is TOAOnlyDetectorViewParams

    def test_make_params_extends_the_requested_base_with_spectrum_params(self):
        class SpectrumParams(pydantic.BaseModel):
            value: int = 3

        spec = SpectrumViewSpec(
            transform=lambda da: da,
            output_dims=['group'],
            params_model=SpectrumParams,
        )
        params = make_detector_view_params(spec, base=TOAOnlyDetectorViewParams)
        assert issubclass(params, TOAOnlyDetectorViewParams)
        assert params().spectrum_params.value == 3
