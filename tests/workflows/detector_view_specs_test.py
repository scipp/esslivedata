# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import subprocess
import sys

import pydantic
import pytest

from ess.livedata.config.instrument import Instrument, instrument_registry
from ess.livedata.config.instruments import get_config
from ess.livedata.config.roi_names import roi_stream_name
from ess.livedata.config.workflow_spec import DETECTORS, WorkflowId
from ess.livedata.parameter_models import (
    TimeUnit,
    TOARange,
    WavelengthRangeFilter,
    WavelengthUnit,
)
from ess.livedata.workflows.detector_view import bind_roi_requests
from ess.livedata.workflows.detector_view.types import (
    ROIPolygonRequest,
    ROIRectangleRequest,
)
from ess.livedata.workflows.detector_view_specs import (
    CoordinateModeSettings,
    DetectorViewOutputs,
    DetectorViewOutputsBase,
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


def _register_detector_view(
    instrument: Instrument,
    source_names: list[str],
    outputs: type = DetectorViewOutputs,
):
    """Register a detector view spec the way an instrument's specs.py does."""
    return instrument.register_spec(
        group=DETECTORS,
        name='detector_xy_projection',
        version=1,
        title='Detector XY Projection',
        description='Projection of a detector bank onto an XY-plane.',
        source_names=source_names,
        params=DetectorViewParams,
        outputs=outputs,
    )


ROI_KEYS = {'roi_rectangle': ROIRectangleRequest, 'roi_polygon': ROIPolygonRequest}


class TestBindROIRequests:
    """ROI request streams are spec-scope, non-gating context bindings."""

    @pytest.fixture
    def instrument(self) -> Instrument:
        instrument = Instrument(name="test_instrument")
        _register_detector_view(instrument, ["detector1", "detector2"])
        bind_roi_requests(instrument.workflow_factory)
        return instrument

    @pytest.fixture
    def workflow_id(self, instrument: Instrument) -> WorkflowId:
        return WorkflowId(
            instrument='test_instrument', name='detector_xy_projection', version=1
        )

    @pytest.mark.parametrize('source', ['detector1', 'detector2'])
    def test_each_source_resolves_its_own_view_scoped_streams(
        self, instrument: Instrument, workflow_id: WorkflowId, source: str
    ) -> None:
        assert instrument.bound_context_keys(workflow_id, source) == {
            roi_stream_name(workflow_id, source, key): request
            for key, request in ROI_KEYS.items()
        }

    @pytest.mark.parametrize('source', ['detector1', 'detector2'])
    def test_roi_streams_do_not_gate(
        self, instrument: Instrument, workflow_id: WorkflowId, source: str
    ) -> None:
        assert instrument.bound_gating_streams(workflow_id, source) == set()

    def test_views_sharing_a_source_get_distinct_streams(self) -> None:
        views = [WorkflowId(instrument='d', name=n, version=1) for n in ('xy', 'cyl')]
        names = {
            roi_stream_name(view, 'det', key) for view in views for key in ROI_KEYS
        }
        assert len(names) == 2 * len(ROI_KEYS)

    def test_spec_without_roi_outputs_gets_no_bindings(self) -> None:
        instrument = Instrument(name="test_instrument")
        handle = _register_detector_view(
            instrument, ["detector1"], outputs=DetectorViewOutputsBase
        )
        bind_roi_requests(instrument.workflow_factory)
        reg = instrument.workflow_factory.registration(handle.workflow_id)
        assert reg.context_bindings == ()


@pytest.fixture(scope='module')
def tbl() -> Instrument:
    get_config('tbl')
    instrument = instrument_registry['tbl']
    instrument.load_factories()
    return instrument


@pytest.fixture(scope='module')
def dummy() -> Instrument:
    get_config('dummy')
    instrument = instrument_registry['dummy']
    instrument.load_factories()
    return instrument


def _expected_roi_bindings(reg) -> set[tuple[str, object]]:
    workflow_id = reg.spec.get_id()
    return {
        (roi_stream_name(workflow_id, source, key), request)
        for source in reg.spec.source_names
        for key, request in ROI_KEYS.items()
    }


class TestLoadFactoriesROIBindings:
    """``load_factories`` binds ROI requests for every spec declaring readbacks."""

    def test_hand_registered_view_declaring_roi_outputs_binds_roi_requests(
        self, dummy: Instrument
    ) -> None:
        workflow_id = WorkflowId(instrument='dummy', name='panel_0_xy', version=1)
        reg = dummy.workflow_factory.registration(workflow_id)
        bindings = {(b.stream_name, b.workflow_key) for b in reg.context_bindings}
        assert bindings == _expected_roi_bindings(reg)
        assert not any(b.gating for b in reg.context_bindings)

    def test_view_declaring_roi_free_outputs_has_no_bindings(
        self, dummy: Instrument
    ) -> None:
        workflow_id = WorkflowId(instrument='dummy', name='area_panel_xy', version=1)
        assert dummy.workflow_factory.registration(workflow_id).context_bindings == ()

    def test_logical_view_with_roi_support_binds_roi_requests(
        self, tbl: Instrument
    ) -> None:
        workflow_id = WorkflowId(instrument='tbl', name='he3_detector_view', version=1)
        reg = tbl.workflow_factory.registration(workflow_id)
        bindings = {(b.stream_name, b.workflow_key) for b in reg.context_bindings}
        assert bindings == _expected_roi_bindings(reg)
        assert not any(b.gating for b in reg.context_bindings)

    def test_logical_view_without_roi_support_has_no_bindings(
        self, tbl: Instrument
    ) -> None:
        workflow_id = WorkflowId(
            instrument='tbl', name='multiblade_detector_view', version=1
        )
        assert tbl.workflow_factory.registration(workflow_id).context_bindings == ()


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
    """A spec can restrict its params to time-of-arrival.

    Logical views run on sources without geometry, so there is no ``Ltotal``
    to index a wavelength lookup table with (ADR 0011); offering the mode
    would fail at job start. Everywhere else coordinate mode stays a parameter
    on one spec (ADR 0010).
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
