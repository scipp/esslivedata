# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import subprocess
import sys

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
    DetectorViewParams,
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
        params=DetectorViewParams,
        outputs=DetectorViewOutputs,
    )


ROI_KEYS = {'roi_rectangle': ROIRectangleRequest, 'roi_polygon': ROIPolygonRequest}


class TestBindROIRequests:
    """ROI request streams are spec-scope, non-gating context bindings."""

    @pytest.fixture
    def instrument(self) -> Instrument:
        instrument = Instrument(name="test_instrument")
        handle = _register_detector_view(instrument, ["detector1", "detector2"])
        bind_roi_requests(handle)
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
        assert instrument.resolve_context_keys(workflow_id, source) == {
            roi_stream_name(workflow_id, source, key): request
            for key, request in ROI_KEYS.items()
        }

    @pytest.mark.parametrize('source', ['detector1', 'detector2'])
    def test_roi_streams_do_not_gate(
        self, instrument: Instrument, workflow_id: WorkflowId, source: str
    ) -> None:
        assert instrument.resolve_gating_streams(workflow_id, source) == set()

    def test_views_sharing_a_source_get_distinct_streams(self) -> None:
        views = [WorkflowId(instrument='d', name=n, version=1) for n in ('xy', 'cyl')]
        names = {
            roi_stream_name(view, 'det', key) for view in views for key in ROI_KEYS
        }
        assert len(names) == 2 * len(ROI_KEYS)

    def test_validate_rejects_roi_outputs_without_request_bindings(self) -> None:
        instrument = Instrument(name="test_instrument")
        handle = _register_detector_view(instrument, ["detector1"])
        with pytest.raises(ValueError, match="binds no request stream"):
            instrument.validate()
        bind_roi_requests(handle)
        instrument.validate()


@pytest.fixture(scope='module')
def tbl() -> Instrument:
    get_config('tbl')
    instrument = instrument_registry['tbl']
    instrument.load_factories()
    return instrument


class TestLogicalViewROIBindings:
    """``load_factories`` binds ROI requests for logical views with ROI support."""

    def test_logical_view_with_roi_support_binds_roi_requests(
        self, tbl: Instrument
    ) -> None:
        workflow_id = WorkflowId(instrument='tbl', name='he3_detector_view', version=1)
        reg = tbl.workflow_factory.registration(workflow_id)
        assert {(b.stream_name, b.workflow_key) for b in reg.context_bindings} == {
            (roi_stream_name(workflow_id, source, key), request)
            for source in reg.spec.source_names
            for key, request in ROI_KEYS.items()
        }
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
