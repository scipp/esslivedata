# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import subprocess
import sys

import pytest

from ess.livedata.config.instrument import Instrument
from ess.livedata.config.workflow_spec import DETECTORS, WorkflowId
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

    def test_roi_aux_render_scopes_names_to_the_view(self) -> None:
        workflow_id = WorkflowId(instrument='dummy', name='detector_view', version=1)
        assert DetectorROIAuxSources().render(workflow_id, 'detector1') == {
            'roi_rectangle': 'dummy/detector_view/1/detector1/roi_rectangle',
            'roi_polygon': 'dummy/detector_view/1/detector1/roi_polygon',
        }

    def test_roi_aux_render_is_independent_of_the_job_generation(self) -> None:
        """The rendered names carry no job_number, so a restart reuses them."""
        workflow_id = WorkflowId(instrument='dummy', name='detector_view', version=1)
        aux = DetectorROIAuxSources()
        assert aux.render(workflow_id, 'detector1') == aux.render(
            workflow_id, 'detector1'
        )

    def test_roi_aux_render_distinguishes_views_sharing_a_source(self) -> None:
        """Two ROI-supporting views of one detector must not share a stream."""
        source = 'detector1'
        aux = DetectorROIAuxSources()
        xy = aux.render(WorkflowId(instrument='d', name='xy_view', version=1), source)
        cyl = aux.render(WorkflowId(instrument='d', name='cyl_view', version=1), source)
        assert set(xy.values()).isdisjoint(cyl.values())

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
