# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import DETECTORS, WorkflowSpec
from ess.livedata.dashboard.roi_selection import (
    has_roi_dim,
    select_roi,
    selected_output_template,
)
from ess.livedata.workflows.detector_view_specs import DetectorViewOutputs


@pytest.fixture
def detector_view_spec() -> WorkflowSpec:
    return WorkflowSpec(
        instrument='dummy',
        name='detector_view',
        version=1,
        title='Detector View',
        description='Detector view with ROIs.',
        outputs=DetectorViewOutputs,
        group=DETECTORS,
    )


def _per_roi_totals() -> sc.DataArray:
    """Σ ROI data for the drawn ROIs 0 and 4, of 2 and 8 detector pixels."""
    return sc.DataArray(
        sc.array(dims=['roi'], values=[10.0, 40.0], unit='counts'),
        coords={
            'roi': sc.array(dims=['roi'], values=[0, 4], unit=None, dtype='int32'),
            'detector_pixels': sc.array(dims=['roi'], values=[2.0, 8.0]),
        },
    )


class TestSelectRoi:
    def test_returns_the_values_of_the_roi(self):
        selected = select_roi(_per_roi_totals(), 4)

        assert sc.identical(selected.data, sc.scalar(40.0, unit='counts'))

    def test_coords_along_roi_become_scalar_coords(self):
        selected = select_roi(_per_roi_totals(), 4)

        assert selected.coords['roi'].value == 4
        assert sc.identical(selected.coords['detector_pixels'], sc.scalar(8.0))

    def test_selects_from_a_history(self):
        totals = _per_roi_totals()
        history = sc.concat([totals, 2 * totals], dim='time')
        history.coords['time'] = sc.array(dims=['time'], values=[1, 2], unit='s')

        selected = select_roi(history, 4)

        assert selected.dims == ('time',)
        assert sc.identical(
            selected.data, sc.array(dims=['time'], values=[40.0, 80.0], unit='counts')
        )
        assert sc.identical(selected.coords['detector_pixels'], sc.scalar(8.0))

    def test_returns_none_for_an_roi_that_is_not_drawn(self):
        assert select_roi(_per_roi_totals(), 1) is None


class TestSelectedOutputTemplate:
    def test_roi_totals_become_a_scalar(self, detector_view_spec):
        template = selected_output_template(detector_view_spec, 'roi_total_in_range', 4)

        assert template.dims == ()

    def test_keeps_detector_pixels_as_scalar_coord(self, detector_view_spec):
        template = selected_output_template(detector_view_spec, 'roi_total_in_range', 4)

        assert template.coords['detector_pixels'].dims == ()

    def test_roi_spectra_become_one_spectrum(self, detector_view_spec):
        template = selected_output_template(detector_view_spec, 'roi_spectra', 4)

        assert template.dims == ('time_of_arrival',)

    def test_without_roi_the_template_is_unchanged(self, detector_view_spec):
        template = selected_output_template(
            detector_view_spec, 'roi_total_in_range', None
        )

        assert sc.identical(
            template, detector_view_spec.get_output_template('roi_total_in_range')
        )

    def test_unknown_view_has_no_template(self, detector_view_spec):
        assert selected_output_template(detector_view_spec, 'unknown', 4) is None


@pytest.mark.parametrize(
    ('view_name', 'expected'),
    [
        ('roi_total_in_range', True),
        ('roi_spectra', True),
        ('total_counts', False),
        ('image', False),
        ('unknown', False),
    ],
)
def test_has_roi_dim(detector_view_spec, view_name, expected):
    assert has_roi_dim(detector_view_spec, view_name) is expected
