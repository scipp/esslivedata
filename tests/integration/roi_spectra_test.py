# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration test for ROI spectra: published ROIs drive backend spectra."""

import pytest

from ess.livedata.config.models import Interval, RectangleROI
from ess.livedata.config.roi_names import get_roi_mapper
from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.workflows.detector_view_specs import DetectorViewParams
from tests.integration.conftest import IntegrationEnv
from tests.integration.helpers import (
    get_output_data,
    wait_for_backend_condition,
    wait_for_job_data,
)


@pytest.mark.integration
@pytest.mark.services('detector')
def test_roi_spectra_follow_published_rois(integration_env: IntegrationEnv) -> None:
    """
    ROI spectra results arrive for published ROIs and follow ROI changes.

    Publishes rectangle ROIs the way the dashboard does (ROIPublisher to the
    LIVEDATA_ROI topic, addressed to the workflow's current job) and asserts
    the detector view's roi_spectra output tracks the published set.
    """
    backend = integration_env.backend
    workflow_id = WorkflowId(instrument='dummy', name='panel_0_xy', version=1)
    source_name = 'panel_0'

    job_ids = backend.workflow_controller.start_workflow(
        workflow_id=workflow_id,
        source_names=[source_name],
        config=DetectorViewParams(),
    )
    wait_for_job_data(backend, workflow_id, job_ids, timeout=60.0)

    geometry = get_roi_mapper().geometry_for_type('rectangle')
    assert geometry is not None

    def roi_spectra_count() -> int | None:
        data = get_output_data(
            backend, workflow_id, source_name, 'roi_spectra_cumulative'
        )
        if data is None:
            return None
        return data.sizes['roi']

    # Publish one ROI (pixel indices on the 128x128 panel_0 logical view).
    roi_a = RectangleROI(x=Interval(min=10, max=60), y=Interval(min=10, max=60))
    backend.roi_publisher.publish(
        workflow_id=workflow_id,
        source_name=source_name,
        rois={0: roi_a},
        geometry=geometry,
    )
    wait_for_backend_condition(backend, lambda: roi_spectra_count() == 1, timeout=30.0)
    spectra = get_output_data(
        backend, workflow_id, source_name, 'roi_spectra_cumulative'
    )
    assert list(spectra.coords['roi'].values) == [0]

    # Change the ROI selection: the spectra must follow.
    roi_b = RectangleROI(x=Interval(min=70, max=120), y=Interval(min=70, max=120))
    backend.roi_publisher.publish(
        workflow_id=workflow_id,
        source_name=source_name,
        rois={0: roi_a, 1: roi_b},
        geometry=geometry,
    )
    wait_for_backend_condition(backend, lambda: roi_spectra_count() == 2, timeout=30.0)
    spectra = get_output_data(
        backend, workflow_id, source_name, 'roi_spectra_cumulative'
    )
    assert list(spectra.coords['roi'].values) == [0, 1]
