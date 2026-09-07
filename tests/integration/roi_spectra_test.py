# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration test for ROI spectra: published ROIs drive backend spectra."""

import pytest

from ess.livedata.config.models import Interval, RectangleROI
from ess.livedata.config.roi_names import get_roi_mapper
from ess.livedata.config.workflow_spec import JobId, WorkflowId
from ess.livedata.workflows.detector_view_specs import DetectorViewParams
from tests.integration.conftest import IntegrationEnv
from tests.integration.helpers import (
    get_output_data,
    topic_high_watermark,
    wait_for_backend_condition,
    wait_for_job_data,
    wait_for_watermark_advance,
)


@pytest.mark.integration
@pytest.mark.services('detector')
def test_roi_selection_drives_spectra_across_the_job_lifecycle(
    integration_env: IntegrationEnv,
) -> None:
    """
    An ROI selection is a property of the view, outliving the jobs computing it.

    Publishes rectangle ROIs the way the dashboard does (ROIPublisher to the
    LIVEDATA_ROI topic, addressed to the view) and asserts the detector view's
    roi_spectra output tracks the published set in the three cases the
    view-scoped stream name has to serve: a selection made before any job
    exists, one changed while a job runs, and one that must survive a recommit
    without being republished.
    """
    backend = integration_env.backend
    instrument = integration_env.instrument
    workflow_id = WorkflowId(instrument='dummy', name='panel_0_xy', version=1)
    source_name = 'panel_0'
    geometry = get_roi_mapper().geometry_for_type('rectangle')
    assert geometry is not None

    def roi_spectra_count() -> int | None:
        data = get_output_data(
            backend, workflow_id, source_name, 'roi_spectra_cumulative'
        )
        if data is None:
            return None
        return data.sizes['roi']

    def start() -> list[JobId]:
        return backend.workflow_controller.start_workflow(
            workflow_id=workflow_id,
            source_names=[source_name],
            config=DetectorViewParams(),
        )

    # Publish before any job exists. The backend accumulates ROI streams by
    # name, independently of jobs, so the request is latched and handed to the
    # job as it activates (pixel indices on the 128x128 panel_0 logical view).
    roi_topic = f'{instrument}_livedata_roi'
    before_publish = topic_high_watermark(roi_topic)
    roi_a = RectangleROI(x=Interval(min=10, max=60), y=Interval(min=10, max=60))
    backend.roi_publisher.publish(
        workflow_id=workflow_id,
        source_name=source_name,
        rois={0: roi_a},
        geometry=geometry,
    )
    # Wait for the broker to hold it, so the job cannot activate before the
    # request the latch is supposed to supply is even produced.
    wait_for_watermark_advance(roi_topic, since=before_publish, timeout=30.0)

    job_ids = start()
    wait_for_job_data(backend, workflow_id, job_ids, timeout=60.0)
    wait_for_backend_condition(backend, lambda: roi_spectra_count() == 1, timeout=30.0)
    spectra = get_output_data(
        backend, workflow_id, source_name, 'roi_spectra_cumulative'
    )
    assert list(spectra.coords['roi'].values) == [0]

    # Change the ROI selection while the job runs: the spectra must follow.
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

    # Recommit: a new job_number, and no republished ROI. Committing evicts the
    # workflow's buffers (ActiveJobRegistry.begin_generation) before returning,
    # so the output reads as absent until the new generation produces its own.
    start()
    assert roi_spectra_count() is None
    wait_for_backend_condition(backend, lambda: roi_spectra_count() == 2, timeout=60.0)
    spectra = get_output_data(
        backend, workflow_id, source_name, 'roi_spectra_cumulative'
    )
    assert list(spectra.coords['roi'].values) == [0, 1]
