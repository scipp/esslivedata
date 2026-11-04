# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Output model of the area detector transmission workflow.

Kept free of heavy dependencies so the dashboard can import it; the workflow
itself lives in :mod:`ess.livedata.workflows.area_detector_transmission`.
"""

from __future__ import annotations

import pydantic
import scipp as sc

from ..config.workflow_spec import WorkflowOutputsBase


def _image_template(unit: str) -> sc.DataArray:
    return sc.DataArray(sc.zeros(dims=['dim_0', 'dim_1'], shape=[0, 0], unit=unit))


class AreaDetectorTransmissionOutputs(WorkflowOutputsBase):
    """Outputs of the area detector transmission workflow.

    Each output is published only when it changed in the latest update, and
    the mean images and the transmission only once the frames they need have been
    seen.
    """

    sample: sc.DataArray = pydantic.Field(
        default_factory=lambda: _image_template('counts'),
        title='Sample',
        description='Mean sample frame (image key 0) since the start of the run.',
    )
    open_beam: sc.DataArray = pydantic.Field(
        default_factory=lambda: _image_template('counts'),
        title='Open beam',
        description='Mean open beam frame (image key 1) since the start of the run.',
    )
    dark: sc.DataArray = pydantic.Field(
        default_factory=lambda: _image_template('counts'),
        title='Dark',
        description='Mean dark frame (image key 2) since the start of the run.',
    )
    transmission: sc.DataArray = pydantic.Field(
        default_factory=lambda: _image_template('dimensionless'),
        title='Transmission',
        description=(
            '(sample - dark) / (open beam - dark), from the mean frames of each '
            'type. NaN where open beam does not exceed dark. Available once '
            'frames of all three types have been received.'
        ),
    )
    dropped_frames: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(sc.scalar(0, unit='counts')),
        title='Dropped frames',
        description=(
            'Frames not used: image key invalid, background, unknown, or not '
            'yet received when the frame arrived.'
        ),
    )
