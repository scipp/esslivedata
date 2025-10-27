# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Lightweight detector view spec registration.

This module provides spec registration for detector views WITHOUT importing
heavy dependencies like ess.reduce.live.raw. It should be imported by instrument
specs modules to register detector view specifications.

Factory implementations that use ess.reduce.live.raw are in detector_data_handler.py
and should only be imported by backend services.
"""

from __future__ import annotations

from typing import Literal

import pydantic

from .. import parameter_models
from ..config import models
from ..config.instrument import Instrument
from ..config.workflow_spec import AuxSourcesBase, JobId
from ..handlers.workflow_factory import SpecHandle


class DetectorViewParams(pydantic.BaseModel):
    pixel_weighting: models.PixelWeighting = pydantic.Field(
        title="Pixel Weighting",
        description="Whether to apply pixel weighting based on the number of pixels "
        "contributing to each screen pixel.",
        default=models.PixelWeighting(
            enabled=False, method=models.WeightingMethod.PIXEL_NUMBER
        ),
    )
    # TODO split out the enabled flag?
    toa_range: parameter_models.TOARange = pydantic.Field(
        title="Time of Arrival Range",
        description="Time of arrival range for detector data.",
        default=parameter_models.TOARange(),
    )
    toa_edges: parameter_models.TOAEdges = pydantic.Field(
        title="Time of Arrival Edges",
        description="Time of arrival edges for histogramming.",
        default=parameter_models.TOAEdges(
            start=0.0,
            stop=1000.0 / 14,
            num_bins=100,
            unit=parameter_models.TimeUnit.MS,
        ),
    )


class DetectorROIAuxSources(AuxSourcesBase):
    """
    Auxiliary source model for ROI configuration in detector workflows.

    Allows users to select between different ROI shapes (rectangle, polygon, ellipse).
    The render() method prefixes stream names with the job number to create job-specific
    ROI configuration streams, since each job instance needs its own ROI.
    """

    roi: Literal['rectangle', 'polygon', 'ellipse'] = pydantic.Field(
        default='rectangle',
        description='Shape to use for the region of interest (ROI).',
    )

    @pydantic.field_validator('roi')
    @classmethod
    def validate_roi_shape(cls, v: str) -> str:
        """Validate that only rectangle is currently supported."""
        if v != 'rectangle':
            raise ValueError(
                f"Currently only 'rectangle' ROI shape is supported, got '{v}'"
            )
        return v

    def render(self, job_id: JobId) -> dict[str, str]:
        """
        Render ROI stream name with job-specific prefix.

        Parameters
        ----------
        job_id:
            Job identifier containing source_name and job_number.

        Returns
        -------
        :
            Mapping from field name 'roi' to job-specific stream name in the
            format '{source_name}/{job_number}/roi_{shape}' (e.g.,
            'mantle/abc-123/roi_rectangle'). The source_name ensures ROI
            streams are unique per detector in multi-detector workflows where
            the same job_number is shared across detectors.
        """
        base = self.model_dump(mode='json')
        return {field: f"{job_id}/roi_{stream}" for field, stream in base.items()}


def register_detector_view_spec(
    *,
    instrument: Instrument,
    projection: Literal["xy_plane", "cylinder_mantle_z"],
    source_names: list[str],
) -> SpecHandle:
    """
    Register detector view specs for a given projection.

    This is a lightweight helper that registers workflow specs without creating
    the actual detector view objects.

    Parameters
    ----------
    instrument:
        Instrument to register specs with.
    projection:
        Projection type to register specs for.
    source_names:
        List of detector source names.

    Returns
    -------
    :
        A SpecHandle.
    """
    if projection == "xy_plane":
        name = "detector_xy_projection"
        title = "Detector XY Projection"
        description = "Projection of a detector bank onto an XY-plane."
    elif projection == "cylinder_mantle_z":
        name = "detector_cylinder_mantle_z"
        title = "Detector Cylinder Mantle Z Projection"
        description = (
            "Projection of a detector bank onto a cylinder mantle along Z-axis."
        )
    else:
        raise ValueError(f"Unsupported projection: {projection}")

    return instrument.register_spec(
        namespace="detector_data",
        name=name,
        version=1,
        title=title,
        description=description,
        source_names=source_names,
        aux_sources=DetectorROIAuxSources,
        params=DetectorViewParams,
    )
