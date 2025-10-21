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
from ..handlers.workflow_factory import SpecHandle


class DetectorViewParams(pydantic.BaseModel):
    """Parameters for detector view workflows."""

    pixel_weighting: models.PixelWeighting = pydantic.Field(
        title="Pixel Weighting",
        description="Whether to apply pixel weighting based on the number of pixels "
        "contributing to each screen pixel.",
        default=models.PixelWeighting(
            enabled=False, method=models.WeightingMethod.PIXEL_NUMBER
        ),
    )
    toa_range: parameter_models.TOARange = pydantic.Field(
        title="Time of Arrival Range",
        description="Time of arrival range for detector data.",
        default=parameter_models.TOARange(),
    )


class LimitedRange(parameter_models.RangeModel):
    """Model for a limited range between 0 and 1."""

    start: float = parameter_models.Field(
        ge=0.0, le=1.0, default=0.0, description="Start of the range."
    )
    stop: float = parameter_models.Field(
        ge=0.0, le=1.0, default=1.0, description="Stop of the range."
    )


class ROIHistogramParams(pydantic.BaseModel):
    """Parameters for ROI histogram workflows."""

    x_range: LimitedRange = pydantic.Field(
        title="X Range",
        description="X range of the ROI as a fraction of the viewport.",
        default=LimitedRange(start=0.0, stop=1.0),
    )
    y_range: LimitedRange = pydantic.Field(
        title="Y Range",
        description="Y range of the ROI as a fraction of the viewport.",
        default=LimitedRange(start=0.0, stop=1.0),
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


def register_detector_view_specs(
    *,
    instrument: Instrument,
    projections: list[Literal["xy_plane", "cylinder_mantle_z"]],
    source_names: list[str],
) -> dict[str, dict[str, SpecHandle]]:
    """
    Register detector view specs for given projections.

    This is a lightweight helper that registers workflow specs without creating
    the actual detector view objects (which require heavy ess.reduce imports).

    Parameters
    ----------
    instrument:
        Instrument to register specs with.
    projections:
        List of projection types to register specs for.
    source_names:
        List of detector source names.

    Returns
    -------
    Dictionary mapping projection name to dict with 'view' and 'roi' handles.
    """
    handles = {}

    for projection in projections:
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

        # Register view spec
        view_handle = instrument.register_spec(
            namespace="detector_data",
            name=name,
            version=1,
            title=title,
            description=description,
            source_names=source_names,
            params=DetectorViewParams,
        )

        # Register ROI histogram spec
        roi_handle = instrument.register_spec(
            namespace="detector_data",
            name=f"{name}_roi",
            version=1,
            title=f"ROI Histogram: {title}",
            description=f"ROI Histogram for {description}",
            source_names=source_names,
            params=ROIHistogramParams,
        )

        handles[projection] = {"view": view_handle, "roi": roi_handle}

    return handles
