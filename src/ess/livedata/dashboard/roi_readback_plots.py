# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Plotters for ROI readback data from workflows.

These plotters display ROI shapes (rectangles, polygons) that come from
workflow outputs, with per-shape colors based on ROI index.
"""

from __future__ import annotations

from typing import Any

import holoviews as hv
import pydantic
import scipp as sc

from ess.livedata.config.models import (
    ROI,
    PolygonROI,
    RectangleROI,
)
from ess.livedata.config.workflow_spec import ResultKey

from .plots import Plotter


class ROIReadbackStyle(pydantic.BaseModel):
    """Style parameters for ROI readback plots."""

    fill_alpha: float = pydantic.Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        title="Fill Opacity",
        description="Fill transparency (0 = transparent, 1 = opaque)",
    )
    line_width: float = pydantic.Field(
        default=2.0,
        ge=0.0,
        le=10.0,
        title="Line Width",
        description="Line width in pixels",
    )


class RectanglesReadbackParams(pydantic.BaseModel):
    """Parameters for rectangles readback plotter."""

    style: ROIReadbackStyle = pydantic.Field(
        default_factory=ROIReadbackStyle,
        title="Appearance",
        description="Visual styling options.",
    )


class PolygonsReadbackParams(pydantic.BaseModel):
    """Parameters for polygons readback plotter."""

    style: ROIReadbackStyle = pydantic.Field(
        default_factory=ROIReadbackStyle,
        title="Appearance",
        description="Visual styling options.",
    )


class RectanglesReadbackPlotter(Plotter):
    """
    Plotter for ROI rectangle readback data from workflows.

    Takes ROI readback DataArrays with roi_index coordinate and renders
    rectangles with per-shape colors based on the ROI index.
    """

    def __init__(self, params: RectanglesReadbackParams) -> None:
        super().__init__()
        self._params = params
        self._colors = hv.Cycle.default_cycles["default_colors"]

    @classmethod
    def from_params(cls, params: RectanglesReadbackParams) -> RectanglesReadbackPlotter:
        """Create plotter from params."""
        return cls(params)

    def plot(self, data: sc.DataArray, data_key: ResultKey, **kwargs) -> hv.Rectangles:
        """
        Create rectangles from ROI readback DataArray.

        Parameters
        ----------
        data:
            DataArray with ROI readback format (roi_index, x, y coordinates).
        data_key:
            Key identifying this data.
        **kwargs:
            Unused additional arguments.

        Returns
        -------
        :
            HoloViews Rectangles element with per-shape colors.
        """
        del kwargs  # Unused

        # Parse ROI data
        rois = ROI.from_concatenated_data_array(data)

        # Filter to only RectangleROI instances
        rect_rois: dict[int, RectangleROI] = {
            idx: roi for idx, roi in rois.items() if isinstance(roi, RectangleROI)
        }

        if not rect_rois:
            return hv.Rectangles([])

        # Convert to HoloViews format with colors
        rectangles = self._to_hv_data(rect_rois)

        style = self._params.style
        return hv.Rectangles(rectangles, vdims=['color']).opts(
            color='color',
            line_color='color',
            fill_alpha=style.fill_alpha,
            line_width=style.line_width,
            # Rectangles (as opposed to Polygons) show legend by default, which we
            # probably do not want.
            show_legend=False,
        )

    def _to_hv_data(
        self, rois: dict[int, RectangleROI]
    ) -> list[tuple[float, float, float, float, str]]:
        """
        Convert RectangleROI instances to HoloViews Rectangles format.

        Colors are assigned by ROI index for stable identity across updates.

        Parameters
        ----------
        rois:
            Dictionary mapping ROI index to RectangleROI.

        Returns
        -------
        :
            List of (x0, y0, x1, y1, color) tuples for HoloViews Rectangles.
        """
        rectangles = []
        for idx in sorted(rois.keys()):
            roi = rois[idx]
            color = self._colors[idx % len(self._colors)]
            rectangles.append(
                (
                    float(roi.x.min),
                    float(roi.y.min),
                    float(roi.x.max),
                    float(roi.y.max),
                    color,
                )
            )
        return rectangles


class PolygonsReadbackPlotter(Plotter):
    """
    Plotter for ROI polygon readback data from workflows.

    Takes ROI readback DataArrays with roi_index coordinate and renders
    polygons with per-shape colors based on the ROI index.
    """

    def __init__(self, params: PolygonsReadbackParams) -> None:
        super().__init__()
        self._params = params
        self._colors = hv.Cycle.default_cycles["default_colors"]

    @classmethod
    def from_params(cls, params: PolygonsReadbackParams) -> PolygonsReadbackPlotter:
        """Create plotter from params."""
        return cls(params)

    def plot(self, data: sc.DataArray, data_key: ResultKey, **kwargs) -> hv.Polygons:
        """
        Create polygons from ROI readback DataArray.

        Parameters
        ----------
        data:
            DataArray with ROI readback format (roi_index, x, y coordinates).
        data_key:
            Key identifying this data.
        **kwargs:
            Unused additional arguments.

        Returns
        -------
        :
            HoloViews Polygons element with per-shape colors.
        """
        del kwargs  # Unused

        # Parse ROI data
        rois = ROI.from_concatenated_data_array(data)

        # Filter to only PolygonROI instances
        poly_rois: dict[int, PolygonROI] = {
            idx: roi for idx, roi in rois.items() if isinstance(roi, PolygonROI)
        }

        if not poly_rois:
            return hv.Polygons([])

        # Convert to HoloViews format with colors
        polygons = self._to_hv_data(poly_rois)

        style = self._params.style
        return hv.Polygons(polygons, vdims=['color']).opts(
            color='color',
            line_color='color',
            fill_alpha=style.fill_alpha,
            line_width=style.line_width,
        )

    def _to_hv_data(self, rois: dict[int, PolygonROI]) -> list[dict[str, Any]]:
        """
        Convert PolygonROI instances to HoloViews Polygons format.

        Colors are assigned by ROI index for stable identity across updates.

        Parameters
        ----------
        rois:
            Dictionary mapping ROI index to PolygonROI.

        Returns
        -------
        :
            List of dicts with 'x', 'y', 'color' for HoloViews Polygons.
        """
        polygons = []
        for idx in sorted(rois.keys()):
            roi = rois[idx]
            color = self._colors[idx % len(self._colors)]
            polygons.append(
                {
                    'x': [float(v) for v in roi.x],
                    'y': [float(v) for v in roi.y],
                    'color': color,
                }
            )
        return polygons


def _register_roi_readback_plotters() -> None:
    """Register ROI readback plotters with the plotter registry."""
    from .plotting import DataRequirements, plotter_registry

    plotter_registry.register_plotter(
        name='rectangles_readback',
        title='Rectangles (ROI Readback)',
        description='Display ROI rectangles from workflow output. '
        'Each rectangle is colored by its ROI index.',
        data_requirements=DataRequirements(
            min_dims=1,
            max_dims=1,
            required_coords=['roi_index', 'x', 'y'],
            required_dim_names=['bounds'],
            multiple_datasets=False,
        ),
        factory=RectanglesReadbackPlotter.from_params,
    )

    plotter_registry.register_plotter(
        name='polygons_readback',
        title='Polygons (ROI Readback)',
        description='Display ROI polygons from workflow output. '
        'Each polygon is colored by its ROI index.',
        data_requirements=DataRequirements(
            min_dims=1,
            max_dims=1,
            required_coords=['roi_index', 'x', 'y'],
            required_dim_names=['vertex'],
            multiple_datasets=False,
        ),
        factory=PolygonsReadbackPlotter.from_params,
    )
