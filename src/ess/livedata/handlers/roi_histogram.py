# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""ROI-based time-of-arrival histogram accumulation."""

from __future__ import annotations

import scipp as sc

from ess.reduce.live.roi import ROIFilter

from ..config import models


class ROIHistogram:
    """
    Manages TOA histogram accumulation for a single Region of Interest.

    This class combines ROI spatial filtering, TOA histogram binning,
    and cumulative tracking for workflow orchestration.
    """

    def __init__(
        self,
        *,
        toa_edges: sc.Variable,
        roi_filter: ROIFilter,
        model: models.ROI,
    ):
        """
        Initialize ROI histogram.

        Parameters
        ----------
        toa_edges:
            Time-of-arrival bin edges for histogramming.
        roi_filter:
            ROI filter for spatial filtering of detector events.
        model:
            ROI configuration model (RectangleROI, PolygonROI, or EllipseROI).
        """
        self._roi_filter = roi_filter
        self._chunks: list[sc.DataArray] = []
        self._edges = toa_edges
        self._edges_ns = toa_edges.to(unit='ns')
        self._model = model
        self._cumulative: sc.DataArray | None = None
        self._dim = 'time_of_arrival'
        # Configure ROI filter from model
        self._configure_filter(model)

    @property
    def model(self) -> models.ROI:
        """Get the ROI configuration model."""
        return self._model

    @property
    def cumulative(self) -> sc.DataArray | None:
        """Get the cumulative histogram, or None if not yet accumulated."""
        return self._cumulative

    def _configure_filter(
        self, roi: models.RectangleROI | models.PolygonROI | models.EllipseROI
    ) -> None:
        """Configure the ROI filter from an ROI model (internal helper)."""
        if isinstance(roi, models.RectangleROI):
            y, x = self._roi_filter._indices.dims
            intervals = roi.get_bounds(x_dim=x, y_dim=y)
            self._roi_filter.set_roi_from_intervals(sc.DataGroup(intervals))
        elif isinstance(roi, models.PolygonROI):
            y, x = self._roi_filter._indices.dims
            polygon = _polygon_model_to_dict(roi, x_dim=x, y_dim=y)
            self._roi_filter.set_roi_from_polygon(polygon)
        else:
            roi_type = type(roi).__name__
            raise ValueError(f"Unsupported ROI type: {roi_type}")

    def add_data(self, data: sc.DataArray) -> None:
        """
        Add detector data for histogram accumulation.

        Parameters
        ----------
        data:
            Detector event data (binned DataArray from GroupIntoPixels).
        """
        filtered, scale = self._roi_filter.apply(data)
        filtered_with_weights = filtered.bins.assign_coords(
            **{self._dim: filtered.bins.data}
        ).bins.assign(sc.bins_like(filtered, scale))
        chunk = filtered_with_weights.hist(
            **{self._dim: self._edges_ns}, dim=filtered.dim
        )
        # Set unit to counts (histogram of weighted events represents counts)
        chunk.data.unit = 'counts'
        self._chunks.append(chunk)

    def _empty_histogram(self) -> sc.DataArray:
        """Create an empty histogram with the configured edges."""
        return sc.DataArray(
            data=sc.zeros(
                dims=[self._dim],
                shape=[len(self._edges) - 1],
                unit='counts',
            ),
            coords={self._dim: self._edges},
        )

    def get_delta(self) -> sc.DataArray:
        """
        Get histogram for current accumulation period and update cumulative.

        Returns the histogram accumulated since the last call to get_delta(),
        and adds it to the cumulative histogram.

        Returns
        -------
        :
            Histogram of events accumulated in this period.
        """
        if not self._chunks:
            delta = self._empty_histogram()
        else:
            delta = sc.reduce(self._chunks).sum()
            self._chunks.clear()
            delta.coords[self._dim] = self._edges

        if self._cumulative is None:
            self._cumulative = delta.copy()
        else:
            self._cumulative += delta
        return delta

    def clear(self) -> None:
        """Clear both chunks and cumulative data, preserving configuration."""
        self._chunks.clear()
        self._cumulative = None


def _polygon_model_to_dict(
    roi: models.PolygonROI, *, x_dim: str, y_dim: str
) -> dict[str, sc.Variable | list[float]]:
    """
    Convert a PolygonROI model to the dict format expected by ROIFilter.

    When unit is None (pixel indices), returns list[float] for index-based selection.
    When unit is set (physical coords), returns sc.Variable for coord-based selection.
    """
    if roi.x_unit is None:
        x_vertices: sc.Variable | list[float] = roi.x
    else:
        x_vertices = sc.array(dims=['vertex'], values=roi.x, unit=roi.x_unit)

    if roi.y_unit is None:
        y_vertices: sc.Variable | list[float] = roi.y
    else:
        y_vertices = sc.array(dims=['vertex'], values=roi.y, unit=roi.y_unit)

    return {x_dim: x_vertices, y_dim: y_vertices}
