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
        else:
            roi_type = type(roi).__name__
            raise ValueError(
                f"Only rectangle ROI is currently supported, got {roi_type}"
            )

    def _add_weights(self, data: sc.DataArray) -> None:
        """Add weights required for histogramming to binned data."""
        constituents = data.bins.constituents
        content = constituents['data']
        content.coords['time_of_arrival'] = content.data
        content.data = sc.ones(
            dims=content.dims, shape=content.shape, dtype='float32', unit='counts'
        )
        data.data = sc.bins(**constituents, validate_indices=False)

    def add_data(self, data: sc.DataArray) -> None:
        """
        Add detector data for histogram accumulation.

        Parameters
        ----------
        data:
            Detector event data (binned DataArray from GroupIntoPixels).
        """
        filtered, scale = self._roi_filter.apply(data)
        self._add_weights(filtered)
        filtered *= scale
        chunk = filtered.hist(time_of_arrival=self._edges_ns, dim=filtered.dim)
        self._chunks.append(chunk)

    def _empty_histogram(self) -> sc.DataArray:
        """Create an empty histogram with the configured edges."""
        return sc.DataArray(
            data=sc.zeros(
                dims=['time_of_arrival'],
                shape=[len(self._edges) - 1],
                unit='counts',
            ),
            coords={'time_of_arrival': self._edges},
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
            delta.coords['time_of_arrival'] = self._edges

        if self._cumulative is None:
            self._cumulative = delta.copy()
        else:
            self._cumulative += delta
        return delta

    def clear(self) -> None:
        """Clear both chunks and cumulative data, preserving configuration."""
        self._chunks.clear()
        self._cumulative = None
