# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
DetectorView workflow and ROI-based histogram accumulation.

This module provides the DetectorView workflow which combines detector counts
visualization with ROI-based time-of-arrival histogram accumulation.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import pydantic
import scipp as sc

from ess.reduce.live import raw
from ess.reduce.live.roi import ROIFilter

from .. import parameter_models
from ..config import models
from ..core.handler import Accumulator
from .workflow_factory import Workflow


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


class ROIBasedTOAHistogram(Accumulator[sc.DataArray, sc.DataArray]):
    def __init__(
        self,
        *,
        toa_edges: sc.Variable,
        roi_filter: ROIFilter,
    ):
        self._roi_filter = roi_filter
        self._chunks: list[sc.DataArray] = []
        self._nbin = -1
        self._edges = toa_edges
        self._edges_ns = toa_edges.to(unit='ns')

    def configure_from_roi_model(
        self, roi: models.RectangleROI | models.PolygonROI | models.EllipseROI
    ) -> None:
        """
        Configure the ROI filter from an ROI model (RectangleROI, PolygonROI, etc.).

        Parameters
        ----------
        roi:
            An ROI model from config.models (RectangleROI, PolygonROI, or EllipseROI).
        """
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
        constituents = data.bins.constituents
        content = constituents['data']
        content.coords['time_of_arrival'] = content.data
        content.data = sc.ones(
            dims=content.dims, shape=content.shape, dtype='float32', unit='counts'
        )
        data.data = sc.bins(**constituents, validate_indices=False)

    def add(self, timestamp: int, data: sc.DataArray) -> None:
        # Note that the preprocessor does *not* add weights of 1 (unlike NeXus loaders).
        # Instead, the data column of the content corresponds to the time of arrival.
        filtered, scale = self._roi_filter.apply(data)
        self._add_weights(filtered)
        filtered *= scale
        chunk = filtered.hist(time_of_arrival=self._edges_ns, dim=filtered.dim)
        self._chunks.append(chunk)

    def get(self) -> sc.DataArray:
        if not self._chunks:
            # Return empty histogram if no data
            da = sc.DataArray(
                data=sc.zeros(
                    dims=['time_of_arrival'],
                    shape=[len(self._edges) - 1],
                    unit='counts',
                ),
                coords={'time_of_arrival': self._edges},
            )
        else:
            da = sc.reduce(self._chunks).sum()
            self._chunks.clear()
            da.coords['time_of_arrival'] = self._edges
        return da

    def clear(self) -> None:
        self._chunks.clear()


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
        self._updated = False
        # Configure ROI filter from initial model without setting updated flag
        self._configure_filter(model)

    @property
    def model(self) -> models.ROI:
        """Get the ROI configuration model."""
        return self._model

    @property
    def cumulative(self) -> sc.DataArray | None:
        """Get the cumulative histogram, or None if not yet accumulated."""
        return self._cumulative

    @property
    def updated(self) -> bool:
        """Check if ROI configuration was updated."""
        return self._updated

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

    def configure_from_roi_model(
        self, roi: models.RectangleROI | models.PolygonROI | models.EllipseROI
    ) -> None:
        """
        Update ROI filter configuration from an ROI model.

        Parameters
        ----------
        roi:
            An ROI model from config.models (RectangleROI, PolygonROI, or EllipseROI).
        """
        self._configure_filter(roi)
        self._model = roi
        self._updated = True
        self._cumulative = None  # Reset cumulative on config change

    def _add_weights(self, data: sc.DataArray) -> None:
        """Add weight coordinate to binned data."""
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

    def clear_updated_flag(self) -> None:
        """Clear the updated flag."""
        self._updated = False


class DetectorView(Workflow):
    """
    Unified workflow for detector counts and ROI histogram.

    Return both a current (since last update) and a cumulative view of the counts,
    and optionally ROI histogram data.
    """

    def __init__(
        self,
        params: DetectorViewParams,
        detector_view: raw.RollingDetectorView,
    ) -> None:
        self._use_toa_range = params.toa_range.enabled
        self._toa_range = params.toa_range.range_ns
        self._use_weights = params.pixel_weighting.enabled
        # Note: Currently we use default weighting based on the number of detector
        # pixels contributing to each screen pixel. In the future more advanced options
        # such as by the signal of a uniform scattered may need to be supported.
        weighting = params.pixel_weighting
        if weighting.method != models.WeightingMethod.PIXEL_NUMBER:
            raise ValueError(f'Unsupported pixel weighting method: {weighting.method}')
        self._use_weights = weighting.enabled
        self._view = detector_view
        self._inv_weights = sc.reciprocal(detector_view.transform_weights())
        self._previous: sc.DataArray | None = None

        self._rois: dict[int, ROIHistogram] = {}
        self._toa_edges = params.toa_edges.get_edges()

    def apply_toa_range(self, data: sc.DataArray) -> sc.DataArray:
        if not self._use_toa_range:
            return data
        low, high = self._toa_range
        # GroupIntoPixels stores time-of-arrival as the data variable of the bins to
        # avoid allocating weights that are all ones. For filtering we need to turn this
        # into a coordinate, since scipp does not support filtering on data variables.
        return data.bins.assign_coords(toa=data.bins.data).bins['toa', low:high]

    def accumulate(self, data: dict[Hashable, Any]) -> None:
        """
        Add data to the accumulator.

        Parameters
        ----------
        data:
            Data to be added. Expected to contain detector event data and optionally
            ROI configuration. Detector data is assumed to be ev44 data that was
            passed through :py:class:`GroupIntoPixels`.
        """
        # Check for ROI configuration update (auxiliary data)
        # Stream name is 'roi' (from 'roi_rectangle' after job_id prefix stripped)
        roi_key = 'roi'
        if roi_key in data:
            roi_data_array = data[roi_key]
            rois = models.ROI.from_concatenated_data_array(roi_data_array)
            self._update_rois(rois)

        # Process detector event data
        detector_data = {k: v for k, v in data.items() if k != roi_key}
        if len(detector_data) == 0:
            # No detector data to process (e.g., empty dict or only rois)
            return
        if len(detector_data) != 1:
            raise ValueError(
                "DetectorViewProcessor expects exactly one detector data item."
            )
        raw = next(iter(detector_data.values()))
        filtered = self.apply_toa_range(raw)
        self._view.add_events(filtered)
        for roi_state in self._rois.values():
            roi_state.add_data(raw)

    def finalize(self) -> dict[str, sc.DataArray]:
        cumulative = self._view.cumulative.copy()
        # This is a hack to get the current counts. Should be updated once
        # ess.reduce.live.raw.RollingDetectorView has been modified to support this.
        current = cumulative
        if self._previous is not None:
            current = current - self._previous
        self._previous = cumulative
        result = sc.DataGroup(cumulative=cumulative, current=current)
        view_result = dict(result * self._inv_weights if self._use_weights else result)

        roi_result = {}
        for idx, roi_state in self._rois.items():
            roi_delta = roi_state.get_delta()

            roi_result[f'roi_current_{idx}'] = roi_delta
            roi_result[f'roi_cumulative_{idx}'] = roi_state.cumulative.copy()

            if roi_state.updated:
                roi_data = roi_state.model.to_data_array()
                roi_stream_name = f'roi_{roi_data.name}_{idx}'
                roi_result[roi_stream_name] = roi_data
                roi_state.clear_updated_flag()

        return {**view_result, **roi_result}

    def clear(self) -> None:
        self._view.clear_counts()
        self._previous = None
        for roi_state in self._rois.values():
            roi_state.clear()

    def _update_rois(self, rois: dict[int, models.ROI]) -> None:
        """Update ROI configuration from incoming ROI models."""
        current_indices = set(rois.keys())
        previous_indices = set(self._rois.keys())

        # Remove deleted ROIs
        for idx in previous_indices - current_indices:
            del self._rois[idx]

        # Add/update ROIs
        for idx, roi_model in rois.items():
            if idx not in self._rois:
                # Create new ROI histogram (this sets _updated = False initially)
                self._rois[idx] = ROIHistogram(
                    toa_edges=self._toa_edges,
                    roi_filter=self._view.make_roi_filter(),
                    model=roi_model,
                )
                # Mark as updated so it gets published on first finalize
                self._rois[idx]._updated = True
            else:
                # Update existing ROI configuration (this sets _updated = True)
                self._rois[idx].configure_from_roi_model(roi_model)
