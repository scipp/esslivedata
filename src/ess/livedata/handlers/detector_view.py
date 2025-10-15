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

        # ROI histogram accumulators (support multiple ROIs)
        self._roi_accumulators: dict[int, ROIBasedTOAHistogram] = {}
        self._roi_cumulatives: dict[int, sc.DataArray] = {}
        self._roi_models: dict[int, models.ROI] = {}
        self._roi_updated: set[int] = set()  # Track which ROIs were updated
        self._toa_edges = params.toa_edges.get_edges()  # Store for later use

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
            # Convert DataArray to dict of ROI models
            # Support both formats:
            # 1. New format: concatenated with roi_index coordinate (multi-ROI)
            # 2. Old format: single ROI without roi_index (backward compat)
            if 'roi_index' in roi_data_array.coords:
                # New multi-ROI format
                rois = models.RectangleROI.from_concatenated_data_array(roi_data_array)
            else:
                # Old single-ROI format - convert to dict with index 0
                single_roi = models.ROI.from_data_array(roi_data_array)
                rois = {0: single_roi}

            # Determine which ROIs were added/updated/deleted
            current_indices = set(rois.keys())
            previous_indices = set(self._roi_models.keys())

            # Remove deleted ROIs
            for idx in previous_indices - current_indices:
                del self._roi_accumulators[idx]
                del self._roi_models[idx]
                self._roi_cumulatives.pop(idx, None)

            # Add/update ROIs
            for idx, roi in rois.items():
                if idx not in self._roi_accumulators:
                    # New ROI: create accumulator
                    self._roi_accumulators[idx] = ROIBasedTOAHistogram(
                        toa_edges=self._toa_edges,
                        roi_filter=self._view.make_roi_filter(),
                    )
                # Update ROI configuration
                self._roi_models[idx] = roi
                self._roi_accumulators[idx].configure_from_roi_model(roi)
                self._roi_updated.add(idx)
                # Reset cumulative for updated ROI
                self._roi_cumulatives.pop(idx, None)

            # If only ROI config was sent (no detector data), return early
            if len(data) == 1:
                return

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
        # Also accumulate for all configured ROI histograms
        for accumulator in self._roi_accumulators.values():
            accumulator.add(0, raw)  # Timestamp not used.

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

        # Add ROI histogram results for all configured ROIs
        roi_result = {}
        for idx, accumulator in self._roi_accumulators.items():
            roi_delta = accumulator.get()

            # Update cumulative
            if idx not in self._roi_cumulatives:
                self._roi_cumulatives[idx] = roi_delta.copy()
            else:
                self._roi_cumulatives[idx] += roi_delta

            # Publish current and cumulative with index suffix
            roi_result[f'roi_current_{idx}'] = roi_delta
            roi_result[f'roi_cumulative_{idx}'] = self._roi_cumulatives[idx]

            # Echo back ROI configuration when updated
            if idx in self._roi_updated:
                roi_data = self._roi_models[idx].to_data_array()
                # Include index in stream name (e.g., 'roi_rectangle_0')
                roi_stream_name = f'roi_{roi_data.name}_{idx}'
                roi_result[roi_stream_name] = roi_data

        # Clear updated flags
        self._roi_updated.clear()

        # Merge detector view and ROI results
        return {**view_result, **roi_result}

    def clear(self) -> None:
        self._view.clear_counts()
        self._previous = None
        # Clear all ROI cumulatives and accumulators
        self._roi_cumulatives.clear()
        for accumulator in self._roi_accumulators.values():
            accumulator.clear()
