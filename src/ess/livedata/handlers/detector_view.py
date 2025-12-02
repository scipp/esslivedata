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

import scipp as sc

from ess.reduce.live import raw

from ..config import models
from ..config.roi_names import get_roi_mapper
from .detector_view_specs import DetectorViewParams
from .roi_histogram import ROIHistogram
from .workflow_factory import Workflow


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
        self._rois_updated = False  # Track ROI updates at workflow level
        self._roi_mapper = get_roi_mapper()
        self._current_start_time: int | None = None

        # Ratemeter: track event counts per finalize period
        self._counts_total: int = 0
        self._counts_in_toa_range: int = 0

    def apply_toa_range(self, data: sc.DataArray) -> sc.DataArray:
        if not self._use_toa_range:
            return data
        low, high = self._toa_range
        # GroupIntoPixels stores time-of-arrival as the data variable of the bins to
        # avoid allocating weights that are all ones. For filtering we need to turn this
        # into a coordinate, since scipp does not support filtering on data variables.
        return data.bins.assign_coords(toa=data.bins.data).bins['toa', low:high]

    def accumulate(
        self, data: dict[Hashable, Any], *, start_time: int, end_time: int
    ) -> None:
        """
        Add data to the accumulator.

        Parameters
        ----------
        data:
            Data to be added. Expected to contain detector event data and optionally
            ROI configuration. Detector data is assumed to be ev44 data that was
            passed through :py:class:`GroupIntoPixels`.
        start_time:
            Start time of the data window in nanoseconds since epoch.
        end_time:
            End time of the data window in nanoseconds since epoch.
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

        # Track start time of first detector data since last finalize
        if self._current_start_time is None:
            self._current_start_time = start_time

        raw = next(iter(detector_data.values()))
        filtered = self.apply_toa_range(raw)
        self._view.add_events(filtered)
        for roi_state in self._rois.values():
            roi_state.add_data(raw)

        # Ratemeter: accumulate event counts
        self._counts_total += raw.bins.size().sum().value
        self._counts_in_toa_range += filtered.bins.size().sum().value

    def finalize(self) -> dict[str, sc.DataArray]:
        if self._current_start_time is None:
            raise RuntimeError(
                "finalize called without any detector data accumulated via accumulate"
            )

        cumulative = self._view.cumulative.copy()
        # This is a hack to get the current counts. Should be updated once
        # ess.reduce.live.raw.RollingDetectorView has been modified to support this.
        current = cumulative
        if self._previous is not None:
            current = current - self._previous
        self._previous = cumulative

        # Add time coord to current result
        time_coord = sc.scalar(self._current_start_time, unit='ns')
        current = current.assign_coords(time=time_coord)
        self._current_start_time = None

        result = sc.DataGroup(cumulative=cumulative, current=current)
        view_result = dict(result * self._inv_weights if self._use_weights else result)

        roi_result = {}
        for idx, roi_state in self._rois.items():
            roi_delta = roi_state.get_delta()

            # Add time coord to ROI current result
            roi_result[self._roi_mapper.current_key(idx)] = roi_delta.assign_coords(
                time=time_coord
            )
            roi_result[self._roi_mapper.cumulative_key(idx)] = (
                roi_state.cumulative.copy()
            )

        # Publish all ROIs as single concatenated message for readback, but only if
        # the ROI collection was updated. This mirrors the frontend's publishing
        # behavior and enables proper deletion detection.
        if self._rois_updated:
            # Extract ROI models from all active ROI states
            roi_models = {idx: roi_state.model for idx, roi_state in self._rois.items()}
            # Convert to concatenated DataArray with roi_index coordinate
            concatenated_rois = models.RectangleROI.to_concatenated_data_array(
                roi_models
            )
            # Use readback key from mapper
            roi_result[self._roi_mapper.readback_keys[0]] = concatenated_rois

            # Clear updated flag after publishing
            self._rois_updated = False

        # Ratemeter: output event counts and reset for next period
        counts_result = {
            'counts_total': sc.scalar(self._counts_total, unit='counts'),
            'counts_in_toa_range': sc.scalar(self._counts_in_toa_range, unit='counts'),
        }
        self._counts_total = 0
        self._counts_in_toa_range = 0

        return {**view_result, **roi_result, **counts_result}

    def clear(self) -> None:
        self._view.clear_counts()
        self._previous = None
        self._current_start_time = None
        for roi_state in self._rois.values():
            roi_state.clear()
        self._counts_total = 0
        self._counts_in_toa_range = 0

    def _update_rois(self, rois: dict[int, models.ROI]) -> None:
        """
        Update ROI configuration from incoming ROI models.

        When any ROI changes (addition, deletion, or modification), all ROIs are
        cleared and recreated. This ensures consistent accumulation periods across
        all ROIs, which is critical since ROI spectra are overlaid on the same plot.
        """
        # Check if the ROI set has changed
        current_indices = set(rois.keys())
        previous_indices = set(self._rois.keys())

        # Detect any change in the ROI collection (addition, deletion, or modification)
        rois_changed = False
        if current_indices != previous_indices:
            # Indices changed (addition or deletion)
            rois_changed = True
        else:
            # Check if any existing ROI model has changed
            for idx, roi_model in rois.items():
                if idx in self._rois and self._rois[idx].model != roi_model:
                    rois_changed = True
                    break

        if rois_changed:
            self._rois_updated = True
            # Clear all ROIs to ensure consistent accumulation periods
            self._rois.clear()
            # Recreate all ROIs from scratch
            for idx, roi_model in rois.items():
                self._rois[idx] = ROIHistogram(
                    toa_edges=self._toa_edges,
                    roi_filter=self._view.make_roi_filter(),
                    model=roi_model,
                )
        # else: No changes detected, preserve all existing ROI states
