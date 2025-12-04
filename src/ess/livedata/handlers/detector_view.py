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
from ..config.roi_names import ROIGeometry, get_roi_mapper
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
        self._updated_geometries: set[str] = (
            set()
        )  # Track which geometries were updated
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
        # Check for ROI configuration updates (auxiliary data)
        # Keys are 'roi_rectangle', 'roi_polygon', etc. from DetectorROIAuxSources
        for geometry in self._roi_mapper.geometries:
            if geometry.readback_key in data:
                roi_data_array = data[geometry.readback_key]
                rois = models.ROI.from_concatenated_data_array(roi_data_array)
                self._update_rois(rois, geometry)

        # Process detector event data (exclude ROI keys)
        roi_keys = self._roi_mapper.readback_keys
        detector_data = {k: v for k, v in data.items() if k not in roi_keys}
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

        # Publish ROI readbacks for each geometry type that was updated.
        # Each geometry type gets its own readback stream.
        for geometry in self._roi_mapper.geometries:
            if geometry.readback_key not in self._updated_geometries:
                continue

            # Extract ROI models for this geometry type
            roi_models = {
                idx: roi_state.model
                for idx, roi_state in self._rois.items()
                if idx in geometry.index_range
            }

            # Convert to concatenated DataArray with roi_index coordinate
            roi_class = geometry.roi_class
            roi_result[geometry.readback_key] = roi_class.to_concatenated_data_array(
                roi_models
            )

        # Clear updated geometries after publishing
        self._updated_geometries.clear()

        # Ratemeter: output event counts and reset for next period
        counts_result = {
            'counts_total': sc.DataArray(
                sc.scalar(self._counts_total, unit='counts'),
                coords={'time': time_coord},
            ),
            'counts_in_toa_range': sc.DataArray(
                sc.scalar(self._counts_in_toa_range, unit='counts'),
                coords={'time': time_coord},
            ),
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

    def _update_rois(self, rois: dict[int, models.ROI], geometry: ROIGeometry) -> None:
        """
        Update ROI configuration for a specific geometry type.

        Only updates ROIs belonging to the specified geometry type, leaving other
        geometry types unchanged. When any ROI of the specified type changes
        (addition, deletion, or modification), all ROIs of that type are cleared
        and recreated to ensure consistent accumulation periods.

        Parameters
        ----------
        rois:
            Dictionary mapping ROI index to ROI model for the geometry type.
        geometry:
            The ROI geometry configuration identifying which geometry type
            is being updated.
        """
        index_range = geometry.index_range

        # Get current and previous indices for THIS geometry type only
        current_indices = set(rois.keys())
        previous_indices = {idx for idx in self._rois.keys() if idx in index_range}

        # Detect any change in this geometry's ROI collection
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
            self._updated_geometries.add(geometry.readback_key)
            # Clear only ROIs of this geometry type
            for idx in list(self._rois.keys()):
                if idx in index_range:
                    del self._rois[idx]
            # Recreate ROIs for this geometry type
            for idx, roi_model in rois.items():
                self._rois[idx] = ROIHistogram(
                    toa_edges=self._toa_edges,
                    roi_filter=self._view.make_roi_filter(),
                    model=roi_model,
                )
        # else: No changes detected, preserve existing ROI states
