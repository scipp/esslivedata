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
        self._empty_roi_spectra = sc.DataArray(
            sc.zeros(
                dims=['roi', 'time_of_arrival'],
                shape=[0, len(self._toa_edges) - 1],
                unit='counts',
            ),
            coords={'time_of_arrival': self._toa_edges},
        )
        self._updated_geometries: set[str] = (
            set()
        )  # Track which geometries were updated
        self._roi_mapper = get_roi_mapper()
        self._initial_readback_sent = False
        self._current_start_time: int | None = None
        self._current_end_time: int | None = None

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

        # Track time range of detector data since last finalize
        if self._current_start_time is None:
            self._current_start_time = start_time
        self._current_end_time = end_time

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

        # Create time coords for delta outputs. The start_time and end_time coords
        # represent the time range of the current (delta) output, which is the
        # period since the last finalize. This differs from the job-level times
        # which cover the entire job duration.
        start_time_coord = sc.scalar(self._current_start_time, unit='ns')
        end_time_coord = sc.scalar(self._current_end_time, unit='ns')

        cumulative = self._view.cumulative.copy()
        # This is a hack to get the current counts. Should be updated once
        # ess.reduce.live.raw.RollingDetectorView has been modified to support this.
        current = cumulative
        if self._previous is not None:
            current = current - self._previous
        self._previous = cumulative

        current = current.assign_coords(
            time=start_time_coord, start_time=start_time_coord, end_time=end_time_coord
        )

        result = sc.DataGroup(cumulative=cumulative, current=current)
        view_result = dict(result * self._inv_weights if self._use_weights else result)

        # Build stacked ROI spectra (sorted by index for color mapping)
        sorted_indices = sorted(self._rois.keys())
        current_spectra = [self._rois[idx].get_delta() for idx in sorted_indices]
        cumulative_spectra = [
            self._rois[idx].cumulative.copy() for idx in sorted_indices
        ]

        roi_coord = sc.array(
            dims=['roi'], values=sorted_indices, unit=None, dtype='int64'
        )

        if current_spectra:
            roi_current = sc.concat(current_spectra, dim='roi')
            roi_cumulative = sc.concat(cumulative_spectra, dim='roi')
        else:
            roi_current = roi_cumulative = self._empty_roi_spectra

        roi_result = {
            'roi_spectra_current': roi_current.assign_coords(
                roi=roi_coord,
                time=start_time_coord,
                start_time=start_time_coord,
                end_time=end_time_coord,
            ),
            'roi_spectra_cumulative': roi_cumulative.assign_coords(roi=roi_coord),
        }

        # Publish ROI readbacks for each geometry type that was updated,
        # or on first finalize to provide initial empty state for request plotters.
        # NOTE: If the dashboard connects after job start, it will miss this initial
        # readback and request plotter layers won't be created until the first ROI
        # update, which will never happen (chicken-egg problem). In production we do not
        # expect frequent frontend application restarts so in practice this might not
        # be a problem. A workaround if it really happens is to stop and restart the
        # job. Solutions may include sending the readback periodcally even if there was
        # no actual change.
        for geometry in self._roi_mapper.geometries:
            should_publish = (
                geometry.readback_key in self._updated_geometries
                or not self._initial_readback_sent
            )
            if not should_publish:
                continue

            # Extract ROI models for this geometry type
            roi_models = {
                idx: roi_state.model
                for idx, roi_state in self._rois.items()
                if idx in geometry.index_range
            }

            # Convert to concatenated DataArray with roi_index coordinate.
            # Include coordinate units from detector view so the dashboard can
            # use them when publishing user-drawn ROIs.
            roi_class = geometry.roi_class
            roi_result[geometry.readback_key] = roi_class.to_concatenated_data_array(
                roi_models, coord_units=self._get_detector_coord_units()
            )

        self._initial_readback_sent = True
        self._updated_geometries.clear()

        # Ratemeter: output event counts and reset for next period
        counts_result = {
            'counts_total': sc.DataArray(
                sc.scalar(self._counts_total, unit='counts'),
                coords={
                    'time': start_time_coord,
                    'start_time': start_time_coord,
                    'end_time': end_time_coord,
                },
            ),
            'counts_in_toa_range': sc.DataArray(
                sc.scalar(self._counts_in_toa_range, unit='counts'),
                coords={
                    'time': start_time_coord,
                    'start_time': start_time_coord,
                    'end_time': end_time_coord,
                },
            ),
        }
        self._counts_total = 0
        self._counts_in_toa_range = 0

        # Reset time tracking for next period
        self._current_start_time = None
        self._current_end_time = None

        return {**view_result, **roi_result, **counts_result}

    def clear(self) -> None:
        self._view.clear_counts()
        self._previous = None
        self._current_start_time = None
        self._current_end_time = None
        for roi_state in self._rois.values():
            roi_state.clear()
        self._counts_total = 0
        self._counts_in_toa_range = 0
        self._initial_readback_sent = False

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

    def _get_detector_coord_units(self) -> dict[str, sc.Unit | None]:
        """Get coordinate units from detector view for ROI readback DataArrays.

        Maps detector coordinate units to ROI 'x' and 'y' coordinates.
        Finds 1D coordinates that vary along each dimension, regardless of name.
        """
        cumulative = self._view.cumulative
        y_dim, x_dim = cumulative.dims

        def find_unit_for_dim(dim: str) -> sc.Unit | None:
            for coord in cumulative.coords.values():
                if coord.dims == (dim,):
                    return coord.unit
            return None

        return {
            'x': find_unit_for_dim(x_dim),
            'y': find_unit_for_dim(y_dim),
        }
