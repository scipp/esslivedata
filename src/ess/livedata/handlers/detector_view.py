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
from scipp.core import label_based_index_to_positional_index

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

    def _convert_roi_bounds_to_indices(
        self,
        dim_name: str,
        min_val: float,
        max_val: float,
        unit: str | None,
    ) -> tuple[int, int]:
        """
        Convert ROI bounds to pixel indices for a single dimension.

        Parameters
        ----------
        dim_name:
            Name of the dimension (e.g., 'x' or 'y').
        min_val:
            Minimum coordinate value.
        max_val:
            Maximum coordinate value.
        unit:
            Unit string for physical coordinates, or None for pixel indices.

        Returns
        -------
        :
            Tuple of (start_index, stop_index).

        Raises
        ------
        RuntimeError
            If unit is None but coordinates exist, or if unit is provided but
            coordinates are missing.
        """
        indices = self._roi_filter._indices
        has_coord = dim_name in indices.coords

        if unit is None:
            # Direct pixel indices - no coordinate lookup needed
            if has_coord:
                raise RuntimeError(
                    f"ROI has {dim_name}_unit=None but dimension '{dim_name}' "
                    "has coordinates. This indicates an implementation error "
                    "in ROI configuration."
                )
            return (int(min_val), int(max_val))
        else:
            # Physical coordinates - need label-based lookup
            if not has_coord:
                raise RuntimeError(
                    f"ROI has {dim_name}_unit='{unit}' but dimension '{dim_name}' "
                    "has no coordinates. This indicates an implementation error "
                    "in ROI configuration."
                )
            coords = indices.coords[dim_name]
            min_scalar = sc.scalar(min_val, unit=unit)
            max_scalar = sc.scalar(max_val, unit=unit)
            _, bounds_slice = label_based_index_to_positional_index(
                indices.sizes, coords, slice(min_scalar, max_scalar)
            )
            return (bounds_slice.start, bounds_slice.stop)

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
            y_indices = self._convert_roi_bounds_to_indices(
                y, roi.y_min, roi.y_max, roi.y_unit
            )
            x_indices = self._convert_roi_bounds_to_indices(
                x, roi.x_min, roi.x_max, roi.x_unit
            )
            new_roi = {y: y_indices, x: x_indices}
            self._roi_filter.set_roi_from_intervals(sc.DataGroup(new_roi))
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

        # ROI histogram accumulator
        self._roi_accumulator = ROIBasedTOAHistogram(
            toa_edges=params.toa_edges.get_edges(),
            roi_filter=detector_view.make_roi_filter(),
        )
        self._roi_cumulative: sc.DataArray | None = None
        self._roi_model: models.ROI | None = None
        self._roi_config_updated = False

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
        roi_config_key = 'roi_config'
        if roi_config_key in data:
            roi_data_array = data[roi_config_key]
            # Convert DataArray to ROI model
            self._roi_model = models.ROI.from_data_array(roi_data_array)
            # Configure the ROI accumulator with the new model
            self._roi_accumulator.configure_from_roi_model(self._roi_model)
            self._roi_config_updated = True
            # Reset cumulative histogram when ROI changes
            # (otherwise we'd be mixing events from different ROI regions)
            self._roi_cumulative = None
            # If only ROI config was sent (no detector data), return early
            if len(data) == 1:
                return

        # Process detector event data
        detector_data = {k: v for k, v in data.items() if k != roi_config_key}
        if len(detector_data) == 0:
            # No detector data to process (e.g., empty dict or only roi_config)
            return
        if len(detector_data) != 1:
            raise ValueError(
                "DetectorViewProcessor expects exactly one detector data item."
            )
        raw = next(iter(detector_data.values()))
        filtered = self.apply_toa_range(raw)
        self._view.add_events(filtered)
        # Also accumulate for ROI histogram (only if ROI is configured)
        if self._roi_model is not None:
            self._roi_accumulator.add(0, raw)  # Timestamp not used.

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

        # Add ROI histogram results (only if ROI is configured)
        roi_result = {}
        if self._roi_model is not None:
            roi_delta = self._roi_accumulator.get()
            if self._roi_cumulative is None:
                self._roi_cumulative = roi_delta.copy()
            else:
                self._roi_cumulative += roi_delta
            roi_result = {
                'roi_cumulative': self._roi_cumulative,
                'roi_current': roi_delta,
            }

            # Publish ROI configuration when it's been updated
            if self._roi_config_updated:
                roi_result['roi_config'] = self._roi_model.to_data_array()
                self._roi_config_updated = False

        # Merge detector view and ROI results
        return {**view_result, **roi_result}

    def clear(self) -> None:
        self._view.clear_counts()
        self._previous = None
        if self._roi_cumulative is not None:
            self._roi_cumulative = None
        self._roi_accumulator.clear()
