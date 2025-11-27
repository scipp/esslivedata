# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Area detector view for dense image data.

This module provides view classes for area detector data (ad00 schema) that use
LogicalView for transforms and ROIFilter for region-of-interest support.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any

import scipp as sc

from ess.reduce.live import raw, roi

from ..config import models
from ..config.roi_names import get_roi_mapper
from .detector_view_specs import DetectorViewParams
from .workflow_factory import Workflow


class DenseDetectorView:
    """
    View for dense detector data (images) with transform and ROI support.

    Unlike RollingDetectorView which is designed for event-mode data, this class
    handles pre-binned image data directly. It accumulates frames and provides
    ROI filtering via LogicalView's index mapping.

    Parameters
    ----------
    logical_view:
        LogicalView instance for transforming images (e.g., folding for downsampling).
    spatial_dims:
        Names of spatial dimensions for ROI filtering. If None, all dimensions from
        the logical view are treated as spatial.
    """

    def __init__(
        self,
        logical_view: raw.LogicalView,
        spatial_dims: tuple[str, ...] | None = None,
    ) -> None:
        self._logical_view = logical_view
        self._cumulative: sc.DataArray | None = None
        # Determine spatial dims: use provided or fall back to all input dims
        indices = logical_view.input_indices()
        self._spatial_dims = spatial_dims if spatial_dims else tuple(indices.dims)

    def add_image(self, data: sc.DataArray) -> None:
        """
        Add an image frame to the accumulation.

        Parameters
        ----------
        data:
            Image data as a DataArray. May have 2D (y, x) or 3D (y, x, toa) shape.
        """
        transformed = self._logical_view(data)
        if self._cumulative is None:
            self._cumulative = transformed.copy()
        else:
            self._cumulative += transformed

    @property
    def cumulative(self) -> sc.DataArray | None:
        """Get the cumulative image, or None if no data added yet."""
        return self._cumulative

    @property
    def spatial_dims(self) -> tuple[str, ...]:
        """Get the spatial dimensions used for ROI filtering."""
        return self._spatial_dims

    @property
    def has_spectral_dim(self) -> bool:
        """Check if data has non-spatial (spectral) dimensions."""
        indices = self._logical_view.input_indices()
        return len(indices.dims) > len(self._spatial_dims)

    def make_roi_filter(self) -> roi.ROIFilter | None:
        """
        Create an ROI filter using the logical view's index mapping.

        Returns None for 3D+ data where ROIFilter doesn't work properly.
        In those cases, direct slicing should be used instead.

        Returns
        -------
        :
            ROIFilter configured with the logical view's input indices,
            or None if data has non-spatial dimensions.
        """
        if self.has_spectral_dim:
            return None
        indices = self._logical_view.input_indices()
        return roi.ROIFilter(indices, spatial_dims=self._spatial_dims)

    def transform_weights(
        self,
        weights: sc.Variable | sc.DataArray | None = None,
        *,
        threshold: float = 0.1,
    ) -> sc.DataArray:
        """
        Transform raw pixel weights to the view's coordinate system.

        Parameters
        ----------
        weights:
            Raw pixel weights. If None, default weights of 1 are used.
        threshold:
            Threshold for identifying bins with low weight. Bins with weight below
            threshold * median are marked invalid (NaN).

        Returns
        -------
        :
            Transformed weights as a DataArray.
        """
        if weights is None:
            # Default: count how many input pixels contribute to each output pixel
            # Use input_indices to determine this
            indices = self._logical_view.input_indices()
            if indices.bins is not None:
                # Binned case: count events per bin
                sizes = indices.bins.size()
                weights = sc.DataArray(
                    sc.array(
                        dims=sizes.dims,
                        values=sizes.values.astype('float64'),
                        unit='one',
                    )
                )
            else:
                # Dense case: each output pixel corresponds to one input pixel
                weights = sc.DataArray(
                    sc.ones(dims=indices.dims, shape=indices.shape, unit='one')
                )
        else:
            weights = self._logical_view(weights)

        median = sc.median(weights.data)
        result = weights.copy()
        # Use dimensionless threshold scalar with matching unit
        threshold_val = sc.scalar(threshold, unit=median.unit)
        mask = result.data < threshold_val * median
        nan_val = sc.scalar(float('nan'), unit=result.data.unit)
        result.data = sc.where(mask, nan_val, result.data)
        return result

    def clear_counts(self) -> None:
        """Clear accumulated data."""
        self._cumulative = None

    @staticmethod
    def from_transform(
        transform: Callable[[sc.DataArray], sc.DataArray],
        input_sizes: dict[str, int],
        reduction_dim: str | list[str] | None = None,
        spatial_dims: tuple[str, ...] | None = None,
    ) -> DenseDetectorView:
        """
        Create a DenseDetectorView from a transform function.

        Parameters
        ----------
        transform:
            Callable that transforms input data (e.g., fold or slice operations).
        input_sizes:
            Dictionary defining the input dimension sizes.
        reduction_dim:
            Dimension(s) to sum over after applying transform.
        spatial_dims:
            Names of spatial dimensions for ROI filtering. If None, all input
            dimensions are treated as spatial.

        Returns
        -------
        :
            DenseDetectorView instance.
        """
        logical_view = raw.LogicalView(
            transform=transform,
            reduction_dim=reduction_dim,
            input_sizes=input_sizes,
        )
        return DenseDetectorView(logical_view, spatial_dims=spatial_dims)


class AreaDetectorROIHistogram:
    """
    ROI histogram for area detector data.

    For dense data, the ROI filtering sums over spatial dimensions. If the data
    has a time-of-arrival dimension, this produces a TOA histogram for the region.

    For 2D data, uses ROIFilter for efficient index-based filtering.
    For 3D data with spectral dimension, uses direct slicing to preserve the spectrum.
    """

    def __init__(
        self,
        *,
        roi_filter: roi.ROIFilter | None,
        model: models.ROI,
        spatial_dims: tuple[str, ...],
    ):
        self._roi_filter = roi_filter
        self._model = model
        self._spatial_dims = spatial_dims
        self._cumulative: sc.DataArray | None = None
        self._bounds: dict[str, tuple[int, int]] | None = None
        self._configure_filter(model)

    @property
    def model(self) -> models.ROI:
        return self._model

    @property
    def cumulative(self) -> sc.DataArray | None:
        return self._cumulative

    def _configure_filter(self, roi_model: models.ROI) -> None:
        if isinstance(roi_model, models.RectangleROI):
            if len(self._spatial_dims) < 2:
                raise ValueError(
                    f"Rectangle ROI requires at least 2 spatial dims, "
                    f"got {self._spatial_dims}"
                )
            y, x = self._spatial_dims[0], self._spatial_dims[1]
            self._bounds = roi_model.get_bounds(x_dim=x, y_dim=y)
            # Only use ROIFilter for 2D data (when all dims are spatial)
            if self._roi_filter is not None:
                self._roi_filter.set_roi_from_intervals(sc.DataGroup(self._bounds))
        else:
            roi_type = type(roi_model).__name__
            raise ValueError(
                f"Only rectangle ROI is currently supported, got {roi_type}"
            )

    def add_data(self, data: sc.DataArray) -> sc.DataArray:
        """
        Add image data and return the ROI-filtered result.

        Parameters
        ----------
        data:
            Image data (transformed by the view).

        Returns
        -------
        :
            Data summed over spatial dimensions within the ROI.
        """
        if self._roi_filter is not None:
            # 2D case: use index-based filtering
            filtered, scale = self._roi_filter.apply(data)
            result = (filtered * scale).sum('detector_number')
        else:
            # 3D+ case: use direct slicing to preserve non-spatial dims
            sliced = data
            for dim, (lo, hi) in self._bounds.items():
                sliced = sliced[dim, int(lo) : int(hi)]
            result = sliced.sum(self._spatial_dims)
        return result

    def get_delta(self, data: sc.DataArray) -> sc.DataArray:
        """
        Process data and return the delta, updating cumulative.

        Parameters
        ----------
        data:
            Image data for this period (transformed by the view).

        Returns
        -------
        :
            ROI-filtered data for this period.
        """
        delta = self.add_data(data)
        if self._cumulative is None:
            self._cumulative = delta.copy()
        else:
            self._cumulative += delta
        return delta

    def clear(self) -> None:
        self._cumulative = None


class AreaDetectorView(Workflow):
    """
    Workflow for area detector image visualization with ROI support.

    Similar to DetectorView but designed for dense image data (ad00 schema)
    rather than event-mode data (ev44 schema).
    """

    def __init__(
        self,
        params: DetectorViewParams,
        dense_view: DenseDetectorView,
    ) -> None:
        self._use_weights = params.pixel_weighting.enabled
        weighting = params.pixel_weighting
        if weighting.method != models.WeightingMethod.PIXEL_NUMBER:
            raise ValueError(f'Unsupported pixel weighting method: {weighting.method}')

        self._view = dense_view
        self._inv_weights = sc.reciprocal(dense_view.transform_weights())
        self._previous: sc.DataArray | None = None

        self._rois: dict[int, AreaDetectorROIHistogram] = {}
        self._rois_updated = False
        self._roi_mapper = get_roi_mapper()
        self._current_start_time: int | None = None

    def accumulate(
        self, data: dict[Hashable, Any], *, start_time: int, end_time: int
    ) -> None:
        """
        Add data to the accumulator.

        Parameters
        ----------
        data:
            Data to be added. Expected to contain area detector image data and
            optionally ROI configuration. Image data is a sc.DataArray from the
            Cumulative preprocessor.
        start_time:
            Start time of the data window in nanoseconds since epoch.
        end_time:
            End time of the data window in nanoseconds since epoch.
        """
        # Check for ROI configuration update
        roi_key = 'roi'
        if roi_key in data:
            roi_data_array = data[roi_key]
            rois = models.ROI.from_concatenated_data_array(roi_data_array)
            self._update_rois(rois)

        # Process image data
        image_data = {k: v for k, v in data.items() if k != roi_key}
        if len(image_data) == 0:
            return
        if len(image_data) != 1:
            raise ValueError("AreaDetectorView expects exactly one detector data item.")

        if self._current_start_time is None:
            self._current_start_time = start_time

        image = next(iter(image_data.values()))
        self._view.add_image(image)

    def finalize(self) -> dict[str, sc.DataArray]:
        if self._current_start_time is None:
            raise RuntimeError(
                "finalize called without any detector data accumulated via accumulate"
            )

        cumulative = self._view.cumulative.copy()
        current = cumulative
        if self._previous is not None:
            current = current - self._previous
        self._previous = cumulative

        time_coord = sc.scalar(self._current_start_time, unit='ns')
        current = current.assign_coords(time=time_coord)
        self._current_start_time = None

        result = sc.DataGroup(cumulative=cumulative, current=current)
        view_result = dict(result * self._inv_weights if self._use_weights else result)

        roi_result = {}
        for idx, roi_state in self._rois.items():
            # For area detectors, compute ROI from the current image delta
            roi_delta = roi_state.get_delta(current)
            roi_result[self._roi_mapper.current_key(idx)] = roi_delta.assign_coords(
                time=time_coord
            )
            roi_result[self._roi_mapper.cumulative_key(idx)] = (
                roi_state.cumulative.copy()
            )

        if self._rois_updated:
            roi_models = {idx: roi_state.model for idx, roi_state in self._rois.items()}
            concatenated_rois = models.RectangleROI.to_concatenated_data_array(
                roi_models
            )
            roi_result[self._roi_mapper.readback_keys[0]] = concatenated_rois
            self._rois_updated = False

        return {**view_result, **roi_result}

    def clear(self) -> None:
        self._view.clear_counts()
        self._previous = None
        self._current_start_time = None
        for roi_state in self._rois.values():
            roi_state.clear()

    def _update_rois(self, rois: dict[int, models.ROI]) -> None:
        current_indices = set(rois.keys())
        previous_indices = set(self._rois.keys())

        rois_changed = False
        if current_indices != previous_indices:
            rois_changed = True
        else:
            for idx, roi_model in rois.items():
                if idx in self._rois and self._rois[idx].model != roi_model:
                    rois_changed = True
                    break

        if rois_changed:
            self._rois_updated = True
            self._rois.clear()
            spatial_dims = self._view.spatial_dims
            for idx, roi_model in rois.items():
                self._rois[idx] = AreaDetectorROIHistogram(
                    roi_filter=self._view.make_roi_filter(),
                    model=roi_model,
                    spatial_dims=spatial_dims,
                )


class AreaDetectorLogicalView:
    """
    Factory for area detector views with optional transform and reduction.

    Creates AreaDetectorView workflows that use LogicalView for data transformation.
    This is the area detector equivalent of DetectorLogicalView.

    Parameters
    ----------
    input_sizes:
        Dictionary defining the input dimension sizes
        (e.g., {'dim_0': 512, 'dim_1': 512}).
    transform:
        Callable that transforms input data (e.g., fold or slice operations).
        If None, identity transform is used.
    reduction_dim:
        Dimension(s) to sum over after applying transform. Enables downsampling
        with proper ROI index mapping.
    spatial_dims:
        Names of spatial dimensions for ROI filtering. If None, all input
        dimensions are treated as spatial. For 3D data (y, x, toa), specify
        ('y', 'x') or equivalent to preserve the spectral dimension during ROI.
    """

    def __init__(
        self,
        *,
        input_sizes: dict[str, int],
        transform: Callable[[sc.DataArray], sc.DataArray] | None = None,
        reduction_dim: str | list[str] | None = None,
        spatial_dims: tuple[str, ...] | None = None,
    ) -> None:
        self._input_sizes = input_sizes
        self._transform = transform if transform is not None else _identity
        self._reduction_dim = reduction_dim
        self._spatial_dims = spatial_dims

    def make_view(
        self, source_name: str, params: DetectorViewParams
    ) -> AreaDetectorView:
        """
        Factory method that creates an area detector view for the given source.

        Parameters
        ----------
        source_name:
            Name of the detector source (used for identification).
        params:
            View configuration parameters.

        Returns
        -------
        :
            AreaDetectorView workflow instance.
        """
        _ = source_name  # Not used currently, but kept for API consistency
        logical_view = raw.LogicalView(
            transform=self._transform,
            reduction_dim=self._reduction_dim,
            input_sizes=self._input_sizes,
        )
        dense_view = DenseDetectorView(logical_view, spatial_dims=self._spatial_dims)
        return AreaDetectorView(params=params, dense_view=dense_view)


def _identity(da: sc.DataArray) -> sc.DataArray:
    return da
