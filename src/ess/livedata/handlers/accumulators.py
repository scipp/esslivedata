# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipp as sc
from streaming_data_types import logdata_f144

from ess.reduce.live.roi import ROIFilter

from ..core.handler import Accumulator, Config
from ..parameter_models import RangeModel
from .to_nxevent_data import DetectorEvents, MonitorEvents


@dataclass
class LogData:
    """
    Dataclass for log data.

    Decouples our handlers from upstream schema changes. This also simplifies handler
    testing since tests do not have to construct a full logdata_f144.LogData object.
    """

    time: int
    value: Any

    @staticmethod
    def from_f144(f144: logdata_f144.LogData) -> LogData:
        return LogData(time=f144.timestamp_unix_ns, value=f144.value)


class NullAccumulator(Accumulator[Any, None]):
    def add(self, timestamp: int, data: Any) -> None:
        pass

    def get(self) -> None:
        return None

    def clear(self) -> None:
        pass


class _CumulativeAccumulationMixin:
    """Mixin providing cumulative data accumulation functionality."""

    def __init__(self, clear_on_get: bool = False):
        self._clear_on_get = clear_on_get
        self._cumulative: sc.DataArray | None = None

    def _add_cumulative(self, data: sc.DataArray) -> None:
        """Add data to the cumulative accumulation."""
        if (
            self._cumulative is None
            or data.sizes != self._cumulative.sizes
            or not sc.identical(
                data.coords[data.dim], self._cumulative.coords[data.dim]
            )
        ):
            self._cumulative = data.copy()
        else:
            self._cumulative += data

    def _get_cumulative(self) -> sc.DataArray:
        """Get the current cumulative data."""
        if self._cumulative is None:
            raise ValueError("No data has been added")
        return self._cumulative

    def clear(self) -> None:
        """Clear the cumulative data."""
        self._cumulative = None

    def _compute_result(self, cumulative: sc.DataArray) -> sc.DataArray:
        """Compute the final result from cumulative data. Override in subclasses."""
        return cumulative

    def get(self) -> sc.DataArray:
        """Get the accumulated result, optionally clearing data if configured."""
        cumulative = self._get_cumulative()
        result = self._compute_result(cumulative)
        if self._clear_on_get:
            self.clear()
        return result


class Cumulative(_CumulativeAccumulationMixin, Accumulator[sc.DataArray, sc.DataArray]):
    def __init__(self, config: Config | None = None, clear_on_get: bool = False):
        super().__init__(clear_on_get=clear_on_get)
        self._config = config or {}

    def add(self, timestamp: int, data: sc.DataArray) -> None:
        _ = timestamp
        self._add_cumulative(data)


class GroupIntoPixels(Accumulator[DetectorEvents, sc.DataArray]):
    def __init__(self, detector_number: sc.Variable):
        self._chunks: list[DetectorEvents] = []
        self._toa_unit = 'ns'
        self._sizes = detector_number.sizes
        self._dim = 'detector_number'
        self._groups = detector_number.flatten(to=self._dim)

    def add(self, timestamp: int, data: DetectorEvents) -> None:
        # timestamp in function signature is required for compliance with Accumulator
        # interface.
        _ = timestamp
        # We could easily support other units, but ev44 is always in ns so this should
        # never happen.
        if data.unit != self._toa_unit:
            raise ValueError(f"Expected unit '{self._toa_unit}', got '{data.unit}'")
        self._chunks.append(data)

    def get(self) -> sc.DataArray:
        # Could optimize the concatenate by reusing a buffer (directly write to it in
        # self.add).
        pixel_ids = (
            np.concatenate([c.pixel_id for c in self._chunks])
            if self._chunks
            else np.array([], dtype=np.int32)
        )
        time_of_arrival = (
            np.concatenate([c.time_of_arrival for c in self._chunks])
            if self._chunks
            else np.array([], dtype=np.int32)
        )
        da = sc.DataArray(
            data=sc.array(dims=['event'], values=time_of_arrival, unit=self._toa_unit),
            coords={self._dim: sc.array(dims=['event'], values=pixel_ids, unit=None)},
        )
        self._chunks.clear()
        return da.group(self._groups).fold(dim=self._dim, sizes=self._sizes)

    def clear(self) -> None:
        self._chunks.clear()


class ROIBasedTOAHistogram(Accumulator[sc.DataArray, sc.DataArray]):
    def __init__(
        self,
        *,
        x_range: RangeModel,
        y_range: RangeModel,
        toa_edges: sc.Variable,
        roi_filter: ROIFilter,
    ):
        self._roi_filter = roi_filter
        self._chunks: list[sc.DataArray] = []
        self._nbin = -1
        self._edges = toa_edges
        self._edges_ns = toa_edges.to(unit='ns')
        self._configure_roi_filter(rx=x_range, ry=y_range)

    def _configure_roi_filter(self, rx: RangeModel, ry: RangeModel) -> None:
        # Access to protected variables should hopefully be avoided by changing the
        # config values to send indices instead of percentages, once we have per-view
        # configuration.
        y, x = self._roi_filter._indices.dims
        sizes = self._roi_filter._indices.sizes
        # Convert fraction to indices
        y_indices = (int(ry.start * (sizes[y] - 1)), int(ry.stop * (sizes[y] - 1)))
        x_indices = (int(rx.start * (sizes[x] - 1)), int(rx.stop * (sizes[x] - 1)))
        new_roi = {y: y_indices, x: x_indices}

        self._roi_filter.set_roi_from_intervals(sc.DataGroup(new_roi))

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
        da = sc.reduce(self._chunks).sum()
        self._chunks.clear()
        da.coords['time_of_arrival'] = self._edges
        return da

    def clear(self) -> None:
        self._chunks.clear()


class CollectTOA(Accumulator[MonitorEvents, np.ndarray]):
    """
    Accumulator that bins time of arrival data into a histogram.

    Monitor data handlers use this as a preprocessor before actual accumulation. For
    detector data it could be used to produce a histogram for a selected ROI.
    """

    def __init__(self):
        self._chunks: list[np.ndarray] = []

    def add(self, timestamp: int, data: MonitorEvents) -> None:
        _ = timestamp
        # We could easily support other units, but ev44 is always in ns so this should
        # never happen.
        if data.unit != 'ns':
            raise ValueError(f"Expected unit 'ns', got '{data.unit}'")
        self._chunks.append(data.time_of_arrival)

    def get(self) -> np.ndarray:
        # Using NumPy here as for these specific operations with medium-sized data it is
        # a bit faster than Scipp. Could optimize the concatenate by reusing a buffer.
        result = np.concatenate(self._chunks or [[]])
        self._chunks.clear()
        return result

    def clear(self) -> None:
        self._chunks.clear()
