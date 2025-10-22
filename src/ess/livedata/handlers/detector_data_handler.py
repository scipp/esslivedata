# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Detector data handler with heavy dependencies (ess.reduce.live.raw).

This module contains factory implementations that require heavy imports.
It should only be imported by backend services, not by frontend/specs modules.

For lightweight spec registration, use detector_view_specs.py instead.
"""

from __future__ import annotations

import pathlib
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable
from typing import Any, Literal

import scipp as sc

from ess.reduce.live import raw

from ..config import models
from ..config.instrument import Instrument
from ..core.handler import Accumulator, JobBasedPreprocessorFactoryBase
from ..core.message import StreamId, StreamKind
from .accumulators import DetectorEvents, GroupIntoPixels, ROIBasedTOAHistogram
from .detector_view_specs import DetectorViewParams, ROIHistogramParams, SpecHandle
from .workflow_factory import Workflow


class DetectorProcessorFactory(ABC):
    """
    Base class for detector processor factories.
    """

    def __init__(self, *, instrument: Instrument) -> None:
        self._instrument = instrument
        self._window_length = 1

    def make_view(self, source_name: str, params: DetectorViewParams) -> DetectorView:
        """Factory method that will be registered as a workflow creation function."""
        return DetectorView(
            params=params, detector_view=self._make_rolling_view(source_name)
        )

    def make_roi(self, source_name: str, params: ROIHistogramParams) -> ROIHistogram:
        """Factory method that will be registered as a workflow creation function."""
        return ROIHistogram(
            params=params, detector_view=self._make_rolling_view(source_name)
        )

    def attach_to_handles(
        self, *, view_handle: SpecHandle, roi_handle: SpecHandle
    ) -> None:
        """
        Attach factory methods to handles from spec registration.

        Parameters
        ----------
        view_handle:
            Handle for the view workflow spec.
        roi_handle:
            Handle for the ROI histogram workflow spec.
        """
        view_handle.attach_factory()(self.make_view)
        roi_handle.attach_factory()(self.make_roi)

    @abstractmethod
    def _make_rolling_view(self, source_name: str) -> raw.RollingDetectorView:
        """Create a RollingDetectorView for the given source name."""


class DetectorProjection(DetectorProcessorFactory):
    def __init__(
        self,
        *,
        instrument: Instrument,
        projection: Literal['xy_plane', 'cylinder_mantle_z'],
        pixel_noise: str | sc.Variable | None = None,
        resolution: dict[str, dict[str, int]],
        resolution_scale: float = 1,
    ) -> None:
        super().__init__(instrument=instrument)
        self._projection = projection
        self._pixel_noise = pixel_noise
        self._resolution = resolution
        self._res_scale = resolution_scale

    def _get_resolution(self, source_name: str) -> dict[str, int]:
        aspect = self._resolution[source_name]
        return {key: value * self._res_scale for key, value in aspect.items()}

    def _make_rolling_view(self, source_name: str) -> raw.RollingDetectorView:
        return raw.RollingDetectorView.from_nexus(
            self._instrument.nexus_file,
            detector_name=self._instrument.get_detector_group_name(source_name),
            window=self._window_length,
            projection=self._projection,
            resolution=self._get_resolution(source_name),
            pixel_noise=self._pixel_noise,
        )


class DetectorLogicalView(DetectorProcessorFactory):
    """
    Factory for logical detector views with optional transform.

    Logical views use detector_number arrays directly, optionally applying a transform
    function to reshape or filter the data.
    """

    def __init__(
        self,
        *,
        instrument: Instrument,
        transform: Callable[[sc.DataArray], sc.DataArray] | None = None,
    ) -> None:
        super().__init__(instrument=instrument)
        self._transform = transform

    def _make_rolling_view(self, source_name: str) -> raw.RollingDetectorView:
        return raw.RollingDetectorView(
            detector_number=self._instrument.get_detector_number(source_name),
            window=self._window_length,
            projection=self._transform,
        )


class ROIHistogram(Workflow):
    def __init__(
        self, params: ROIHistogramParams, detector_view: raw.RollingDetectorView
    ) -> None:
        self._accumulator = ROIBasedTOAHistogram(
            toa_edges=params.toa_edges.get_edges(),
            x_range=params.x_range,
            y_range=params.y_range,
            roi_filter=detector_view.make_roi_filter(),
        )
        self._cumulative: sc.DataArray | None = None

    def accumulate(self, data: dict[Hashable, Any]) -> None:
        if len(data) != 1:
            raise ValueError("ROIHistogram expects exactly one data item.")
        raw = next(iter(data.values()))
        self._accumulator.add(0, raw)  # Timestamp not used.

    def finalize(self) -> dict[str, sc.DataArray]:
        delta = self._accumulator.get()
        if self._cumulative is None:
            self._cumulative = delta.copy()
        else:
            self._cumulative += delta
        return {'cumulative': self._cumulative, 'current': delta}

    def clear(self) -> None:
        if self._cumulative is not None:
            self._cumulative = None
        self._accumulator.clear()


class DetectorHandlerFactory(
    JobBasedPreprocessorFactoryBase[DetectorEvents, sc.DataArray]
):
    """
    Factory for detector data handlers.

    Handlers are created based on the instrument name in the message key which should
    identify the detector name. Depending on the configured detector views a NeXus file
    with geometry information may be required to setup the view. Currently the NeXus
    files are always obtained via Pooch.
    """

    def make_preprocessor(self, key: StreamId) -> Accumulator | None:
        match key.kind:
            case StreamKind.DETECTOR_EVENTS:
                detector_number = self._instrument.get_detector_number(key.name)
                return GroupIntoPixels(detector_number=detector_number)
            case _:
                return None


class DetectorView(Workflow):
    """
    Accumulator for detector counts, based on a rolling detector view.

    Return both a current (since last update) and a cumulative view of the counts.
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
            Data to be added. It is assumed that this is ev44 data that was passed
            through :py:class:`GroupIntoPixels`.
        """
        if len(data) != 1:
            raise ValueError("DetectorViewProcessor expects exactly one data item.")
        raw = next(iter(data.values()))
        filtered = self.apply_toa_range(raw)
        self._view.add_events(filtered)

    def finalize(self) -> dict[str, sc.DataArray]:
        cumulative = self._view.cumulative.copy()
        # This is a hack to get the current counts. Should be updated once
        # ess.reduce.live.raw.RollingDetectorView has been modified to support this.
        current = cumulative
        if self._previous is not None:
            current = current - self._previous
        self._previous = cumulative
        result = sc.DataGroup(cumulative=cumulative, current=current)
        return dict(result * self._inv_weights if self._use_weights else result)

    def clear(self) -> None:
        self._view.clear_counts()
        self._previous = None


# Note: Currently no need for a geometry file for NMX since the view is purely logical.
# DetectorHandlerFactory will fall back to use the detector_number configured in the
# detector view config.
# Note: There will be multiple files per instrument, valid for different date ranges.
# Files should thus not be replaced by making use of the pooch versioning mechanism.
_registry = {
    'geometry-dream-2025-01-01.nxs': 'md5:91aceb884943c76c0c21400ee74ad9b6',
    'geometry-dream-2025-05-01.nxs': 'md5:773fc7e84d0736a0121818cbacc0697f',
    'geometry-dream-no-shape-2025-05-01.nxs': 'md5:4471e2490a3dd7f6e3ed4aa0a1e0b47d',
    'geometry-loki-2025-01-01.nxs': 'md5:8d0e103276934a20ba26bb525e53924a',
    'geometry-loki-2025-03-26.nxs': 'md5:279dc8cf7dae1fac030d724bc45a2572',
    'geometry-bifrost-2025-01-01.nxs': 'md5:ae3caa99dd56de9495b9321eea4e4fef',
    'geometry-odin-2025-09-25.nxs': 'md5:5615a6203813b4ab84a191f7478ceb3c',
}


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache('beamlime'),
        env='LIVEDATA_DATA_DIR',
        retry_if_failed=3,
        base_url='https://public.esss.dk/groups/scipp/beamlime/geometry/',
        version='0',
        registry=_registry,
    )


def _parse_filename_lut(instrument: str) -> sc.DataArray:
    """
    Returns a scipp DataArray with datetime index and filename values.
    """
    registry = [name for name in _registry if instrument in name]
    if not registry:
        raise ValueError(f'No geometry files found for instrument {instrument}')
    pattern = re.compile(r'(\d{4}-\d{2}-\d{2})')
    dates = [
        pattern.search(entry).group(1) for entry in registry if pattern.search(entry)
    ]
    datetimes = sc.datetimes(dims=['datetime'], values=[*dates, '9999-12-31'], unit='s')
    return sc.DataArray(
        sc.array(dims=['datetime'], values=registry), coords={'datetime': datetimes}
    )


def get_nexus_geometry_filename(
    instrument: str, date: sc.Variable | None = None
) -> pathlib.Path:
    """
    Get filename for NeXus file based on instrument and date.

    The file is fetched and cached with Pooch.
    """
    _pooch = _make_pooch()
    dt = (date if date is not None else sc.datetime('now')).to(unit='s')
    try:
        filename = _parse_filename_lut(instrument)['datetime', dt].value
    except IndexError:
        raise ValueError(f'No geometry file found for given date {date}') from None
    return pathlib.Path(_pooch.fetch(filename))
