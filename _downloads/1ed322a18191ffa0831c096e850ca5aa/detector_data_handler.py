# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import pathlib
import re
from collections.abc import Callable
from typing import Literal

import scipp as sc

from ess.reduce.live import raw

from ..config.instrument import Instrument
from ..core.handler import Accumulator, JobBasedPreprocessorFactoryBase
from ..core.message import StreamId, StreamKind
from .accumulators import (
    Cumulative,
    DetectorEvents,
    GroupIntoPixels,
    LatestValueHandler,
)
from .detector_view import DetectorView, DetectorViewParams

ProjectionType = Literal['xy_plane', 'cylinder_mantle_z']


class DetectorProjection:
    """
    Factory for detector projection views.

    Creates detector views by projecting detector pixels onto a 2D plane
    (xy_plane or cylinder_mantle_z).

    Parameters
    ----------
    instrument:
        Instrument configuration.
    projection:
        Projection type(s). Either a single projection type applied to all sources,
        or a dict mapping source names to their projection types. This allows using
        different projections for different detector banks in a single workflow.
    pixel_noise:
        Pixel noise parameter passed to RollingDetectorView.from_nexus.
    resolution:
        Dict mapping source names to resolution specs (e.g., {'x': 10, 'y': 20}).
    resolution_scale:
        Scale factor for resolution values.

    Example
    -------
    Single projection for all detectors:

    .. code-block:: python

        projection = DetectorProjection(
            instrument=instrument,
            projection='xy_plane',
            resolution={'detector_0': {'x': 10, 'y': 20}},
        )

    Mixed projections per detector:

    .. code-block:: python

        projection = DetectorProjection(
            instrument=instrument,
            projection={
                'mantle_detector': 'cylinder_mantle_z',
                'endcap_detector': 'xy_plane',
            },
            resolution={
                'mantle_detector': {'arc_length': 10, 'z': 40},
                'endcap_detector': {'y': 30, 'x': 20},
            },
        )
    """

    def __init__(
        self,
        *,
        instrument: Instrument,
        projection: ProjectionType | dict[str, ProjectionType],
        pixel_noise: str | sc.Variable | None = None,
        resolution: dict[str, dict[str, int]],
        resolution_scale: float = 1,
    ) -> None:
        self._instrument = instrument
        self._projection = projection
        self._pixel_noise = pixel_noise
        self._resolution = resolution
        self._res_scale = resolution_scale
        self._window_length = 1

    def _get_projection(self, source_name: str) -> ProjectionType:
        if isinstance(self._projection, dict):
            return self._projection[source_name]
        return self._projection

    def _get_resolution(self, source_name: str) -> dict[str, int]:
        aspect = self._resolution[source_name]
        return {key: value * self._res_scale for key, value in aspect.items()}

    def make_view(self, source_name: str, params: DetectorViewParams) -> DetectorView:
        """Factory method that creates a detector view for the given source."""
        detector_view = raw.RollingDetectorView.from_nexus(
            self._instrument.nexus_file,
            detector_name=self._instrument.get_detector_group_name(source_name),
            window=self._window_length,
            projection=self._get_projection(source_name),
            resolution=self._get_resolution(source_name),
            pixel_noise=self._pixel_noise,
        )
        return DetectorView(params=params, detector_view=detector_view)


def _identity(da: sc.DataArray, source_name: str) -> sc.DataArray:
    return da


class DetectorLogicalView:
    """
    Factory for logical detector views with optional transform and reduction.

    Logical views use detector_number arrays directly, optionally applying a transform
    function to reshape or filter the data. Uses ``LogicalView`` from
    ``ess.reduce.live`` for proper index mapping, enabling ROI support.

    Parameters
    ----------
    instrument:
        Instrument configuration.
    transform:
        Callable that transforms input data (e.g., fold, slice, or reshape operations).
        The signature is ``(da: DataArray, source_name: str) -> DataArray``, where
        ``source_name`` identifies which detector bank the data is from. This allows
        a single transform to handle multiple detector banks with different parameters.
        If reduction_dim is specified, the transform should NOT include summing - that
        is handled separately to enable proper ROI index mapping.
    reduction_dim:
        Dimension(s) to sum over after applying transform. If specified, enables proper
        ROI support by tracking which input pixels contribute to each output pixel.

    Example
    -------
    Simple view without reduction (ROI supported):

    .. code-block:: python

        view = DetectorLogicalView(instrument=instrument)

    View with downsampling and ROI support:

    .. code-block:: python

        def fold_image(da, source_name):
            da = da.fold('x', {'x': 512, 'x_bin': -1})
            da = da.fold('y', {'y': 512, 'y_bin': -1})
            return da

        view = DetectorLogicalView(
            instrument=instrument,
            transform=fold_image,
            reduction_dim=['x_bin', 'y_bin'],
        )
    """

    def __init__(
        self,
        *,
        instrument: Instrument,
        transform: Callable[[sc.DataArray, str], sc.DataArray] | None = None,
        reduction_dim: str | list[str] | None = None,
        roi_support: bool = True,
    ) -> None:
        self._instrument = instrument
        self._transform = transform if transform is not None else _identity
        self._reduction_dim = reduction_dim
        self._roi_support = roi_support
        self._window_length = 1

    def make_view(self, source_name: str, params: DetectorViewParams) -> DetectorView:
        """Factory method that creates a detector view for the given source."""

        def bound_transform(da: sc.DataArray) -> sc.DataArray:
            return self._transform(da, source_name)

        detector_view = raw.RollingDetectorView.with_logical_view(
            detector_number=self._instrument.get_detector_number(source_name),
            window=self._window_length,
            transform=bound_transform,
            reduction_dim=self._reduction_dim,
        )
        return DetectorView(
            params=params, detector_view=detector_view, roi_support=self._roi_support
        )


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
                # Skip detectors that are not configured
                if key.name not in self._instrument.detector_names:
                    return None
                detector_number = self._instrument.get_detector_number(key.name)
                return GroupIntoPixels(detector_number=detector_number)
            case StreamKind.AREA_DETECTOR:
                return Cumulative(clear_on_get=True)
            case StreamKind.LIVEDATA_ROI:
                return LatestValueHandler()
            case _:
                return None


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
    'geometry-tbl-2025-12-03.nxs': 'md5:040a70659155eb386245755455ee3e62',
    'geometry-estia-2025-12-16.nxs': 'md5:07d33010189a50ee46ee5f649f848ca5',
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
