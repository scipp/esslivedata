# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import pathlib
import re

import scipp as sc

from ..config.instrument import Instrument
from ..core.handler import Accumulator, JobBasedPreprocessorFactoryBase
from ..core.message import StreamId, StreamKind
from .accumulators import Cumulative, LatestValueHandler
from .group_by_pixel import GroupByPixel
from .to_nxevent_data import ToNXevent_data
from .to_nxlog import ToNXlog

_GEOMETRY_RELEASE_URL = (
    'https://github.com/scipp/esslivedata/releases/download/geometry-v0/'
)


class DetectorHandlerFactory(JobBasedPreprocessorFactoryBase):
    """
    Factory for detector data handlers.

    Handlers are created based on the instrument name in the message key which should
    identify the detector name. Depending on the configured detector views a NeXus file
    with geometry information may be required to setup the view.

    Parameters
    ----------
    instrument:
        The instrument configuration.
    """

    def __init__(self, *, instrument: Instrument, group_by_pixel: bool = True) -> None:
        super().__init__(instrument=instrument)
        self._group_by_pixel = group_by_pixel

    def make_preprocessor(self, key: StreamId) -> Accumulator | None:
        match key.kind:
            case StreamKind.DETECTOR_EVENTS:
                # Skip detectors that are not configured
                if key.name not in self._instrument.detector_names:
                    return None
                if self._group_by_pixel:
                    detector_number = self._instrument.get_detector_number(key.name)
                    return GroupByPixel(ToNXevent_data(), detector_number)
                return ToNXevent_data()
            case StreamKind.AREA_DETECTOR:
                return Cumulative(clear_on_get=True)
            case StreamKind.LIVEDATA_ROI:
                return LatestValueHandler()
            case StreamKind.LOG:
                attrs = self._instrument.f144_attribute_registry.get(key.name)
                if attrs is None:
                    return None
                return ToNXlog(attrs=attrs)
            case _:
                return None


# Note: Currently no need for a geometry file for NMX since the view is purely logical.
# DetectorHandlerFactory will fall back to use the detector_number configured in the
# detector view config.
# Note: There will be multiple files per instrument, valid for different date ranges.
# Files should thus not be replaced by making use of the pooch versioning mechanism.
_registry = {
    'geometry-dream-2025-01-01.nxs': 'md5:91aceb884943c76c0c21400ee74ad9b6',
    'geometry-dream-no-shape-2025-05-01.nxs': 'md5:4471e2490a3dd7f6e3ed4aa0a1e0b47d',
    'geometry-loki-2025-01-01.nxs': 'md5:8d0e103276934a20ba26bb525e53924a',
    'geometry-loki-2025-03-26.nxs': 'md5:279dc8cf7dae1fac030d724bc45a2572',
    'geometry-loki-2026-02-11.nxs': 'md5:0b40ba0ec640f1c497ec02b233f42ec6',
    'geometry-loki-2026-02-23.nxs': 'md5:5110801aa7c7d79a32cb73b9fb71f167',
    'geometry-loki-2026-03-11.nxs': 'md5:fc0dafdea9a3f66b3003a5d5b7e66698',
    'geometry-loki-2026-04-13.nxs': 'md5:5c4a6883cf34cee9836f543f4d70ae5c',
    'geometry-bifrost-2025-01-01.nxs': 'md5:ae3caa99dd56de9495b9321eea4e4fef',
    'geometry-odin-2025-09-25.nxs': 'md5:5615a6203813b4ab84a191f7478ceb3c',
    'geometry-tbl-2025-12-03.nxs': 'md5:040a70659155eb386245755455ee3e62',
    'geometry-estia-2025-12-16.nxs': 'md5:07d33010189a50ee46ee5f649f848ca5',
}


def _get_data_dir() -> pathlib.Path | None:
    """Return the geometry data directory if LIVEDATA_DATA_DIR is set."""
    import os

    data_dir = os.environ.get('LIVEDATA_DATA_DIR')
    if data_dir is not None:
        return pathlib.Path(data_dir)
    return None


def _fetch_with_pooch(filename: str) -> pathlib.Path:
    """Fetch a geometry file using pooch, downloading if necessary."""
    import pooch

    p = pooch.create(
        path=pooch.os_cache('beamlime'),
        retry_if_failed=3,
        base_url=_GEOMETRY_RELEASE_URL,
        version='0',
        registry=_registry,
    )
    return pathlib.Path(p.fetch(filename))


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
    Get the path to a NeXus geometry file based on instrument and date.

    If LIVEDATA_DATA_DIR is set, the file is read directly from that directory.
    Otherwise, the file is fetched (and cached) using pooch.
    """
    dt = (date if date is not None else sc.datetime('now')).to(unit='s')
    try:
        filename = _parse_filename_lut(instrument)['datetime', dt].value
    except IndexError:
        raise ValueError(f'No geometry file found for given date {date}') from None
    data_dir = _get_data_dir()
    if data_dir is not None:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Geometry file '{filename}' not found in LIVEDATA_DATA_DIR={data_dir}"
            )
        return path
    return _fetch_with_pooch(filename)
