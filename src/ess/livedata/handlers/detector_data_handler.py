# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import pathlib
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import pydantic
import scipp as sc
from pydantic import field_validator

from ess.reduce.live import raw

from ..config.instrument import Instrument
from ..config.workflow_spec import AuxSourcesBase, JobId
from ..core.handler import Accumulator, JobBasedPreprocessorFactoryBase
from ..core.message import StreamId, StreamKind
from .accumulators import DetectorEvents, GroupIntoPixels, LatestValue
from .detector_view import DetectorView, DetectorViewParams


class DetectorROIAuxSources(AuxSourcesBase):
    """
    Auxiliary source model for ROI configuration in detector workflows.

    Allows users to select between different ROI shapes (rectangle, polygon, ellipse).
    The render() method prefixes stream names with the job number to create job-specific
    ROI configuration streams, since each job instance needs its own ROI.
    """

    roi: Literal['rectangle', 'polygon', 'ellipse'] = pydantic.Field(
        default='rectangle',
        description='Shape to use for the region of interest (ROI).',
    )

    @field_validator('roi')
    @classmethod
    def validate_roi_shape(cls, v: str) -> str:
        """Validate that only rectangle is currently supported."""
        if v != 'rectangle':
            raise ValueError(
                f"Currently only 'rectangle' ROI shape is supported, got '{v}'"
            )
        return v

    def render(self, job_id: JobId) -> dict[str, str]:
        """
        Render ROI stream name with job-specific prefix.

        Parameters
        ----------
        job_id:
            Job identifier containing source_name and job_number.

        Returns
        -------
        :
            Mapping from field name 'roi' to job-specific stream name in the
            format '{source_name}/{job_number}/roi_{shape}' (e.g.,
            'mantle/abc-123/roi_rectangle'). The source_name ensures ROI
            streams are unique per detector in multi-detector workflows where
            the same job_number is shared across detectors.
        """
        base = self.model_dump(mode='json')
        return {field: f"{job_id}/roi_{stream}" for field, stream in base.items()}


@dataclass(frozen=True, kw_only=True)
class ViewConfig:
    name: str
    title: str
    description: str
    source_names: list[str] = field(default_factory=list)


class DetectorProcessorFactory(ABC):
    def __init__(self, *, instrument: Instrument, config: ViewConfig) -> None:
        self._instrument = instrument
        self._config = config
        self._window_length = 1
        self._register_with_instrument(instrument)

    def make_view(self, source_name: str, params: DetectorViewParams) -> DetectorView:
        """Factory method that will be registered as a workflow creation function."""
        return DetectorView(
            params=params, detector_view=self._make_rolling_view(source_name)
        )

    def _register_with_instrument(self, instrument: Instrument) -> None:
        instrument.register_workflow(
            namespace='detector_data',
            name=self._config.name,
            version=1,
            title=self._config.title,
            description=self._config.description,
            source_names=self._config.source_names,
            aux_sources=DetectorROIAuxSources,
        )(self.make_view)

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
        self._projection = projection
        self._pixel_noise = pixel_noise
        self._resolution = resolution
        self._res_scale = resolution_scale
        source_names = list(resolution.keys())
        if projection == 'xy_plane':
            config = ViewConfig(
                name='detector_xy_projection',
                title='Detector XY Projection',
                description='Projection of a detector bank onto an XY-plane.',
                source_names=source_names,
            )
        elif projection == 'cylinder_mantle_z':
            config = ViewConfig(
                name='detector_cylinder_mantle_z',
                title='Detector Cylinder Mantle Z Projection',
                description='Projection of a detector bank onto a cylinder mantle '
                'along Z-axis.',
                source_names=source_names,
            )
        else:
            raise ValueError(f'Unsupported projection: {projection}')
        super().__init__(instrument=instrument, config=config)

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


@dataclass(frozen=True, kw_only=True)
class LogicalViewConfig(ViewConfig):
    # If no projection defined, the shape of the detector_number is used.
    transform: Callable[[sc.DataArray], sc.DataArray] | None = None


class DetectorLogicalView(DetectorProcessorFactory):
    def __init__(self, *, instrument: Instrument, config: LogicalViewConfig) -> None:
        super().__init__(instrument=instrument, config=config)
        self._config = config

    def _make_rolling_view(self, source_name: str) -> raw.RollingDetectorView:
        return raw.RollingDetectorView(
            detector_number=self._instrument.get_detector_number(source_name),
            window=self._window_length,
            projection=self._config.transform,
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
                detector_number = self._instrument.get_detector_number(key.name)
                return GroupIntoPixels(detector_number=detector_number)
            case StreamKind.LIVEDATA_ROI:
                return LatestValue()
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
