# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
TBL workflow spec registration.
"""

from ess.livedata.config import (
    Instrument,
    SourceMetadata,
    filter_authorized_streams,
    instrument_registry,
    name_streams,
)
from ess.livedata.config.device_contract import DETECTOR_VIEW_DEVICES
from ess.livedata.config.workflow_spec import DETECTORS
from ess.livedata.workflows.detector_view_specs import (
    DetectorViewOutputsBase,
    SpectrumViewSpec,
)
from ess.livedata.workflows.monitor_workflow_specs import (
    register_monitor_workflow_specs,
)

from .streams_parsed import PARSED_STREAMS
from .views import (
    get_he3_detector_view,
    get_he3_spectrum,
    get_multiblade_spectrum,
    get_multiblade_view,
    identity,
    name_image_dims,
)

detector_names = [
    'timepix3_detector',
    'multiblade_detector',
    'he3_detector_bank0',
    'he3_detector_bank1',
    'ngem_detector',
    # not listing orca since it does not have (and does not need) detector numbers
]

monitor_names = ['monitor_1']

#: Side length the Timepix3 panel is ingested at; see name_image_dims.
TIMEPIX3_IMAGE_RESOLUTION = 512
#: Largest grid the Timepix3 panel can read out. The readout is
#: reconfigured during operation, so this bounds the inferred streamed
#: resolution rather than stating it.
TIMEPIX3_PANEL_RESOLUTION = 4096

instrument = Instrument(
    name='tbl',
    detector_names=detector_names,
    monitors=monitor_names,
    # Bandwidth choppers feeding the wavelength-LUT cascade, in beam order
    # (source -> sample): bwc_2 sits at 14.5 m from the source, bwc_1 at 15.5 m.
    choppers=['bwc_2', 'bwc_1'],
    streams=name_streams(filter_authorized_streams(PARSED_STREAMS)),
    source_metadata={
        'timepix3_detector': SourceMetadata(title='Timepix3'),
        'multiblade_detector': SourceMetadata(title='Multiblade'),
        'he3_detector_bank0': SourceMetadata(title='He3 Bank 0'),
        'he3_detector_bank1': SourceMetadata(title='He3 Bank 1'),
        'ngem_detector': SourceMetadata(title='nGEM'),
        'orca_detector': SourceMetadata(title='ORCA'),
        'monitor_1': SourceMetadata(title='Beam Monitor'),
    },
)

instrument_registry.register(instrument)

#: Positions along the beam relative to the sample, in metres, as the geometry
#: file states them. Wavelength mode computes flight paths from these, but they
#: are unverified and mostly look like placeholders (no NICOS device reports
#: them yet, see #1356). The view descriptions show them to the user; a test
#: keeps this in sync with the geometry file.
GEOMETRY_FILE_Z_POSITIONS = {
    'source': -30.0,
    'monitor_1': -14.8,
    'he3_detector_bank0': 1.0,
    'he3_detector_bank1': 2.0,
    'multiblade_detector': 1.0,
    'ngem_detector': -1.0,
}


def _unverified_positions_warning(source_names: list[str]) -> str:
    positions = ', '.join(
        f'{instrument.source_metadata[name].title} at '
        f'{GEOMETRY_FILE_Z_POSITIONS[name]:+.1f} m'
        for name in source_names
    )
    return (
        '<b>Warning:</b> wavelength mode uses unverified positions from the'
        f' geometry file: {positions}, source at'
        f" {GEOMETRY_FILE_Z_POSITIONS['source']:+.1f} m. Positions are along"
        ' the beam relative to the sample. Treat wavelengths as approximate.'
    )


register_monitor_workflow_specs(
    instrument,
    monitor_names,
    extra_description=_unverified_positions_warning(monitor_names),
)

instrument.configure_detector_downsampling(
    'timepix3_detector',
    resolution=TIMEPIX3_IMAGE_RESOLUTION,
    source_resolution=TIMEPIX3_PANEL_RESOLUTION,
    reconfigurable=True,
)

instrument.add_logical_view(
    name='tbl_detector_timepix3',
    title='Timepix3 Detector',
    description=(
        f'{TIMEPIX3_IMAGE_RESOLUTION}x{TIMEPIX3_IMAGE_RESOLUTION} image,'
        ' downsampled from the streamed resolution as the events are'
        ' ingested. The streamed resolution is read from the event ids; a'
        ' readout reconfiguration restarts the cumulative image, since counts'
        ' taken before and after it are not commensurable.'
    ),
    source_names=['timepix3_detector'],
    transform=name_image_dims,
    # No wavelength mode: the geometry file holds positions for the full panel
    # resolution, not for the downsampled grid the events are ingested at.
    roi_support=True,
    device_outputs=DETECTOR_VIEW_DEVICES,
)

instrument.add_logical_view(
    name='multiblade_detector_view',
    title='Multiblade Detector',
    description=(
        'Counts folded into blade, wire, and strip dimensions<br><br>'
        + _unverified_positions_warning(['multiblade_detector'])
    ),
    source_names=['multiblade_detector'],
    wavelength=True,
    transform=get_multiblade_view,
    # ROI geometries are rectangles and polygons on a 2D screen; this view is 3D.
    roi_support=False,
    output_ndim=3,
    spectrum_view=SpectrumViewSpec(
        transform=get_multiblade_spectrum,
        output_dims=['blade', 'wire'],
        extra_description='Summed across strips, yielding per-blade, per-wire spectra.',
    ),
    device_outputs=DETECTOR_VIEW_DEVICES,
)

instrument.add_logical_view(
    name='he3_detector_view',
    title='He3 Detector',
    description=(
        'Combined view of both detector banks with tube and pixel axes<br><br>'
        + _unverified_positions_warning(['he3_detector_bank0', 'he3_detector_bank1'])
    ),
    source_names=['he3_detector_bank0', 'he3_detector_bank1'],
    wavelength=True,
    transform=get_he3_detector_view,
    roi_support=True,
    spectrum_view=SpectrumViewSpec(
        transform=get_he3_spectrum,
        output_dims=['tube', 'pixel'],
        extra_description='Per-tube, per-pixel spectra, without spatial reduction.',
    ),
    device_outputs=DETECTOR_VIEW_DEVICES,
)

instrument.add_logical_view(
    name='ngem_detector_view',
    title='NGEM Detector',
    description=(
        '2D detector counts view<br><br>'
        + _unverified_positions_warning(['ngem_detector'])
    ),
    source_names=['ngem_detector'],
    wavelength=True,
    transform=identity,
    reduction_dim='dim_0',
    roi_support=True,
    device_outputs=DETECTOR_VIEW_DEVICES,
)

orca_view_handle = instrument.register_spec(
    group=DETECTORS,
    name='tbl_area_detector_orca',
    version=1,
    title='Orca Detector',
    description='512x512 image downsampled from full resolution',
    source_names=['orca_detector'],
    # AreaDetectorView renders images only; it neither reads ROI requests nor
    # publishes readbacks, so it must not declare the ROI outputs.
    outputs=DetectorViewOutputsBase,
    device_outputs=DETECTOR_VIEW_DEVICES,
)
